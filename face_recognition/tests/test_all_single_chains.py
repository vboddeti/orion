#!/usr/bin/env python3
"""
Test all 4 single chains (Backbone -> Linear -> Normalization) in parallel.
Calculates MAE and Relative Error for each chain to identify potential anomalies.
"""
import sys
import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from joblib import Parallel, delayed
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import orion
import orion.nn as on
from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4
from face_recognition.models.weight_loader import load_checkpoint_for_config
from face_recognition.utils.fhe_utils import unpack_fhe_tensor

# Define SingleChainModel locally to ensure it's pickleable for joblib
class SingleChainModel(on.Module):
    """Wrapper for Backbone -> Linear -> L2Norm chain."""
    
    def __init__(self, backbone, linear, normalization):
        super().__init__()
        self.backbone = backbone
        self.linear = linear
        self.normalization = normalization
    
    def forward(self, x):
        feat = self.backbone(x)
        linear_out = self.linear(feat)
        norm_out = self.normalization(linear_out)
        return norm_out
    
    def init_orion_params(self, fuse_bn_linear=True):
        self.backbone.init_orion_params()
        if fuse_bn_linear:
            self.backbone.fuse_bn_into_linear(self.linear)

def run_single_chain(net_idx, image_path):
    """Run a single chain test and return metrics."""
    try:
        print(f"[Net {net_idx}] Starting test...")
        
        # Initialize isolated scheme for this process
        # We must use a separate config or re-init to ensure clean state
        orion.init_scheme("configs/pcnn-backbone.yml")
        
        # Load Full Model
        full_model = CryptoFaceNet4()
        load_checkpoint_for_config(full_model, input_size=64, verbose=False)
        
        # Extract components
        backbone = full_model.nets[net_idx]
        linear = full_model.linear[net_idx]
        normalization = full_model.normalization
        
        # Create single chain model
        model = SingleChainModel(backbone, linear, normalization)
        model.eval()
        
        # Load Image
        img = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        img_tensor = transform(img).unsqueeze(0)
        
        # Extract patch
        P = 32
        h, w = net_idx // 2, net_idx % 2
        patch = img_tensor[:, :, h * P : (h + 1) * P, w * P : (w + 1) * P]
        
        # Cleartext Inference
        with torch.no_grad():
            model.he_mode = False
            out_clear = model(patch)
        
        # Compile & Fit
        model.init_orion_params()
        orion.fit(model, patch)
        input_level = orion.compile(model)
        
        # Encrypt
        vec_ptxt = orion.encode(patch, input_level)
        vec_ctxt = orion.encrypt(vec_ptxt)
        
        # FHE Inference
        # MUST call .he() to propagate mode to all submodules
        model.he()
        with torch.no_grad():
            out_ctxt = model(vec_ctxt)
        
        # Decrypt
        out_fhe = out_ctxt.decrypt().decode()
        
        # Metrics
        if out_fhe.shape != out_clear.shape:
             # Basic reshape attempt if direct unpack fails or isn't needed
             out_fhe = out_fhe.reshape(out_clear.shape)

        diff = (out_clear - out_fhe).abs()
        mae = diff.mean().item()
        max_err = diff.max().item()
        
        # Relative Error: norm(diff) / norm(clear)
        rel_err = torch.norm(out_clear - out_fhe) / torch.norm(out_clear)
        rel_err_val = rel_err.item()
        
        print(f"[Net {net_idx}] Completed. MAE: {mae:.6f}, RelErr: {rel_err_val:.6f}")
        
        return {
            "net_idx": net_idx,
            "mae": mae,
            "max_err": max_err,
            "rel_err": rel_err_val,
            "clear_min": out_clear.min().item(),
            "clear_max": out_clear.max().item(),
            "fhe_min": out_fhe.min().item(),
            "fhe_max": out_fhe.max().item(),
            "status": "SUCCESS"
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "net_idx": net_idx,
            "status": "FAILURE",
            "error": str(e)
        }

def main():
    img_path = "/research/hal-datastage/datasets/original/LFW/lfw-mtcnn-aligned/John_Abizaid/John_Abizaid_0002.jpg"
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        
    print(f"Running parallel tests for 4 nets using image: {img_path}")
    
    # Run in parallel
    results = Parallel(n_jobs=4)(delayed(run_single_chain)(i, img_path) for i in range(4))
    
    print("\n" + "="*80)
    print(f"{'Net':<5} | {'Status':<10} | {'MAE':<10} | {'Max Err':<10} | {'Rel Err':<10} | {'Range (Clear)':<20}")
    print("-" * 80)
    
    for res in sorted(results, key=lambda x: x['net_idx']):
        if res['status'] == 'SUCCESS':
            print(f"{res['net_idx']:<5} | {res['status']:<10} | {res['mae']:<10.6f} | {res['max_err']:<10.6f} | {res['rel_err']:<10.6f} | [{res['clear_min']:.3f}, {res['clear_max']:.3f}]")
        else:
            print(f"{res['net_idx']:<5} | {res['status']:<10} | {'-':<10} | {'-':<10} | {'-':<10} | {res['error']}")
            
    print("="*80)

if __name__ == "__main__":
    main()
