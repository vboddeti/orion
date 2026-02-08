
"""
Test FHE with Backbone + Linear + L2Norm (single chain).
Requested by user to isolate the issue: "backbone->linear->l2norm".

This removes aggregation complexity but keeps normalization depth.
"""
import sys
import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import orion
import orion.nn as on
from orion.core import scheme
from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4
from face_recognition.models.weight_loader import load_checkpoint_for_config
from face_recognition.utils.fhe_utils import unpack_fhe_tensor
from models.pcnn import L2NormPoly

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
        # Fuse BN into Linear to save 1 FHE level
        if fuse_bn_linear:
            self.backbone.fuse_bn_into_linear(self.linear)

def test_single_chain(image_path, net_idx=0):
    """Test single chain FHE execution."""
    print(f"\n{'='*80}")
    print(f"Single Chain FHE Test (Net {net_idx})")
    print(f"Backbone -> Linear -> L2NormPoly")
    print(f"{'='*80}\n")
    
    # 1. Initialize Scheme
    print("Step 1: Initializing CKKS scheme...")
    orion.init_scheme("configs/pcnn-backbone.yml")
    
    # 2. Load Full Model
    print("Step 2: Loading model components...")
    full_model = CryptoFaceNet4()
    load_checkpoint_for_config(full_model, input_size=64, verbose=False)
    
    # Extract components for net_idx
    backbone = full_model.nets[net_idx]
    linear = full_model.linear[net_idx]
    normalization = full_model.normalization
    
    # Create single chain model
    model = SingleChainModel(backbone, linear, normalization)
    model.eval()
    
    # 3. Load Image
    print(f"Step 3: Loading image patch {net_idx}...")
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
    print(f"  Patch shape: {patch.shape}")
    
    # 4. Cleartext Inference
    print("Step 4: Running cleartext inference...")
    with torch.no_grad():
        out_clear = model(patch)
    print(f"  Cleartext output range: [{out_clear.min():.4f}, {out_clear.max():.4f}]")
    
    # 5. Compile
    print("Step 5: Compiling...")
    model.init_orion_params()
    orion.fit(model, patch)
    input_level = orion.compile(model)
    print(f"  Model compiled. Input level: {input_level}")
    
    # 6. Encrypt
    print("Step 6: Encrypting...")
    vec_ptxt = orion.encode(patch, input_level)
    vec_ctxt = orion.encrypt(vec_ptxt)
    print(f"  Input encrypted. Level: {vec_ctxt.level()}")
    
    # 7. FHE Inference
    print("\nStep 7: Running FHE inference...")
    model.he()
    
    # Enable debug printing
    # scheme = orion.core.scheme.Scheme 
    
    with torch.no_grad():
        out_ctxt = model(vec_ctxt)
        
    print(f"  FHE output level: {out_ctxt.level()}")
    
    # 8. Decrypt
    print("\nStep 8: Decrypting...")
    out_fhe = out_ctxt.decrypt().decode()
    print(f"  FHE output range: [{out_fhe.min():.4f}, {out_fhe.max():.4f}]")
    
    # Compare
    if out_fhe.shape != out_clear.shape:
        out_fhe_unpacked = unpack_fhe_tensor(out_fhe, gap=1, target_shape=out_clear.shape)
    else:
        out_fhe_unpacked = out_fhe
        
    diff = (out_clear - out_fhe_unpacked).abs()
    mae = diff.mean().item()
    
    print(f"\nRESULTS:")
    print(f"MAE: {mae:.6f}")
    
    if mae < 1.0 and out_ctxt.level() >= 0:
        print("\n✓ SUCCESS")
        return True
    else:
        print("\n✗ FAILURE")
        return False

if __name__ == "__main__":
    img_path = "/research/hal-datastage/datasets/original/LFW/lfw-mtcnn-aligned/John_Abizaid/John_Abizaid_0002.jpg"
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found")
        sys.exit(1)
        
    test_single_chain(img_path)
