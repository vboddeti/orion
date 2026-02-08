"""
Test FHE with Backbone + Linear only (no aggregation/normalization).

This isolates the backbone+linear stage to verify if levels are exhausted here.
"""
import sys
import os
import torch
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import orion
import orion.nn as on
from orion.core import scheme
from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4
from face_recognition.models.weight_loader import load_checkpoint_for_config
from face_recognition.utils.fhe_utils import unpack_fhe_tensor


class BackboneLinear(on.Module):
    """Wrapper that includes one backbone + its linear layer."""
    
    def __init__(self, backbone, linear):
        super().__init__()
        self.backbone = backbone
        self.linear = linear
    
    def forward(self, x):
        feat = self.backbone(x)
        out = self.linear(feat)
        return out
    
    def init_orion_params(self):
        self.backbone.init_orion_params()


def test_backbone_linear(image_path, net_idx=0):
    """Test backbone + linear for a single patch."""
    print(f"\n{'='*80}")
    print(f"Backbone + Linear FHE Test (Net {net_idx})")
    print(f"{'='*80}\n")
    
    # 1. Initialize Scheme
    print("Step 1: Initializing CKKS scheme...")
    orion.init_scheme("configs/pcnn-backbone.yml")
    
    # 2. Load Full Model & Extract Backbone+Linear
    print("Step 2: Loading model...")
    full_model = CryptoFaceNet4()
    load_checkpoint_for_config(full_model, input_size=64, verbose=False)
    
    # Create wrapper with backbone + linear
    model = BackboneLinear(full_model.nets[net_idx], full_model.linear[net_idx])
    model.eval()
    
    # 3. Load & Preprocess Image (extract one patch)
    print(f"Step 3: Loading image and extracting patch {net_idx}...")
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img_tensor = transform(img).unsqueeze(0)  # (1, 3, 64, 64)
    
    # Extract patch (32x32)
    P = 32
    h, w = net_idx // 2, net_idx % 2
    patch = img_tensor[:, :, h * P : (h + 1) * P, w * P : (w + 1) * P]
    print(f"  Patch shape: {patch.shape}")
    
    # 4. Cleartext Ground Truth
    print("Step 4: Running cleartext inference...")
    with torch.no_grad():
        out_clear = model(patch)
    print(f"  Cleartext output shape: {out_clear.shape}")
    print(f"  Cleartext output range: [{out_clear.min():.4f}, {out_clear.max():.4f}]")
    
    # 5. Fuse & Compile
    print("Step 5: Fusing and Compiling...")
    model.init_orion_params()
    orion.fit(model, patch)
    input_level = orion.compile(model)
    print(f"  Model compiled. Input level: {input_level}")
    
    # 6. Encrypt
    print("Step 6: Encrypting input...")
    vec_ptxt = orion.encode(patch, input_level)
    vec_ctxt = orion.encrypt(vec_ptxt)
    print(f"  Input encrypted. Level: {vec_ctxt.level()}")
    
    # 7. FHE Inference
    print("\nStep 7: Running FHE inference...")
    model.he()
    with torch.no_grad():
        out_ctxt = model(vec_ctxt)
    
    print(f"  FHE output level: {out_ctxt.level()}")
    
    # 8. Decrypt & Compare
    print("\nStep 8: Decrypting...")
    out_fhe = out_ctxt.decrypt().decode()
    print(f"  FHE output shape: {out_fhe.shape}")
    print(f"  FHE output range: [{out_fhe.min():.4f}, {out_fhe.max():.4f}]")
    
    # Unpack if needed
    if out_fhe.shape != out_clear.shape:
        out_fhe_unpacked = unpack_fhe_tensor(out_fhe, gap=1, target_shape=out_clear.shape)
    else:
        out_fhe_unpacked = out_fhe
    
    diff = (out_clear - out_fhe_unpacked).abs()
    mae = diff.mean().item()
    max_err = diff.max().item()
    
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"MAE: {mae:.6f}")
    print(f"Max Error: {max_err:.6f}")
    print(f"FHE Output Level: {out_ctxt.level()}")
    
    success = mae < 1.0 and out_ctxt.level() > 0
    if success:
        print(f"\n✓ SUCCESS: Backbone + Linear FHE inference works!")
    else:
        print(f"\n✗ FAILURE: Error too high or level exhausted")
    
    return success


if __name__ == "__main__":
    img_path = "/research/hal-datastage/datasets/original/LFW/lfw-mtcnn-aligned/John_Abizaid/John_Abizaid_0002.jpg"
    
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found")
        sys.exit(1)
    
    success = test_backbone_linear(img_path, net_idx=0)
    sys.exit(0 if success else 1)
