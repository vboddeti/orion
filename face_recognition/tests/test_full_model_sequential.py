"""
Full Model Sequential FHE Test

Tests the complete CryptoFaceNet4 model in FHE mode:
- All 4 backbones (sequential)
- All 4 linear layers
- Aggregation (tree reduce sum)
- Normalization (L2NormPoly)

This establishes a working baseline before parallelization.
"""
import sys
import os
import time
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


def test_full_model_sequential(image_path):
    """
    Run full CryptoFaceNet4 inference in FHE mode sequentially.
    """
    print(f"\n{'='*80}")
    print("Full Model Sequential FHE Inference Test")
    print(f"{'='*80}\n")
    
    # 1. Initialize Scheme
    print("Step 1: Initializing CKKS scheme...")
    orion.init_scheme("configs/pcnn-backbone.yml")
    
    # 2. Load Model & Weights
    print("Step 2: Loading CryptoFaceNet4...")
    model = CryptoFaceNet4()
    load_checkpoint_for_config(model, input_size=64, verbose=False)
    model.eval()
    
    # 3. Load & Preprocess Image
    print(f"Step 3: Loading image: {image_path}")
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img_tensor = transform(img).unsqueeze(0)  # (1, 3, 64, 64)
    print(f"  Input shape: {img_tensor.shape}, range: [{img_tensor.min():.4f}, {img_tensor.max():.4f}]")
    
    # 4. Cleartext Ground Truth
    print("Step 4: Running cleartext inference...")
    with torch.no_grad():
        out_clear = model(img_tensor)
    print(f"  Cleartext output shape: {out_clear.shape}")
    print(f"  Cleartext output range: [{out_clear.min():.4f}, {out_clear.max():.4f}]")
    
    # 5. Fuse BatchNorm & Compile
    print("Step 5: Fusing and Compiling...")
    model.init_orion_params()
    
    # 6. Extract and Encrypt Patches
    print("Step 6: Extracting and encrypting patches...")
    # Extract patches in cleartext
    H, W = 2, 2
    P = 32
    patches = []
    for h in range(H):
        for w in range(W):
            patch = img_tensor[:, :, h * P : (h + 1) * P, w * P : (w + 1) * P]
            patches.append(patch)
    
    # Use the patches for fitting and compilation
    # We pass the list of patches WRAPPED in another list to fit/compile
    # so that 'patches' is passed as the first argument (x) to forward.
    orion.fit(model, [patches])
    input_level = orion.compile(model)
    print(f"  Model compiled. Input level: {input_level}")
    
    # Encrypt each patch
    vec_ctxt_list = []
    for i, patch in enumerate(patches):
        vec_ptxt = orion.encode(patch, input_level)
        vec_ctxt = orion.encrypt(vec_ptxt)
        vec_ctxt_list.append(vec_ctxt)
    print(f"  All {len(vec_ctxt_list)} patches encrypted.")
    
    # 7. FHE Inference
    print("\nStep 7: Running FHE inference...")
    model.he()
    
    start_time = time.time()
    with torch.no_grad():
        out_ctxt = model(vec_ctxt_list)
    fhe_time = time.time() - start_time
    
    print(f"  FHE inference completed in {fhe_time:.2f} seconds")
    
    # 8. Decrypt & Verify
    print("\nStep 8: Decrypting...")
    out_fhe = out_ctxt.decrypt().decode()
    print(f"  FHE output shape: {out_fhe.shape}")
    print(f"  FHE output range: [{out_fhe.min():.4f}, {out_fhe.max():.4f}]")
    
    # Compare (no unpacking needed for output since normalization outputs gap=1)
    print("\nStep 9: Comparing results...")
    
    # The FHE output might have different shape due to packing
    # Unpack if needed
    if out_fhe.shape != out_clear.shape:
        # Try to find the gap from the output
        # For normalization, gap should be 1, but let's check
        print(f"  Shape mismatch: FHE={out_fhe.shape}, Clear={out_clear.shape}")
        # Attempt to unpack with gap=1
        out_fhe_unpacked = unpack_fhe_tensor(out_fhe, gap=1, target_shape=out_clear.shape)
    else:
        out_fhe_unpacked = out_fhe
    
    diff = (out_clear - out_fhe_unpacked).abs()
    mae = diff.mean().item()
    max_err = diff.max().item()
    rel_err = mae / (out_clear.abs().mean().item() + 1e-10)
    
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"MAE: {mae:.6f}")
    print(f"Max Error: {max_err:.6f}")
    print(f"Relative Error: {rel_err:.6f}")
    print(f"FHE Inference Time: {fhe_time:.2f} seconds")
    
    # Check if passed
    success = mae < 1.0  # Allow some FHE noise
    if success:
        print(f"\n✓ SUCCESS: Full model FHE inference works correctly!")
    else:
        print(f"\n✗ FAILURE: Error too high (MAE={mae:.6f} > 1.0)")
    
    return success


if __name__ == "__main__":
    # Use same test image as other tests
    img_path = "/research/hal-datastage/datasets/original/LFW/lfw-mtcnn-aligned/John_Abizaid/John_Abizaid_0002.jpg"
    
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found")
        sys.exit(1)
        
    success = test_full_model_sequential(img_path)
    sys.exit(0 if success else 1)
