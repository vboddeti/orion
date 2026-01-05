"""
Compare Orion PCNN plaintext vs FHE inference using pre-trained checkpoints.

This script validates that FHE inference produces results close to plaintext
inference within acceptable numerical error bounds.
"""
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

sys.path.append('/research/hal-vishnu/code/orion-fhe')

import orion
from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4
from face_recognition.models.weight_loader import load_checkpoint_for_config


def load_lfw_image(image_path, input_size=64):
    """Load and preprocess LFW image."""
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(input_size, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    img_tensor = transform(img)
    return img_tensor.unsqueeze(0)


def compare_outputs(plaintext_out, fhe_out):
    """Compare plaintext and FHE outputs."""
    print(f"\n{'='*80}")
    print(f"OUTPUT COMPARISON")
    print(f"{'='*80}")

    print(f"\nPlaintext output:")
    print(f"  Shape: {plaintext_out.shape}")
    print(f"  Range: [{plaintext_out.min():.6f}, {plaintext_out.max():.6f}]")
    print(f"  Mean:  {plaintext_out.mean():.6f}")
    print(f"  Std:   {plaintext_out.std():.6f}")

    print(f"\nFHE output:")
    print(f"  Shape: {fhe_out.shape}")
    print(f"  Range: [{fhe_out.min():.6f}, {fhe_out.max():.6f}]")
    print(f"  Mean:  {fhe_out.mean():.6f}")
    print(f"  Std:   {fhe_out.std():.6f}")

    # Compute errors
    diff = (plaintext_out - fhe_out).abs()
    rel_diff = diff / (plaintext_out.abs() + 1e-8)

    mae = diff.mean().item()
    max_abs_error = diff.max().item()
    mean_rel_error = rel_diff.mean().item()
    max_rel_error = rel_diff.max().item()

    print(f"\nError metrics:")
    print(f"  MAE (Mean Absolute Error):    {mae:.6f}")
    print(f"  Max Absolute Error:           {max_abs_error:.6f}")
    print(f"  Mean Relative Error:          {mean_rel_error:.6f} ({mean_rel_error*100:.4f}%)")
    print(f"  Max Relative Error:           {max_rel_error:.6f} ({max_rel_error*100:.4f}%)")

    # Success criteria (typical for FHE inference)
    # MAE < 1.0 is considered good for FHE
    if mae < 1.0:
        print(f"\n✓ FHE INFERENCE SUCCESSFUL (MAE < 1.0)")
        return True
    else:
        print(f"\n✗ FHE INFERENCE FAILED (MAE >= 1.0)")
        return False


def main():
    print(f"\n{'='*80}")
    print(f"ORION PCNN: PLAINTEXT VS FHE COMPARISON")
    print(f"{'='*80}")

    # Check if config exists
    config_path = Path("configs/cryptoface_net4.yml")
    if not config_path.exists():
        print(f"\n✗ Config not found: {config_path}")
        print(f"Please create a config file for CryptoFaceNet4")
        return

    # Load test image
    lfw_dir = Path("/research/hal-datastage/datasets/original/LFW/lfw-mtcnn-aligned")
    person_dirs = sorted([d for d in lfw_dir.iterdir() if d.is_dir()])
    first_person = person_dirs[0]
    images = sorted(list(first_person.glob("*.jpg")) + list(first_person.glob("*.png")))
    image_path = images[0]

    print(f"\nTest image: {image_path}")
    img = load_lfw_image(image_path, input_size=64)
    print(f"Image shape: {img.shape}, range: [{img.min():.4f}, {img.max():.4f}]")

    # Initialize FHE scheme
    print(f"\n{'='*80}")
    print(f"Step 1: Initialize FHE Scheme")
    print(f"{'='*80}")
    orion.init_scheme(str(config_path))
    print(f"✓ Scheme initialized with config: {config_path}")

    # Load model with pre-trained weights
    print(f"\n{'='*80}")
    print(f"Step 2: Load Model with Pre-trained Weights")
    print(f"{'='*80}")
    model = CryptoFaceNet4()
    load_checkpoint_for_config(model, input_size=64, verbose=True)

    # Collect BatchNorm statistics (warmup)
    print(f"\n{'='*80}")
    print(f"Step 3: Warmup - Collect BatchNorm Statistics")
    print(f"{'='*80}")
    model.eval()
    with torch.no_grad():
        for i in range(20):
            _ = model(torch.randn(4, 3, 64, 64))
    print(f"✓ Collected statistics from 20 batches")

    # Fuse operations BEFORE fit
    print(f"\n{'='*80}")
    print(f"Step 4: Fuse HerPN Operations")
    print(f"{'='*80}")
    model.init_orion_params()
    print(f"✓ HerPN parameters initialized and fused")

    # Run plaintext inference
    print(f"\n{'='*80}")
    print(f"Step 5: Plaintext Inference")
    print(f"{'='*80}")
    model.eval()
    with torch.no_grad():
        plaintext_out = model(img)
    print(f"✓ Plaintext output shape: {plaintext_out.shape}")

    # Fit and compile for FHE
    print(f"\n{'='*80}")
    print(f"Step 6: Fit - Trace Model and Collect Statistics")
    print(f"{'='*80}")
    orion.fit(model, img)
    print(f"✓ Model traced and statistics collected")

    print(f"\n{'='*80}")
    print(f"Step 7: Compile - Assign Levels and Generate FHE Parameters")
    print(f"{'='*80}")
    input_level = orion.compile(model)
    print(f"✓ Model compiled, input_level = {input_level}")

    # Encode and encrypt
    print(f"\n{'='*80}")
    print(f"Step 8: Encode and Encrypt Input")
    print(f"{'='*80}")
    vec_ptxt = orion.encode(img, input_level)
    print(f"✓ Input encoded")
    vec_ctxt = orion.encrypt(vec_ptxt)
    print(f"✓ Input encrypted")

    # Run FHE inference
    print(f"\n{'='*80}")
    print(f"Step 9: FHE Inference")
    print(f"{'='*80}")
    model.he()  # Switch to FHE mode
    out_ctxt = model(vec_ctxt)
    print(f"✓ FHE inference completed")

    # Decrypt and decode
    print(f"\n{'='*80}")
    print(f"Step 10: Decrypt and Decode Output")
    print(f"{'='*80}")
    fhe_out = out_ctxt.decrypt().decode()
    print(f"✓ Output decrypted and decoded")
    print(f"✓ FHE output shape: {fhe_out.shape}")

    # Compare results
    success = compare_outputs(plaintext_out, fhe_out)

    if success:
        print(f"\n{'='*80}")
        print(f"✓ VALIDATION SUCCESSFUL")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print(f"✗ VALIDATION FAILED")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
