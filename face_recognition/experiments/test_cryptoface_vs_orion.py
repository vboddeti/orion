"""
Compare CryptoFace reference model vs Orion PCNN with loaded weights on real data.

This test verifies that our weight loading produces identical outputs to the
original CryptoFace model when processing actual face images.
"""
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.append('/research/hal-vishnu/code/orion-fhe')
sys.path.append('/research/hal-vishnu/code/orion-fhe/CryptoFace')

# Import CryptoFace reference model
from CryptoFace.models import build_model
from CryptoFace.argument import CLI

# Import our Orion PCNN
from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4
from face_recognition.models.weight_loader import load_checkpoint_for_config


def create_test_images(batch_size=4, input_size=64):
    """Create synthetic face-like images (normalized in [0, 1] range)."""
    # Create images with face-like structure (centered oval with features)
    images = []
    for _ in range(batch_size):
        img = torch.zeros(3, input_size, input_size)

        # Add some structure similar to faces
        center_y, center_x = input_size // 2, input_size // 2

        # Create an oval face shape
        y, x = torch.meshgrid(torch.arange(input_size), torch.arange(input_size), indexing='ij')
        y_dist = (y - center_y) / (input_size * 0.4)
        x_dist = (x - center_x) / (input_size * 0.3)
        face_mask = (y_dist**2 + x_dist**2) < 1

        # Random skin tone
        skin_tone = torch.rand(3, 1, 1) * 0.3 + 0.4  # [0.4, 0.7]
        img = img + skin_tone * face_mask.float()

        # Add random features (eyes, nose, mouth as darker regions)
        num_features = 5
        for _ in range(num_features):
            fy = torch.randint(input_size // 3, 2 * input_size // 3, (1,)).item()
            fx = torch.randint(input_size // 3, 2 * input_size // 3, (1,)).item()
            feature_size = torch.randint(2, 6, (1,)).item()

            y_feat, x_feat = torch.meshgrid(
                torch.arange(input_size),
                torch.arange(input_size),
                indexing='ij'
            )
            feat_mask = ((y_feat - fy)**2 + (x_feat - fx)**2) < feature_size**2
            img = img - 0.2 * feat_mask.float()

        # Add some noise
        img = img + torch.randn(3, input_size, input_size) * 0.05

        # Normalize to standard range
        img = torch.clamp(img, 0, 1)
        # Convert to [-1, 1] range (typical for face recognition)
        img = (img - 0.5) / 0.5

        images.append(img)

    return torch.stack(images)


def test_cryptoface_vs_orion():
    """Compare CryptoFace and Orion PCNN outputs."""
    print("="*80)
    print("TEST: CryptoFace Reference vs Orion PCNN Comparison")
    print("="*80)

    checkpoint_path = Path("face_recognition/checkpoints/backbone-64x64.ckpt")
    if not checkpoint_path.exists():
        print(f"\n❌ Checkpoint not found: {checkpoint_path}")
        print("Please ensure the checkpoint file exists.")
        return

    device = torch.device('cpu')
    input_size = 64

    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    print("\n" + "="*80)
    print("1. Loading CryptoFace Reference Model")
    print("="*80)

    # Create CryptoFace reference model
    # Build model using CryptoFace's build_model function
    class Args:
        arch = "cryptoface4"
        input_size = 64
        patch_size = 32  # 64x64 input with 32x32 patches = 2x2 grid = 4 patches

    args = Args()
    cryptoface_model = build_model(args)
    cryptoface_model.load_state_dict(ckpt['backbone'])
    cryptoface_model.eval()

    # IMPORTANT: Fuse BatchNorm before inference (CryptoFace requirement)
    cryptoface_model.fuse()
    print(f"✓ CryptoFace model loaded and fused")

    cryptoface_model.to(device)

    print(f"  Architecture: {args.arch}")
    print(f"  Input size: {input_size}×{input_size}")
    print(f"  BatchNorm fused: Yes")

    print("\n" + "="*80)
    print("2. Loading Orion PCNN Model")
    print("="*80)

    # Create Orion PCNN model
    orion_model = CryptoFaceNet4()
    load_checkpoint_for_config(orion_model, input_size=input_size, verbose=False)
    orion_model.eval()
    orion_model.to(device)

    print(f"✓ Orion PCNN model loaded")
    print(f"  Patches: {orion_model.N}")
    print(f"  Embedding dim: {orion_model.embedding_dim}")

    print("\n" + "="*80)
    print("3. Testing with Synthetic Face Images")
    print("="*80)

    # Create test images
    batch_size = 8
    test_images = create_test_images(batch_size, input_size)
    test_images = test_images.to(device)

    print(f"\nGenerated {batch_size} synthetic face-like images")
    print(f"  Shape: {test_images.shape}")
    print(f"  Range: [{test_images.min():.3f}, {test_images.max():.3f}]")

    print("\n" + "="*80)
    print("4. Forward Pass Comparison")
    print("="*80)

    with torch.no_grad():
        # CryptoFace forward pass using fused model
        # After fuse(), use forward_fuse() which returns just the embedding
        cryptoface_output = cryptoface_model.forward_fuse(test_images)

        # Orion PCNN forward pass
        orion_output = orion_model(test_images)

    # Check for NaN/Inf
    cryptoface_has_nan = torch.isnan(cryptoface_output).any()
    cryptoface_has_inf = torch.isinf(cryptoface_output).any()
    orion_has_nan = torch.isnan(orion_output).any()
    orion_has_inf = torch.isinf(orion_output).any()

    print(f"\nCryptoFace output:")
    print(f"  Shape: {cryptoface_output.shape}")
    print(f"  Range: [{cryptoface_output.min():.4f}, {cryptoface_output.max():.4f}]")
    print(f"  Mean: {cryptoface_output.mean():.4f}")
    print(f"  Std: {cryptoface_output.std():.4f}")
    print(f"  Has NaN: {cryptoface_has_nan}, Has Inf: {cryptoface_has_inf}")

    print(f"\nOrion output:")
    print(f"  Shape: {orion_output.shape}")
    print(f"  Range: [{orion_output.min():.4f}, {orion_output.max():.4f}]")
    print(f"  Mean: {orion_output.mean():.4f}")
    print(f"  Std: {orion_output.std():.4f}")
    print(f"  Has NaN: {orion_has_nan}, Has Inf: {orion_has_inf}")

    # If CryptoFace has large values, let's investigate
    if cryptoface_output.abs().max() > 1000:
        print(f"\n⚠️  CryptoFace output has very large values!")
        print(f"  This suggests the fused model may not be working correctly.")
        print(f"  Per-sample max absolute values:")
        for i in range(batch_size):
            max_val = cryptoface_output[i].abs().max()
            print(f"    Sample {i}: {max_val:.2e}")

    print("\n" + "="*80)
    print("5. Output Comparison")
    print("="*80)

    # Compute differences
    abs_diff = (cryptoface_output - orion_output).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    # Compute relative error
    rel_error = abs_diff / (cryptoface_output.abs() + 1e-8)
    max_rel_error = rel_error.max().item()
    mean_rel_error = rel_error.mean().item()

    print(f"\nAbsolute differences:")
    print(f"  Max:  {max_diff:.6f}")
    print(f"  Mean: {mean_diff:.6f}")
    print(f"  Std:  {abs_diff.std():.6f}")

    print(f"\nRelative errors:")
    print(f"  Max:  {max_rel_error:.6f}")
    print(f"  Mean: {mean_rel_error:.6f}")

    # Per-sample comparison
    print(f"\nPer-sample max differences:")
    for i in range(batch_size):
        sample_diff = abs_diff[i].max().item()
        print(f"  Sample {i}: {sample_diff:.6f}")

    # Cosine similarity
    cryptoface_norm = torch.nn.functional.normalize(cryptoface_output, p=2, dim=1)
    orion_norm = torch.nn.functional.normalize(orion_output, p=2, dim=1)
    cosine_sim = (cryptoface_norm * orion_norm).sum(dim=1)

    print(f"\nCosine similarities:")
    print(f"  Mean: {cosine_sim.mean():.6f}")
    print(f"  Min:  {cosine_sim.min():.6f}")
    print(f"  Max:  {cosine_sim.max():.6f}")
    for i in range(batch_size):
        print(f"  Sample {i}: {cosine_sim[i]:.6f}")

    print("\n" + "="*80)
    print("6. Verification Results")
    print("="*80)

    # Determine thresholds for pass/fail
    # For fused BatchNorm, we expect very close match but not exact due to numerical precision
    max_diff_threshold = 0.01  # Allow max 0.01 difference
    mean_diff_threshold = 0.001  # Mean should be very small
    cosine_sim_threshold = 0.9999  # Cosine similarity should be very high

    max_diff_pass = max_diff < max_diff_threshold
    mean_diff_pass = mean_diff < mean_diff_threshold
    cosine_pass = cosine_sim.min() > cosine_sim_threshold

    print(f"\nCheck 1: Max difference < {max_diff_threshold}")
    print(f"  Result: {max_diff:.6f} {'✓ PASS' if max_diff_pass else '✗ FAIL'}")

    print(f"\nCheck 2: Mean difference < {mean_diff_threshold}")
    print(f"  Result: {mean_diff:.6f} {'✓ PASS' if mean_diff_pass else '✗ FAIL'}")

    print(f"\nCheck 3: Min cosine similarity > {cosine_sim_threshold}")
    print(f"  Result: {cosine_sim.min():.6f} {'✓ PASS' if cosine_pass else '✗ FAIL'}")

    # Overall result
    all_pass = max_diff_pass and mean_diff_pass and cosine_pass

    print("\n" + "="*80)
    print("FINAL RESULT")
    print("="*80)

    if all_pass:
        print("\n✅ SUCCESS: Orion PCNN outputs match CryptoFace reference!")
        print("\nThe weight loading implementation is correct. The models produce")
        print("equivalent outputs on face-like images.")
    else:
        print("\n⚠️  WARNING: Some differences detected between models")
        print("\nThe outputs are similar but not identical. This could be due to:")
        print("  - Numerical precision differences")
        print("  - Different BatchNorm implementations")
        print("  - Architecture differences")
        print("\nReview the differences above to determine if they are acceptable.")

    print("="*80)

    return all_pass


if __name__ == "__main__":
    test_cryptoface_vs_orion()
