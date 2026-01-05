"""
Verify Weight Loading Correctness

Compares outputs between:
1. Original CryptoFace PCNN model
2. Orion CryptoFacePCNN model

Both models load the same checkpoint and process the same random input.
We verify that outputs match within numerical tolerance.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add paths
sys.path.append('/research/hal-vishnu/code/orion-fhe')
sys.path.append('/research/hal-vishnu/code/orion-fhe/CryptoFace')

# Import CryptoFace model
from CryptoFace.models import PatchCNN as CryptoFacePatchCNN

# Import our Orion model
from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4, CryptoFaceNet9, CryptoFaceNet16
from face_recognition.models.weight_loader import load_checkpoint_for_config


def verify_model_outputs(input_size=64, seed=42, verbose=True):
    """
    Verify that CryptoFace and Orion models produce identical outputs.

    Args:
        input_size: Input image size (64, 96, or 128)
        seed: Random seed for reproducibility
        verbose: Print detailed comparison

    Returns:
        bool: True if outputs match within tolerance
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"VERIFICATION: CryptoFace vs Orion CryptoFacePCNN ({input_size}×{input_size})")
        print(f"{'='*80}")

    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Determine patch size and model
    patch_size = 32
    checkpoint_map = {
        64: ("backbone-64x64.ckpt", CryptoFaceNet4, 4),
        96: ("backbone-96x96.ckpt", CryptoFaceNet9, 9),
        128: ("backbone-128x128.ckpt", CryptoFaceNet16, 16),
    }

    if input_size not in checkpoint_map:
        raise ValueError(f"Invalid input_size: {input_size}. Must be 64, 96, or 128.")

    checkpoint_name, orion_model_fn, num_patches = checkpoint_map[input_size]
    checkpoint_path = f"face_recognition/checkpoints/{checkpoint_name}"

    # Step 1: Load CryptoFace model
    if verbose:
        print(f"\n{'='*80}")
        print(f"Step 1: Loading CryptoFace original model...")
        print(f"{'='*80}")

    cryptoface_model = CryptoFacePatchCNN(input_size=input_size, patch_size=patch_size)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cryptoface_model.load_state_dict(ckpt['backbone'], strict=False)
    cryptoface_model.eval()

    # Fuse BatchNorm (required for inference)
    cryptoface_model.fuse()

    if verbose:
        print(f"✓ Loaded CryptoFace model: {num_patches} patches")

    # Step 2: Load Orion model
    if verbose:
        print(f"\n{'='*80}")
        print(f"Step 2: Loading Orion CryptoFacePCNN model...")
        print(f"{'='*80}")

    orion_model = orion_model_fn()
    load_checkpoint_for_config(orion_model, input_size=input_size, verbose=False)

    # CRITICAL: Fuse BatchNorm into HerPN operations (equivalent to CryptoFace's fuse())
    orion_model.init_orion_params()
    orion_model.eval()

    if verbose:
        print(f"✓ Loaded Orion model: {num_patches} patches")

    # Step 3: Generate random input with realistic preprocessing
    # Real face images after Normalize(mean=0.5, std=0.5) have Gaussian distribution
    # centered near 0 with std around 0.3, clipped to [-1, 1]
    if verbose:
        print(f"\n{'='*80}")
        print(f"Step 3: Generating random input (realistic preprocessing)...")
        print(f"{'='*80}")

    torch.manual_seed(seed)  # Reset seed for consistent input
    batch_size = 4
    test_input = torch.randn(batch_size, 3, input_size, input_size) * 0.3
    test_input = torch.clamp(test_input, -1, 1)  # Clip to [-1, 1]

    if verbose:
        print(f"Input shape: {test_input.shape}")
        print(f"Input range: [{test_input.min():.4f}, {test_input.max():.4f}]")
        print(f"Input mean: {test_input.mean():.4f}, std: {test_input.std():.4f}")

    # Step 4: Forward pass through CryptoFace model
    if verbose:
        print(f"\n{'='*80}")
        print(f"Step 4: Running CryptoFace model...")
        print(f"{'='*80}")

    with torch.no_grad():
        cryptoface_output = cryptoface_model.forward_fuse(test_input)

    if verbose:
        print(f"CryptoFace output shape: {cryptoface_output.shape}")
        print(f"CryptoFace output range: [{cryptoface_output.min():.6f}, {cryptoface_output.max():.6f}]")
        print(f"CryptoFace output mean: {cryptoface_output.mean():.6f}")
        print(f"CryptoFace output std:  {cryptoface_output.std():.6f}")

    # Step 5: Forward pass through Orion model
    if verbose:
        print(f"\n{'='*80}")
        print(f"Step 5: Running Orion model...")
        print(f"{'='*80}")

    with torch.no_grad():
        orion_output = orion_model(test_input)

    if verbose:
        print(f"Orion output shape: {orion_output.shape}")
        print(f"Orion output range: [{orion_output.min():.6f}, {orion_output.max():.6f}]")
        print(f"Orion output mean: {orion_output.mean():.6f}")
        print(f"Orion output std:  {orion_output.std():.6f}")

    # Step 6: Compare outputs
    if verbose:
        print(f"\n{'='*80}")
        print(f"Step 6: Comparing outputs...")
        print(f"{'='*80}")

    # Convert to numpy for easier comparison
    cryptoface_np = cryptoface_output.numpy()
    orion_np = orion_output.numpy()

    # Compute differences
    abs_diff = np.abs(cryptoface_np - orion_np)
    rel_diff = abs_diff / (np.abs(cryptoface_np) + 1e-8)

    max_abs_diff = abs_diff.max()
    mean_abs_diff = abs_diff.mean()
    max_rel_diff = rel_diff.max()
    mean_rel_diff = rel_diff.mean()

    # Cosine similarity per sample
    cosine_sims = []
    for i in range(batch_size):
        cos_sim = np.dot(cryptoface_np[i], orion_np[i]) / (
            np.linalg.norm(cryptoface_np[i]) * np.linalg.norm(orion_np[i])
        )
        cosine_sims.append(cos_sim)
    mean_cosine_sim = np.mean(cosine_sims)

    if verbose:
        print(f"\nAbsolute Differences:")
        print(f"  Max:  {max_abs_diff:.8f}")
        print(f"  Mean: {mean_abs_diff:.8f}")
        print(f"\nRelative Differences:")
        print(f"  Max:  {max_rel_diff:.8f}")
        print(f"  Mean: {mean_rel_diff:.8f}")
        print(f"\nCosine Similarity (per sample):")
        for i, cos_sim in enumerate(cosine_sims):
            print(f"  Sample {i}: {cos_sim:.10f}")
        print(f"  Mean: {mean_cosine_sim:.10f}")

    # Determine if outputs match
    # Use lenient thresholds due to potential numerical differences
    tolerance_abs = 1e-4  # Absolute difference tolerance
    tolerance_cosine = 0.9999  # Cosine similarity threshold

    abs_match = max_abs_diff < tolerance_abs
    cosine_match = mean_cosine_sim > tolerance_cosine

    if verbose:
        print(f"\n{'='*80}")
        print(f"VERIFICATION RESULT")
        print(f"{'='*80}")
        print(f"Absolute difference < {tolerance_abs}: {abs_match} {'✓' if abs_match else '✗'}")
        print(f"Cosine similarity > {tolerance_cosine}: {cosine_match} {'✓' if cosine_match else '✗'}")

        if abs_match and cosine_match:
            print(f"\n{'='*80}")
            print(f"✓✓✓ VERIFICATION PASSED! ✓✓✓")
            print(f"{'='*80}")
            print(f"Orion model produces identical outputs to CryptoFace model!")
        else:
            print(f"\n{'='*80}")
            print(f"⚠ VERIFICATION FAILED")
            print(f"{'='*80}")
            print(f"Outputs differ beyond tolerance.")
            if not abs_match:
                print(f"  Max absolute diff {max_abs_diff:.8f} exceeds {tolerance_abs}")
            if not cosine_match:
                print(f"  Mean cosine sim {mean_cosine_sim:.10f} below {tolerance_cosine}")

    return abs_match and cosine_match


def verify_all_models():
    """Verify all three model variants."""
    print(f"\n{'#'*80}")
    print(f"# COMPREHENSIVE VERIFICATION: All CryptoFace Models")
    print(f"{'#'*80}")

    results = {}
    for input_size in [64, 96, 128]:
        try:
            passed = verify_model_outputs(input_size=input_size, verbose=True)
            results[input_size] = passed
        except Exception as e:
            print(f"\n✗ Error verifying {input_size}×{input_size}: {e}")
            results[input_size] = False

    # Summary
    print(f"\n{'#'*80}")
    print(f"# VERIFICATION SUMMARY")
    print(f"{'#'*80}")
    for input_size, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"CryptoFaceNet ({input_size}×{input_size}): {status}")

    all_passed = all(results.values())
    if all_passed:
        print(f"\n{'='*80}")
        print(f"✓✓✓ ALL MODELS VERIFIED SUCCESSFULLY! ✓✓✓")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print(f"⚠ SOME MODELS FAILED VERIFICATION")
        print(f"{'='*80}")

    return all_passed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify weight loading correctness")
    parser.add_argument("--input-size", type=int, choices=[64, 96, 128],
                        help="Input size to verify (64, 96, or 128). If not specified, verifies all.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if args.input_size:
        # Verify single model
        passed = verify_model_outputs(input_size=args.input_size, seed=args.seed, verbose=True)
        sys.exit(0 if passed else 1)
    else:
        # Verify all models
        all_passed = verify_all_models()
        sys.exit(0 if all_passed else 1)
