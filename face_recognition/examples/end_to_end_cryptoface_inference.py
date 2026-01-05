"""
End-to-End CryptoFace Checkpoint Loading and FHE Inference

This example demonstrates the complete workflow for:
1. Loading CryptoFace pre-trained checkpoints
2. Verifying all fusion is correct
3. Running cleartext inference
4. Running FHE inference
5. Comparing results

This addresses the two main issues with checkpoint loading:
1. BatchNorm fusion: All BatchNorms (final + per-layer HerPN) are fused into weights
2. Dynamic operations: All tensor reshaping/expansion happens at compile time, not runtime

Key Steps:
-----------
1. Load model and checkpoint weights
2. Warmup: Collect BatchNorm statistics (20 forward passes)
3. Fusion: Call init_orion_params() to fuse HerPN layers
4. Verification: Verify all fusion is correct
5. Cleartext: Run cleartext inference
6. FHE Compilation: Fit and compile for FHE
7. FHE Inference: Encrypt, infer, decrypt
8. Comparison: Verify MAE < 1.0
"""

import sys
from pathlib import Path
import torch
import numpy as np

sys.path.append('/research/hal-vishnu/code/orion-fhe')

import orion
from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4, CryptoFaceNet9, CryptoFaceNet16
from face_recognition.models.weight_loader import load_checkpoint_for_config
from face_recognition.utils.checkpoint_verification import CheckpointVerifier


def run_end_to_end_example(
    model_name='net4',
    config_path='configs/cryptoface_net4.yml',
    checkpoint_dir='face_recognition/checkpoints',
    test_single_backbone=False,
    verbose=True
):
    """
    Run complete end-to-end workflow for CryptoFace checkpoint loading and FHE inference.

    Args:
        model_name: One of 'net4', 'net9', 'net16'
        config_path: Path to CKKS config file
        checkpoint_dir: Directory containing checkpoints
        test_single_backbone: If True, test single backbone instead of full model
        verbose: Print detailed progress

    Returns:
        Dictionary with results and metrics
    """
    results = {}

    # ==============================================================================
    # STEP 1: Create model and load checkpoint
    # ==============================================================================
    if verbose:
        print(f"\n{'='*80}")
        print(f"STEP 1: Loading Model and Checkpoint")
        print(f"{'='*80}")

    # Create model
    model_map = {
        'net4': (CryptoFaceNet4, 64),
        'net9': (CryptoFaceNet9, 96),
        'net16': (CryptoFaceNet16, 128),
    }

    if model_name not in model_map:
        raise ValueError(f"Invalid model_name: {model_name}. Choose from {list(model_map.keys())}")

    model_cls, input_size = model_map[model_name]
    model = model_cls()

    if verbose:
        print(f"✓ Created {model_name.upper()} model ({input_size}×{input_size} input, {model.N} patches)")

    # Load checkpoint weights
    load_checkpoint_for_config(model, input_size=input_size, verbose=verbose)

    if verbose:
        print(f"✓ Loaded checkpoint for {input_size}×{input_size} input")

    # ==============================================================================
    # STEP 2: Fusion - Initialize HerPN Parameters
    # ==============================================================================
    if verbose:
        print(f"\n{'='*80}")
        print(f"STEP 2: Fusing HerPN Operations")
        print(f"{'='*80}")
        print("This fuses BatchNorm layers into HerPN activations.")
        print("CRITICAL: This must happen BEFORE orion.fit() for correct level assignment!")
        print("")
        print("NOTE: Skipping warmup phase - checkpoint already contains BatchNorm statistics.")
        print("      Warmup is only needed when training from scratch, not when loading checkpoints.")
        print("      Running unfused forward pass can cause NaN due to BatchNorm over constant tensors.")

    model.init_orion_params()

    if verbose:
        print(f"✓ HerPN fusion complete")
        print(f"  - All BatchNorm layers fused into quadratic HerPN operations")
        print(f"  - Scale factors (w2) absorbed into conv weights")
        print(f"  - Shortcut scaling properly configured")

    # ==============================================================================
    # STEP 3: Verification - Check All Fusion is Correct
    # ==============================================================================
    if verbose:
        print(f"\n{'='*80}")
        print(f"STEP 3: Verifying Checkpoint Loading and Fusion")
        print(f"{'='*80}")

    verifier = CheckpointVerifier(model, verbose=verbose)
    all_checks_passed = verifier.verify_all()

    if not all_checks_passed:
        print(f"\n✗ VERIFICATION FAILED - Cannot proceed to FHE inference")
        return {'success': False, 'verification_passed': False}

    results['verification_passed'] = True

    # ==============================================================================
    # STEP 4: Cleartext Inference
    # ==============================================================================
    if verbose:
        print(f"\n{'='*80}")
        print(f"STEP 4: Cleartext Inference")
        print(f"{'='*80}")

    # Create test input
    test_input = torch.randn(1, 3, input_size, input_size)

    if verbose:
        print(f"Test input shape: {test_input.shape}")
        print(f"Test input range: [{test_input.min():.4f}, {test_input.max():.4f}]")

    # For testing, we can either test the full model or a single backbone
    if test_single_backbone:
        # Extract first patch and test just the backbone
        patch = test_input[:, :, :32, :32]
        test_model = model.nets[0]  # First backbone
        test_input_for_fhe = patch

        if verbose:
            print(f"\n⚠ Testing SINGLE BACKBONE (patch 0) for faster demonstration")
            print(f"  Patch shape: {patch.shape}")
    else:
        test_model = model
        test_input_for_fhe = test_input

        if verbose:
            print(f"\nTesting FULL MODEL ({model.N} patches)")

    # Run cleartext inference
    model.eval()
    with torch.no_grad():
        cleartext_output = test_model(test_input_for_fhe)

    if verbose:
        print(f"\n✓ Cleartext inference complete")
        print(f"  Output shape: {cleartext_output.shape}")
        print(f"  Output range: [{cleartext_output.min():.6f}, {cleartext_output.max():.6f}]")
        print(f"  Output mean: {cleartext_output.mean():.6f}")
        print(f"  Output std: {cleartext_output.std():.6f}")

    results['cleartext_output'] = cleartext_output

    # ==============================================================================
    # STEP 5: FHE Compilation
    # ==============================================================================
    if verbose:
        print(f"\n{'='*80}")
        print(f"STEP 5: FHE Compilation (Fit and Compile)")
        print(f"{'='*80}")

    # Initialize scheme
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"\n✗ Config file not found: {config_path}")
        print(f"  Please create config file with appropriate CKKS parameters")
        return {'success': False, 'verification_passed': True, 'config_missing': True}

    if verbose:
        print(f"Initializing FHE scheme from: {config_path}")

    orion.init_scheme(str(config_path))

    if verbose:
        print(f"✓ FHE scheme initialized")

    # Fit: Trace model and collect statistics
    if verbose:
        print(f"\nFitting model (tracing computation graph)...")

    orion.fit(test_model, test_input_for_fhe)

    if verbose:
        print(f"✓ Model traced")

    # Compile: Assign levels and generate FHE parameters
    if verbose:
        print(f"\nCompiling model (level assignment and parameter generation)...")

    input_level = orion.compile(test_model)

    if verbose:
        print(f"✓ Model compiled")
        print(f"  Input level: {input_level}")

    results['input_level'] = input_level

    # ==============================================================================
    # STEP 6: FHE Inference
    # ==============================================================================
    if verbose:
        print(f"\n{'='*80}")
        print(f"STEP 6: FHE Inference (Encrypt → Infer → Decrypt)")
        print(f"{'='*80}")

    # Encode and encrypt
    if verbose:
        print(f"Encoding and encrypting input...")

    vec_ptxt = orion.encode(test_input_for_fhe, input_level)
    vec_ctxt = orion.encrypt(vec_ptxt)

    if verbose:
        print(f"✓ Input encrypted")

    # Switch to FHE mode
    test_model.he()

    # Run FHE inference
    if verbose:
        print(f"\nRunning FHE inference...")

    out_ctxt = test_model(vec_ctxt)

    if verbose:
        print(f"✓ FHE inference complete")

    # Decrypt and decode
    if verbose:
        print(f"\nDecrypting and decoding output...")

    fhe_output = out_ctxt.decrypt().decode()

    if verbose:
        print(f"✓ Output decrypted")
        print(f"  FHE output shape: {fhe_output.shape}")
        print(f"  FHE output range: [{fhe_output.min():.6f}, {fhe_output.max():.6f}]")

    results['fhe_output'] = fhe_output

    # ==============================================================================
    # STEP 7: Comparison and Validation
    # ==============================================================================
    if verbose:
        print(f"\n{'='*80}")
        print(f"STEP 7: Comparing Cleartext vs FHE Results")
        print(f"{'='*80}")

    # Compute error metrics
    diff = (cleartext_output - fhe_output).abs()
    rel_diff = diff / (cleartext_output.abs() + 1e-8)

    mae = diff.mean().item()
    max_abs_error = diff.max().item()
    mean_rel_error = rel_diff.mean().item()

    if verbose:
        print(f"\nError Metrics:")
        print(f"  MAE (Mean Absolute Error):    {mae:.6f}")
        print(f"  Max Absolute Error:           {max_abs_error:.6f}")
        print(f"  Mean Relative Error:          {mean_rel_error:.4%}")

    results['mae'] = mae
    results['max_abs_error'] = max_abs_error
    results['mean_rel_error'] = mean_rel_error

    # Success criteria
    success = mae < 1.0

    if verbose:
        print(f"\n{'='*80}")
        if success:
            print(f"✓ FHE INFERENCE SUCCESSFUL!")
            print(f"  MAE ({mae:.6f}) < 1.0 threshold")
        else:
            print(f"✗ FHE INFERENCE FAILED")
            print(f"  MAE ({mae:.6f}) >= 1.0 threshold")
        print(f"{'='*80}")

    results['success'] = success

    # ==============================================================================
    # Summary
    # ==============================================================================
    if verbose:
        print(f"\n{'='*80}")
        print(f"SUMMARY")
        print(f"{'='*80}")
        print(f"Model: {model_name.upper()}")
        print(f"Input size: {input_size}×{input_size}")
        print(f"Patches: {model.N}")
        print(f"Embedding dim: {model.embedding_dim}")
        print(f"")
        print(f"Verification: {'✓ PASSED' if results['verification_passed'] else '✗ FAILED'}")
        print(f"FHE Inference: {'✓ SUCCESS' if success else '✗ FAILED'}")
        print(f"MAE: {mae:.6f}")
        print(f"{'='*80}\n")

    return results


def main():
    """Main entry point for end-to-end example."""
    import argparse

    parser = argparse.ArgumentParser(description='CryptoFace End-to-End FHE Inference')
    parser.add_argument('--model', type=str, default='net4',
                       choices=['net4', 'net9', 'net16'],
                       help='Model variant to test')
    parser.add_argument('--config', type=str, default='configs/cryptoface_net4.yml',
                       help='Path to CKKS config file')
    parser.add_argument('--single-backbone', action='store_true',
                       help='Test single backbone instead of full model (faster)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')

    args = parser.parse_args()

    results = run_end_to_end_example(
        model_name=args.model,
        config_path=args.config,
        test_single_backbone=args.single_backbone,
        verbose=not args.quiet
    )

    # Exit with appropriate code
    if results.get('success', False):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
