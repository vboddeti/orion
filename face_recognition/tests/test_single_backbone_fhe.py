"""
Test single backbone (patch network) with cleartext vs FHE inference.

This tests just ONE of the parallel patch CNNs to simplify debugging
before testing the full multi-patch model.
"""
import sys
import torch
from pathlib import Path

sys.path.append('/research/hal-vishnu/code/orion-fhe')

import orion
from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4
from face_recognition.models.weight_loader import load_checkpoint_for_config


def test_single_backbone_inference(
    patch_idx=0,
    config_path='configs/cryptoface_net4.yml',
    verbose=True
):
    """
    Test cleartext vs FHE inference for a single backbone.

    Args:
        patch_idx: Which patch backbone to test (0-3 for Net4)
        config_path: Path to CKKS config
        verbose: Print detailed output

    Returns:
        dict with test results
    """
    results = {}

    if verbose:
        print(f"\n{'='*80}")
        print(f"SINGLE BACKBONE TEST (Patch {patch_idx})")
        print(f"{'='*80}")

    # ==============================================================================
    # STEP 1: Load Model and Extract Single Backbone
    # ==============================================================================
    if verbose:
        print(f"\nSTEP 1: Loading Model and Extracting Backbone {patch_idx}")
        print(f"{'='*80}")

    # Create full model and load checkpoint
    full_model = CryptoFaceNet4()
    load_checkpoint_for_config(full_model, input_size=64, verbose=False)

    if verbose:
        print(f"✓ Loaded checkpoint")

    # Fuse HerPN
    full_model.init_orion_params()

    if verbose:
        print(f"✓ Fused HerPN operations")

    # Extract single backbone
    backbone = full_model.nets[patch_idx]

    if verbose:
        print(f"✓ Extracted backbone for patch {patch_idx}")
        print(f"  Input:  32×32 RGB patch")
        print(f"  Output: 256-dim feature vector")

    # ==============================================================================
    # STEP 2: Create Test Input (Single Patch)
    # ==============================================================================
    if verbose:
        print(f"\nSTEP 2: Creating Test Input")
        print(f"{'='*80}")

    # Create random 32x32 patch (single patch input)
    torch.manual_seed(42)  # For reproducibility
    test_patch = torch.randn(1, 3, 32, 32)

    if verbose:
        print(f"✓ Created test patch: {test_patch.shape}")
        print(f"  Range: [{test_patch.min():.4f}, {test_patch.max():.4f}]")
        print(f"  Mean: {test_patch.mean():.4f}, Std: {test_patch.std():.4f}")

    # ==============================================================================
    # STEP 3: Cleartext Inference
    # ==============================================================================
    if verbose:
        print(f"\nSTEP 3: Cleartext Inference")
        print(f"{'='*80}")

    backbone.eval()
    with torch.no_grad():
        cleartext_output = backbone(test_patch)

    if verbose:
        print(f"✓ Cleartext inference complete")
        print(f"  Output shape: {cleartext_output.shape}")
        print(f"  Output range: [{cleartext_output.min():.6f}, {cleartext_output.max():.6f}]")
        print(f"  Output mean: {cleartext_output.mean():.6f}")
        print(f"  Output std: {cleartext_output.std():.6f}")

    results['cleartext_output'] = cleartext_output
    results['cleartext_shape'] = cleartext_output.shape

    # ==============================================================================
    # STEP 4: Initialize FHE Scheme
    # ==============================================================================
    if verbose:
        print(f"\nSTEP 4: Initializing FHE Scheme")
        print(f"{'='*80}")

    config_path = Path(config_path)
    if not config_path.exists():
        print(f"✗ Config file not found: {config_path}")
        return {'success': False, 'error': 'config_missing'}

    orion.init_scheme(str(config_path))

    if verbose:
        print(f"✓ FHE scheme initialized from: {config_path}")

    # ==============================================================================
    # STEP 5: Fit and Compile for FHE
    # ==============================================================================
    if verbose:
        print(f"\nSTEP 5: Tracing and Compiling Model")
        print(f"{'='*80}")

    # Fit: Trace computation graph
    if verbose:
        print(f"  Tracing model...")

    orion.fit(backbone, test_patch)

    if verbose:
        print(f"  ✓ Model traced")

    # Compile: Assign levels and generate FHE parameters
    if verbose:
        print(f"  Compiling model...")

    input_level = orion.compile(backbone)

    if verbose:
        print(f"  ✓ Model compiled")
        print(f"  Input level: {input_level}")

    results['input_level'] = input_level

    # ==============================================================================
    # STEP 6: FHE Inference
    # ==============================================================================
    if verbose:
        print(f"\nSTEP 6: FHE Inference")
        print(f"{'='*80}")

    # Encode and encrypt
    if verbose:
        print(f"  Encoding input...")

    vec_ptxt = orion.encode(test_patch, input_level)

    if verbose:
        print(f"  ✓ Input encoded")
        print(f"  Encrypting input...")

    vec_ctxt = orion.encrypt(vec_ptxt)

    if verbose:
        print(f"  ✓ Input encrypted")

    # Switch to FHE mode
    backbone.he()

    # Run FHE inference
    if verbose:
        print(f"  Running FHE inference...")

    out_ctxt = backbone(vec_ctxt)

    if verbose:
        print(f"  ✓ FHE inference complete")

    # Decrypt and decode
    if verbose:
        print(f"  Decrypting output...")

    fhe_output = out_ctxt.decrypt().decode()

    if verbose:
        print(f"  ✓ Output decrypted")
        print(f"  FHE output shape: {fhe_output.shape}")
        print(f"  FHE output range: [{fhe_output.min():.6f}, {fhe_output.max():.6f}]")

    results['fhe_output'] = fhe_output
    results['fhe_shape'] = fhe_output.shape

    # ==============================================================================
    # STEP 7: Compare Results
    # ==============================================================================
    if verbose:
        print(f"\nSTEP 7: Comparing Cleartext vs FHE")
        print(f"{'='*80}")

    # Check shapes match
    if cleartext_output.shape != fhe_output.shape:
        if verbose:
            print(f"✗ Shape mismatch!")
            print(f"  Cleartext: {cleartext_output.shape}")
            print(f"  FHE:       {fhe_output.shape}")
        results['success'] = False
        results['error'] = 'shape_mismatch'
        return results

    # Compute error metrics
    diff = (cleartext_output - fhe_output).abs()
    rel_diff = diff / (cleartext_output.abs() + 1e-8)

    mae = diff.mean().item()
    max_abs_error = diff.max().item()
    mean_rel_error = rel_diff.mean().item()
    max_rel_error = rel_diff.max().item()

    if verbose:
        print(f"\nError Metrics:")
        print(f"  MAE (Mean Absolute Error):     {mae:.6f}")
        print(f"  Max Absolute Error:            {max_abs_error:.6f}")
        print(f"  Mean Relative Error:           {mean_rel_error:.4%}")
        print(f"  Max Relative Error:            {max_rel_error:.4%}")

    # Sample comparison (first 10 values)
    if verbose:
        print(f"\nSample Comparison (first 10 values):")
        print(f"  {'Index':<8} {'Cleartext':<15} {'FHE':<15} {'Abs Error':<15} {'Rel Error':<15}")
        print(f"  {'-'*75}")
        for i in range(min(10, cleartext_output.shape[1])):
            ct_val = cleartext_output[0, i].item()
            fhe_val = fhe_output[0, i].item()
            abs_err = abs(ct_val - fhe_val)
            rel_err = abs_err / (abs(ct_val) + 1e-8)
            print(f"  {i:<8} {ct_val:<15.6f} {fhe_val:<15.6f} {abs_err:<15.6f} {rel_err:<15.4%}")

    results['mae'] = mae
    results['max_abs_error'] = max_abs_error
    results['mean_rel_error'] = mean_rel_error
    results['max_rel_error'] = max_rel_error

    # Success criteria: MAE < 1.0
    success = mae < 1.0

    if verbose:
        print(f"\n{'='*80}")
        if success:
            print(f"✓ TEST PASSED - FHE inference matches cleartext!")
            print(f"  MAE ({mae:.6f}) < 1.0 threshold")
        else:
            print(f"✗ TEST FAILED - FHE inference does not match cleartext")
            print(f"  MAE ({mae:.6f}) >= 1.0 threshold")
        print(f"{'='*80}")

    results['success'] = success

    # ==============================================================================
    # Summary
    # ==============================================================================
    if verbose:
        print(f"\nSUMMARY")
        print(f"{'='*80}")
        print(f"Backbone:         Patch {patch_idx}")
        print(f"Input shape:      {test_patch.shape}")
        print(f"Output shape:     {cleartext_output.shape}")
        print(f"Input level:      {input_level}")
        print(f"MAE:              {mae:.6f}")
        print(f"Max Abs Error:    {max_abs_error:.6f}")
        print(f"Mean Rel Error:   {mean_rel_error:.4%}")
        print(f"Test Result:      {'✓ PASSED' if success else '✗ FAILED'}")
        print(f"{'='*80}\n")

    return results


def main():
    """Run single backbone test."""
    import argparse

    parser = argparse.ArgumentParser(description='Test single backbone cleartext vs FHE')
    parser.add_argument('--patch', type=int, default=0,
                       help='Which patch to test (0-3 for Net4)')
    parser.add_argument('--config', type=str, default='configs/cryptoface_net4.yml',
                       help='Path to CKKS config file')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')

    args = parser.parse_args()

    results = test_single_backbone_inference(
        patch_idx=args.patch,
        config_path=args.config,
        verbose=not args.quiet
    )

    # Exit with appropriate code
    if results.get('success', False):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
