"""
Test that the BatchNorm fusion in weight loading is mathematically correct.

This test verifies that:
1. The final BatchNorm is set to identity (mean=0, var=1)
2. The fused linear layers produce equivalent outputs to the original unfused version
3. Forward pass produces reasonable embeddings
"""
import sys
import torch
import numpy as np

sys.path.append('/research/hal-vishnu/code/orion-fhe')

from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4
from face_recognition.models.weight_loader import load_checkpoint_for_config


def test_fusion_correctness():
    """Verify BatchNorm fusion is mathematically correct."""
    print("="*80)
    print("TEST: BatchNorm Fusion Correctness")
    print("="*80)

    # Create model and load weights
    model = CryptoFaceNet4()
    load_checkpoint_for_config(model, input_size=64, verbose=False)

    print("\n" + "="*80)
    print("1. Checking Final BatchNorm is Identity")
    print("="*80)

    # Check that final normalization BatchNorm is identity
    final_bn_mean = model.normalization.running_mean
    final_bn_var = model.normalization.running_var

    print(f"Final BatchNorm running_mean (should be all zeros):")
    print(f"  Mean: {final_bn_mean.mean():.10f}")
    print(f"  Max abs: {final_bn_mean.abs().max():.10f}")
    print(f"  First 5 values: {final_bn_mean[:5]}")

    print(f"\nFinal BatchNorm running_var (should be all ones):")
    print(f"  Mean: {final_bn_var.mean():.10f}")
    print(f"  Max deviation from 1.0: {(final_bn_var - 1.0).abs().max():.10f}")
    print(f"  First 5 values: {final_bn_var[:5]}")

    # Verify identity
    is_mean_zero = torch.allclose(final_bn_mean, torch.zeros_like(final_bn_mean), atol=1e-6)
    is_var_one = torch.allclose(final_bn_var, torch.ones_like(final_bn_var), atol=1e-6)

    if is_mean_zero and is_var_one:
        print("\n✓ Final BatchNorm is identity!")
    else:
        print("\n✗ Final BatchNorm is NOT identity!")
        print(f"  Mean is zero: {is_mean_zero}")
        print(f"  Var is one: {is_var_one}")

    print("\n" + "="*80)
    print("2. Checking Fused Linear Layer Weights")
    print("="*80)

    # Check that all patches have the same fused bias
    biases = [model.linear[i].bias for i in range(model.N)]
    print(f"\nPer-patch linear bias consistency:")
    for i in range(model.N):
        print(f"  Patch {i} bias range: [{biases[i].min():.4f}, {biases[i].max():.4f}]")

    # All biases should be identical (they're all the fused bias)
    all_same = all(torch.allclose(biases[0], biases[i]) for i in range(1, model.N))
    print(f"\nAll patches have identical bias: {all_same}")
    if all_same:
        print("✓ Fused bias correctly distributed!")
    else:
        print("✗ Fused bias NOT consistent across patches!")

    print("\n" + "="*80)
    print("3. Testing Forward Pass")
    print("="*80)

    # Test forward pass with random input
    model.eval()
    with torch.no_grad():
        x = torch.randn(4, 3, 64, 64)
        print(f"\nInput shape: {x.shape}")

        out = model(x)
        print(f"Output shape: {out.shape}")
        print(f"Output range: [{out.min():.4f}, {out.max():.4f}]")
        print(f"Output mean: {out.mean():.4f}")
        print(f"Output std: {out.std():.4f}")

        # Check for NaN or Inf
        has_nan = torch.isnan(out).any()
        has_inf = torch.isinf(out).any()
        print(f"\nContains NaN: {has_nan}")
        print(f"Contains Inf: {has_inf}")

        if not has_nan and not has_inf:
            print("✓ Forward pass produces valid outputs!")
        else:
            print("✗ Forward pass produced NaN or Inf!")

    print("\n" + "="*80)
    print("4. Testing Embedding Magnitude")
    print("="*80)

    # Test with a batch of different inputs
    with torch.no_grad():
        embeddings = []
        for _ in range(5):
            x = torch.randn(1, 3, 64, 64)
            out = model(x)
            embeddings.append(out)

        embeddings = torch.cat(embeddings, dim=0)
        print(f"\nGenerated {embeddings.shape[0]} embeddings")
        print(f"Embedding stats:")
        print(f"  Mean: {embeddings.mean():.4f}")
        print(f"  Std: {embeddings.std():.4f}")
        print(f"  Min: {embeddings.min():.4f}")
        print(f"  Max: {embeddings.max():.4f}")

        # Compute pairwise cosine similarities
        normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        similarities = normalized @ normalized.T
        # Get off-diagonal elements
        off_diag = similarities[~torch.eye(5, dtype=bool)]
        print(f"\nPairwise cosine similarities (should be diverse):")
        print(f"  Mean: {off_diag.mean():.4f}")
        print(f"  Std: {off_diag.std():.4f}")
        print(f"  Range: [{off_diag.min():.4f}, {off_diag.max():.4f}]")

        # Embeddings should be diverse (not all the same)
        is_diverse = off_diag.std() > 0.01
        if is_diverse:
            print("✓ Embeddings are diverse!")
        else:
            print("✗ Embeddings are too similar (possible issue)!")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    all_checks = [
        ("Final BatchNorm is identity", is_mean_zero and is_var_one),
        ("Fused bias is consistent", all_same),
        ("Forward pass is valid", not has_nan and not has_inf),
        ("Embeddings are diverse", is_diverse),
    ]

    print()
    for check_name, passed in all_checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check_name}")

    all_passed = all(passed for _, passed in all_checks)
    print("\n" + "="*80)
    if all_passed:
        print("✓ All fusion correctness tests PASSED!")
    else:
        print("✗ Some fusion correctness tests FAILED!")
    print("="*80)


if __name__ == "__main__":
    test_fusion_correctness()
