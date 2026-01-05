"""
Test L2NormPoly implementation against reference.

This script verifies that the L2NormPoly module correctly implements
the polynomial approximation of L2 normalization used in CryptoFace.
"""
import torch
import numpy as np
from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4

# Reference implementation from CryptoFace/helper.py
def l2norm_reference(x_np, a, b, c):
    """Reference L2 normalization implementation."""
    y = np.sum(x_np ** 2, axis=1, keepdims=True)
    norm_inv = a * y ** 2 + b * y + c
    x_norm = x_np * norm_inv
    return x_norm


def test_l2norm_poly():
    """Test L2NormPoly against reference implementation."""
    print("="*70)
    print("Testing L2NormPoly Implementation")
    print("="*70)

    # Test coefficients (from LFW dataset)
    a = 2.41e-07
    b = -2.44e-04
    c = 1.09e-01

    # Create model with L2 normalization
    print("\nCreating CryptoFaceNet4 with L2 normalization...")
    model = CryptoFaceNet4(l2_norm_coeffs=(a, b, c))
    model.eval()

    # Test on random embeddings
    print("\nTesting on random embeddings...")
    batch_size = 8
    embedding_dim = 256

    # Generate random embeddings
    embeddings = torch.randn(batch_size, embedding_dim)

    # Apply L2 normalization using our module
    model_output = model.normalization(embeddings)

    # Apply reference implementation
    embeddings_np = embeddings.detach().numpy()
    reference_output = l2norm_reference(embeddings_np, a, b, c)

    # Compare results
    model_output_np = model_output.detach().numpy()
    max_diff = np.max(np.abs(model_output_np - reference_output))
    mean_diff = np.mean(np.abs(model_output_np - reference_output))

    print(f"\nComparison with reference implementation:")
    print(f"  Max absolute difference:  {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")

    # Check if close enough (tolerance: 1e-6)
    if max_diff < 1e-6:
        print(f"\n✓ L2NormPoly matches reference implementation!")
        return True
    else:
        print(f"\n✗ L2NormPoly differs from reference (max diff: {max_diff:.2e})")
        return False


def test_l2norm_magnitude():
    """Test that L2 normalization approximately normalizes vectors to unit length."""
    print("\n" + "="*70)
    print("Testing L2 Normalization Magnitude")
    print("="*70)

    # Test coefficients
    a = 2.41e-07
    b = -2.44e-04
    c = 1.09e-01

    # Create model
    model = CryptoFaceNet4(l2_norm_coeffs=(a, b, c))
    model.eval()

    # Test on various embeddings
    batch_size = 16
    embedding_dim = 256

    embeddings = torch.randn(batch_size, embedding_dim)

    # Normalize
    normalized = model.normalization(embeddings)

    # Compute L2 norms
    norms = torch.norm(normalized, p=2, dim=1)

    print(f"\nL2 norms after normalization:")
    print(f"  Mean: {norms.mean():.6f}")
    print(f"  Std:  {norms.std():.6f}")
    print(f"  Min:  {norms.min():.6f}")
    print(f"  Max:  {norms.max():.6f}")

    # Check if norms are close to 1.0
    # Note: Polynomial approximation won't be perfect, so allow some tolerance
    mean_norm = norms.mean().item()
    if 0.9 < mean_norm < 1.1:
        print(f"\n✓ Normalization produces approximately unit-length vectors")
        return True
    else:
        print(f"\n✗ Normalization produces vectors with mean norm {mean_norm:.6f} (expected ~1.0)")
        return False


if __name__ == "__main__":
    # Run tests
    test1_passed = test_l2norm_poly()
    test2_passed = test_l2norm_magnitude()

    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    print(f"L2NormPoly vs Reference:  {'PASS' if test1_passed else 'FAIL'}")
    print(f"Normalization Magnitude:  {'PASS' if test2_passed else 'FAIL'}")
    print("="*70)

    if test1_passed and test2_passed:
        print("\n✓ All tests passed!")
        exit(0)
    else:
        print("\n✗ Some tests failed")
        exit(1)
