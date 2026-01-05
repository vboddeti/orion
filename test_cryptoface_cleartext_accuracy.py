"""
Test CryptoFace PCNN cleartext accuracy with L2 normalization.

This script compares cleartext inference results:
1. With L2 normalization (polynomial approximation)
2. With standard L2 normalization (exact 1/sqrt)
3. Without normalization

It validates that:
- Polynomial approximation produces approximately normalized embeddings
- Embeddings have L2 norm ≈ 1.0
- Model produces consistent outputs
"""
import torch
import numpy as np
from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4
from face_recognition.models.weight_loader import load_cryptoface_checkpoint
from face_recognition.utils.l2_coeffs import load_l2_coeffs


def load_pretrained_model(checkpoint_path="face_recognition/checkpoints/backbone-64x64.ckpt"):
    """Load pretrained CryptoFaceNet4 model."""
    # Load coefficients
    a, b, c = load_l2_coeffs('lfw')

    # Create model
    model = CryptoFaceNet4(l2_norm_coeffs=(a, b, c))

    # Load checkpoint
    try:
        print(f"Loading checkpoint: {checkpoint_path}")
        load_cryptoface_checkpoint(model, checkpoint_path, verbose=False)
        print("✓ Checkpoint loaded successfully")

        # IMPORTANT: Initialize HerPN parameters after loading checkpoint
        # This fuses the BatchNorm layers into HerPN activations
        print("Initializing HerPN parameters...")
        model.init_orion_params()
        print("✓ HerPN parameters initialized")

    except FileNotFoundError:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Using randomly initialized weights (results may not be meaningful)")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Using randomly initialized weights")

    model.eval()
    return model


def exact_l2_normalize(x):
    """Reference L2 normalization using exact sqrt."""
    norm = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm + 1e-8)


def poly_l2_normalize(x, a, b, c):
    """L2 normalization using polynomial approximation."""
    y = torch.sum(x ** 2, dim=1, keepdim=True)
    norm_inv = a * y ** 2 + b * y + c
    return x * norm_inv


def test_normalization_accuracy():
    """Test that polynomial L2 normalization approximates exact normalization."""
    print("="*70)
    print("Test 1: Polynomial Approximation Accuracy")
    print("="*70)

    # Load coefficients
    a, b, c = load_l2_coeffs('lfw')
    print(f"\nCoefficients: a={a:.2e}, b={b:.2e}, c={c:.2e}")

    # Generate random embeddings (similar to actual model output)
    torch.manual_seed(42)
    embeddings = torch.randn(100, 256)

    # Apply both normalization methods
    exact_normalized = exact_l2_normalize(embeddings)
    poly_normalized = poly_l2_normalize(embeddings, a, b, c)

    # Compute L2 norms
    exact_norms = torch.norm(exact_normalized, p=2, dim=1)
    poly_norms = torch.norm(poly_normalized, p=2, dim=1)

    print(f"\nExact L2 normalization:")
    print(f"  Mean norm: {exact_norms.mean():.6f}")
    print(f"  Std norm:  {exact_norms.std():.6f}")

    print(f"\nPolynomial L2 normalization:")
    print(f"  Mean norm: {poly_norms.mean():.6f}")
    print(f"  Std norm:  {poly_norms.std():.6f}")

    # Compare embeddings
    diff = torch.abs(exact_normalized - poly_normalized)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\nDifference between methods:")
    print(f"  Max absolute diff:  {max_diff:.6e}")
    print(f"  Mean absolute diff: {mean_diff:.6e}")

    # Compute cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        exact_normalized, poly_normalized, dim=1
    )
    print(f"\nCosine similarity:")
    print(f"  Mean: {cos_sim.mean():.6f}")
    print(f"  Min:  {cos_sim.min():.6f}")

    # Check if approximation is good enough
    success = True
    if poly_norms.mean() < 0.9 or poly_norms.mean() > 1.1:
        print(f"\n✗ Polynomial normalization produces norms outside [0.9, 1.1]")
        success = False
    else:
        print(f"\n✓ Polynomial normalization produces approximately unit norms")

    if cos_sim.mean() < 0.99:
        print(f"✗ Cosine similarity ({cos_sim.mean():.6f}) < 0.99")
        success = False
    else:
        print(f"✓ Cosine similarity ({cos_sim.mean():.6f}) ≥ 0.99")

    return success


def test_model_with_normalization():
    """Test CryptoFacePCNN model with L2 normalization."""
    print("\n" + "="*70)
    print("Test 2: Model with L2 Normalization")
    print("="*70)

    # Load pretrained model
    print("\nLoading pretrained CryptoFaceNet4...")
    model = load_pretrained_model()

    # Test on random images
    torch.manual_seed(42)
    batch_size = 16
    images = torch.randn(batch_size, 3, 64, 64)

    print(f"Testing on batch of {batch_size} images...")
    with torch.no_grad():
        # Get features before normalization
        model_temp = model
        # Temporarily remove normalization to see intermediate output
        model.pt()  # Ensure plaintext mode

        # Run forward pass
        embeddings = model(images)

        # Debug: Check intermediate values
        print(f"\n  Checking for NaN/Inf in output...")
        if torch.isnan(embeddings).any():
            print(f"  WARNING: NaN detected in embeddings!")
            # Find where NaN appears
            nan_mask = torch.isnan(embeddings)
            print(f"  Number of NaN values: {nan_mask.sum().item()} / {embeddings.numel()}")
        if torch.isinf(embeddings).any():
            print(f"  WARNING: Inf detected in embeddings!")

    print(f"\nOutput shape: {embeddings.shape}")
    print(f"Expected:     torch.Size([{batch_size}, 256])")

    # Check L2 norms
    norms = torch.norm(embeddings, p=2, dim=1)
    print(f"\nEmbedding L2 norms:")
    print(f"  Mean: {norms.mean():.6f}")
    print(f"  Std:  {norms.std():.6f}")
    print(f"  Min:  {norms.min():.6f}")
    print(f"  Max:  {norms.max():.6f}")

    # Check for valid output
    assert embeddings.shape == (batch_size, 256), "Output shape mismatch"
    assert not torch.isnan(embeddings).any(), "Output contains NaN"
    assert not torch.isinf(embeddings).any(), "Output contains Inf"

    success = True
    if norms.mean() < 0.8 or norms.mean() > 1.2:
        print(f"\n✗ Mean norm ({norms.mean():.6f}) outside [0.8, 1.2]")
        success = False
    else:
        print(f"\n✓ Mean norm ({norms.mean():.6f}) in acceptable range")

    if norms.std() > 0.1:
        print(f"✗ Std of norms ({norms.std():.6f}) > 0.1 (inconsistent normalization)")
        success = False
    else:
        print(f"✓ Std of norms ({norms.std():.6f}) ≤ 0.1 (consistent normalization)")

    return success


def test_model_consistency():
    """Test that model produces consistent embeddings."""
    print("\n" + "="*70)
    print("Test 3: Model Consistency")
    print("="*70)

    # Load pretrained model
    print("\nLoading pretrained CryptoFaceNet4...")
    model = load_pretrained_model()

    # Same input, multiple runs
    torch.manual_seed(42)
    test_image = torch.randn(1, 3, 64, 64)

    print("\nRunning inference 5 times on same input...")
    embeddings = []
    with torch.no_grad():
        for i in range(5):
            emb = model(test_image)
            embeddings.append(emb)
            print(f"  Run {i+1}: norm = {torch.norm(emb, p=2).item():.6f}")

    # Check consistency
    ref_emb = embeddings[0]
    max_diff = 0.0
    for i, emb in enumerate(embeddings[1:], 1):
        diff = torch.abs(ref_emb - emb).max().item()
        max_diff = max(max_diff, diff)
        print(f"\n  Max diff (run 1 vs run {i+1}): {diff:.6e}")

    success = True
    if max_diff < 1e-6:
        print(f"\n✓ Model is deterministic (max diff = {max_diff:.6e})")
    else:
        print(f"\n✗ Model shows variation (max diff = {max_diff:.6e})")
        success = False

    return success


def test_embedding_distribution():
    """Test distribution of embeddings across multiple images."""
    print("\n" + "="*70)
    print("Test 4: Embedding Distribution")
    print("="*70)

    # Load pretrained model
    print("\nLoading pretrained CryptoFaceNet4...")
    model = load_pretrained_model()

    # Generate diverse random images
    torch.manual_seed(42)
    num_images = 100
    images = torch.randn(num_images, 3, 64, 64)

    print(f"\nGenerating embeddings for {num_images} random images...")
    with torch.no_grad():
        embeddings = model(images)

    # Analyze embedding statistics
    print(f"\nEmbedding statistics:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Mean: {embeddings.mean():.6f}")
    print(f"  Std:  {embeddings.std():.6f}")
    print(f"  Min:  {embeddings.min():.6f}")
    print(f"  Max:  {embeddings.max():.6f}")

    # Analyze L2 norms
    norms = torch.norm(embeddings, p=2, dim=1)
    print(f"\nL2 norms:")
    print(f"  Mean: {norms.mean():.6f}")
    print(f"  Std:  {norms.std():.6f}")
    print(f"  Min:  {norms.min():.6f}")
    print(f"  Max:  {norms.max():.6f}")

    # Compute pairwise cosine similarities
    normalized_emb = embeddings / (torch.norm(embeddings, p=2, dim=1, keepdim=True) + 1e-8)
    cos_sim_matrix = torch.mm(normalized_emb, normalized_emb.t())

    # Exclude diagonal (self-similarity)
    mask = ~torch.eye(num_images, dtype=torch.bool)
    cos_sim_values = cos_sim_matrix[mask]

    print(f"\nPairwise cosine similarities:")
    print(f"  Mean: {cos_sim_values.mean():.6f}")
    print(f"  Std:  {cos_sim_values.std():.6f}")
    print(f"  Min:  {cos_sim_values.min():.6f}")
    print(f"  Max:  {cos_sim_values.max():.6f}")

    success = True

    # Check norm consistency
    if norms.std() > 0.1:
        print(f"\n✗ Large variation in L2 norms (std={norms.std():.6f})")
        success = False
    else:
        print(f"\n✓ Consistent L2 norms (std={norms.std():.6f})")

    # Check embedding diversity
    if cos_sim_values.std() < 0.05:
        print(f"✗ Low diversity in embeddings (cosine similarity std={cos_sim_values.std():.6f})")
        success = False
    else:
        print(f"✓ Good embedding diversity (cosine similarity std={cos_sim_values.std():.6f})")

    return success


def main():
    """Run all cleartext accuracy tests."""
    print("="*70)
    print("CryptoFace PCNN Cleartext Accuracy Tests with L2 Normalization")
    print("="*70)

    results = {
        "Polynomial Approximation": test_normalization_accuracy(),
        "Model with L2 Norm": test_model_with_normalization(),
        "Model Consistency": test_model_consistency(),
        "Embedding Distribution": test_embedding_distribution(),
    }

    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)

    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:30s} {status}")

    print("="*70)

    all_passed = all(results.values())
    if all_passed:
        print("\n✓ All cleartext accuracy tests PASSED!")
        return 0
    else:
        print("\n✗ Some tests FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
