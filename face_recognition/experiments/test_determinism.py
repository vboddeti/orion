"""
Test that the model produces deterministic outputs and handles different inputs correctly.
"""
import sys
import torch

sys.path.append('/research/hal-vishnu/code/orion-fhe')

from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4
from face_recognition.models.weight_loader import load_checkpoint_for_config


def test_determinism():
    """Test model determinism and sensitivity to inputs."""
    print("="*80)
    print("TEST: Model Determinism and Input Sensitivity")
    print("="*80)

    # Create model and load weights
    model = CryptoFaceNet4()
    load_checkpoint_for_config(model, input_size=64, verbose=False)
    model.eval()

    print("\n" + "="*80)
    print("1. Testing Determinism (same input → same output)")
    print("="*80)

    # Test with same input multiple times
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
        out3 = model(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Output 1 range: [{out1.min():.4f}, {out1.max():.4f}]")
    print(f"Output 2 range: [{out2.min():.4f}, {out2.max():.4f}]")
    print(f"Output 3 range: [{out3.min():.4f}, {out3.max():.4f}]")

    # Check if outputs are identical
    identical_12 = torch.allclose(out1, out2, atol=1e-6)
    identical_23 = torch.allclose(out2, out3, atol=1e-6)

    print(f"\nOutputs 1 and 2 identical: {identical_12}")
    print(f"Outputs 2 and 3 identical: {identical_23}")

    if identical_12 and identical_23:
        print("✓ Model is deterministic!")
    else:
        print("✗ Model is NOT deterministic (possible issue)!")
        print(f"Max diff 1-2: {(out1 - out2).abs().max():.10f}")
        print(f"Max diff 2-3: {(out2 - out3).abs().max():.10f}")

    print("\n" + "="*80)
    print("2. Testing Input Sensitivity (different inputs → different outputs)")
    print("="*80)

    # Test with very different inputs
    x1 = torch.zeros(1, 3, 64, 64)  # All zeros
    x2 = torch.ones(1, 3, 64, 64)   # All ones
    x3 = torch.randn(1, 3, 64, 64)  # Random noise

    with torch.no_grad():
        emb1 = model(x1)
        emb2 = model(x2)
        emb3 = model(x3)

    print(f"\nEmbedding 1 (zeros input):")
    print(f"  Range: [{emb1.min():.4f}, {emb1.max():.4f}]")
    print(f"  Mean: {emb1.mean():.4f}, Std: {emb1.std():.4f}")

    print(f"\nEmbedding 2 (ones input):")
    print(f"  Range: [{emb2.min():.4f}, {emb2.max():.4f}]")
    print(f"  Mean: {emb2.mean():.4f}, Std: {emb2.std():.4f}")

    print(f"\nEmbedding 3 (random input):")
    print(f"  Range: [{emb3.min():.4f}, {emb3.max():.4f}]")
    print(f"  Mean: {emb3.mean():.4f}, Std: {emb3.std():.4f}")

    # Compute differences
    diff_12 = (emb1 - emb2).abs().max()
    diff_23 = (emb2 - emb3).abs().max()
    diff_13 = (emb1 - emb3).abs().max()

    print(f"\nMax absolute differences:")
    print(f"  Zeros vs Ones: {diff_12:.4f}")
    print(f"  Ones vs Random: {diff_23:.4f}")
    print(f"  Zeros vs Random: {diff_13:.4f}")

    # Compute cosine similarities
    emb1_norm = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2_norm = torch.nn.functional.normalize(emb2, p=2, dim=1)
    emb3_norm = torch.nn.functional.normalize(emb3, p=2, dim=1)

    cos_12 = (emb1_norm @ emb2_norm.T).item()
    cos_23 = (emb2_norm @ emb3_norm.T).item()
    cos_13 = (emb1_norm @ emb3_norm.T).item()

    print(f"\nCosine similarities:")
    print(f"  Zeros vs Ones: {cos_12:.6f}")
    print(f"  Ones vs Random: {cos_23:.6f}")
    print(f"  Zeros vs Random: {cos_13:.6f}")

    # Check if model is sensitive to inputs
    is_sensitive = (diff_12 > 1.0) and (diff_23 > 1.0) and (diff_13 > 1.0)
    cosines_not_perfect = (abs(cos_12) < 0.9999) and (abs(cos_23) < 0.9999) and (abs(cos_13) < 0.9999)

    if is_sensitive and cosines_not_perfect:
        print("\n✓ Model is sensitive to different inputs!")
    else:
        print("\n✗ Model shows low sensitivity to inputs (check if weights loaded correctly)!")

    print("\n" + "="*80)
    print("3. Testing BatchNorm Layers in Backbones")
    print("="*80)

    # Check if backbone BatchNorm statistics are loaded
    print("\nBackbone 0 - Layer 1 BatchNorm statistics:")
    bn0_1 = model.nets[0].layer1.bn0_1
    print(f"  running_mean: min={bn0_1.running_mean.min():.4f}, max={bn0_1.running_mean.max():.4f}")
    print(f"  running_var: min={bn0_1.running_var.min():.4f}, max={bn0_1.running_var.max():.4f}")
    print(f"  First 5 means: {bn0_1.running_mean[:5]}")
    print(f"  First 5 vars: {bn0_1.running_var[:5]}")

    # Check if statistics are non-default (should be loaded from checkpoint)
    mean_is_default = torch.allclose(bn0_1.running_mean, torch.zeros_like(bn0_1.running_mean))
    var_is_default = torch.allclose(bn0_1.running_var, torch.ones_like(bn0_1.running_var))

    if not mean_is_default and not var_is_default:
        print("\n✓ Backbone BatchNorm statistics loaded!")
    else:
        print("\n✗ Backbone BatchNorm statistics NOT loaded (still at defaults)!")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    all_checks = [
        ("Model is deterministic", identical_12 and identical_23),
        ("Model is sensitive to inputs", is_sensitive and cosines_not_perfect),
        ("Backbone BN statistics loaded", not mean_is_default and not var_is_default),
    ]

    print()
    for check_name, passed in all_checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check_name}")

    all_passed = all(passed for _, passed in all_checks)
    print("\n" + "="*80)
    if all_passed:
        print("✓ All determinism tests PASSED!")
    else:
        print("⚠ Some tests failed - review results above")
    print("="*80)


if __name__ == "__main__":
    test_determinism()
