"""
Test that BatchNorm fusion is mathematically equivalent to the unfused version.

This test verifies that:
    fused_linear(x) ≡ batchnorm(unfused_linear(x))

Where fused_linear has BatchNorm statistics fused into its weights.
"""
import sys
import torch

sys.path.append('/research/hal-vishnu/code/orion-fhe')

from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4


def test_fusion_mathematical_equivalence():
    """Verify fusion is mathematically equivalent to unfused version."""
    print("="*80)
    print("TEST: Mathematical Equivalence of BatchNorm Fusion")
    print("="*80)

    print("\nThis test loads weights twice:")
    print("  1. With fusion (current implementation)")
    print("  2. Without fusion (for comparison)")
    print("Then verifies outputs are identical.\n")

    # Load checkpoint path
    checkpoint_path = "face_recognition/checkpoints/backbone-64x64.ckpt"
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['backbone']

    # Get required parameters
    N = 4  # CryptoFaceNet4 has 4 patches
    embedding_dim = 256
    backbone_dim = 256

    full_weight = state_dict['linear.weight']  # [256, 1024]
    full_bias = state_dict['linear.bias']      # [256]
    bn_mean = state_dict['bn.running_mean']    # [256]
    bn_var = state_dict['bn.running_var']      # [256]
    bn_eps = 1e-5  # Default BatchNorm epsilon

    print("="*80)
    print("1. Manual Fusion Computation")
    print("="*80)

    # Manually compute fused weights (matching CryptoFace pcnn.py:78-85)
    print("\nComputing fused weights manually...")
    weight_fused_manual = torch.divide(full_weight.T, torch.sqrt(bn_var + bn_eps))
    weight_fused_manual = weight_fused_manual.T
    bias_fused_manual = torch.divide(full_bias - bn_mean, torch.sqrt(bn_var + bn_eps))
    bias_fused_manual = bias_fused_manual / N

    print(f"  Fused weight shape: {weight_fused_manual.shape}")
    print(f"  Fused bias shape: {bias_fused_manual.shape}")
    print(f"  Fused bias range: [{bias_fused_manual.min():.4f}, {bias_fused_manual.max():.4f}]")

    # Chunk manually
    chunked_manual = torch.chunk(weight_fused_manual, N, dim=1)
    print(f"  Chunked into {len(chunked_manual)} pieces of shape {chunked_manual[0].shape}")

    print("\n" + "="*80)
    print("2. Loading Model with Automatic Fusion")
    print("="*80)

    from face_recognition.models.weight_loader import load_checkpoint_for_config

    model = CryptoFaceNet4()
    load_checkpoint_for_config(model, input_size=64, verbose=False)
    model.eval()  # IMPORTANT: Set to eval mode so BatchNorm uses running stats

    print("\n" + "="*80)
    print("3. Comparing Manual vs Automatic Fusion")
    print("="*80)

    # Compare each patch's linear layer
    all_weights_match = True
    all_biases_match = True

    for i in range(N):
        weight_match = torch.allclose(
            model.linear[i].weight.data,
            chunked_manual[i],
            atol=1e-6
        )
        bias_match = torch.allclose(
            model.linear[i].bias.data,
            bias_fused_manual,
            atol=1e-6
        )

        if not weight_match:
            diff = (model.linear[i].weight.data - chunked_manual[i]).abs().max()
            print(f"  Patch {i} weight: ✗ MISMATCH (max diff: {diff:.10f})")
            all_weights_match = False
        else:
            print(f"  Patch {i} weight: ✓ matches")

        if not bias_match:
            diff = (model.linear[i].bias.data - bias_fused_manual).abs().max()
            print(f"  Patch {i} bias:   ✗ MISMATCH (max diff: {diff:.10f})")
            all_biases_match = False
        else:
            print(f"  Patch {i} bias:   ✓ matches")

    if all_weights_match and all_biases_match:
        print("\n✓ All fused weights match manual computation!")
    else:
        print("\n✗ Fusion mismatch detected!")

    print("\n" + "="*80)
    print("4. Testing Forward Pass Equivalence")
    print("="*80)

    # Create a simple test: linear layer output with and without BatchNorm
    print("\nSimulating: Linear → BatchNorm vs Fused Linear")

    # Create unfused version
    linear_unfused = torch.nn.Linear(N * backbone_dim, embedding_dim)
    linear_unfused.weight.data = full_weight.clone()
    linear_unfused.bias.data = full_bias.clone()

    bn_unfused = torch.nn.BatchNorm1d(embedding_dim, affine=False)
    bn_unfused.running_mean.data = bn_mean.clone()
    bn_unfused.running_var.data = bn_var.clone()
    bn_unfused.eps = bn_eps
    bn_unfused.eval()

    # Test input (simulating concatenated backbone outputs)
    x_test = torch.randn(8, N * backbone_dim)

    # Unfused path: Linear → BatchNorm
    with torch.no_grad():
        out_unfused_before_bn = linear_unfused(x_test)
        out_unfused = bn_unfused(out_unfused_before_bn)

    # Fused path: Create fused linear layer manually for fair comparison
    with torch.no_grad():
        # Create fused linear layer that operates on full input
        linear_fused = torch.nn.Linear(N * backbone_dim, embedding_dim)

        # Set to fused weights (reconstruct from chunked weights)
        fused_weight = torch.cat([model.linear[i].weight.data for i in range(N)], dim=1)
        linear_fused.weight.data = fused_weight

        # Set fused bias (sum of N copies of per-patch bias = total fused bias)
        # model.linear[i].bias = bias_fused / N, so N copies sum to bias_fused
        linear_fused.bias.data = model.linear[0].bias.data * N

        # Apply fused linear
        out_fused_before_bn = linear_fused(x_test)

        # Apply identity BatchNorm (should do nothing since mean=0, var=1)
        out_fused = model.normalization(out_fused_before_bn)

    # Compare intermediate and final outputs
    print(f"\nUnfused (before BN): range [{out_unfused_before_bn.min():.4f}, {out_unfused_before_bn.max():.4f}]")
    print(f"Fused (before BN):   range [{out_fused_before_bn.min():.4f}, {out_fused_before_bn.max():.4f}]")

    before_bn_match = torch.allclose(out_unfused_before_bn, out_fused_before_bn, atol=1e-4)
    if before_bn_match:
        print("✓ Outputs match BEFORE BatchNorm!")
    else:
        diff_before = (out_unfused_before_bn - out_fused_before_bn).abs().max()
        print(f"✗ Outputs differ before BN by {diff_before:.6f}")

    print(f"\nUnfused (after BN): range [{out_unfused.min():.4f}, {out_unfused.max():.4f}]")
    print(f"Fused (after BN):   range [{out_fused.min():.4f}, {out_fused.max():.4f}]")

    max_diff = (out_unfused - out_fused).abs().max()
    mean_diff = (out_unfused - out_fused).abs().mean()
    print(f"\nDifference statistics:")
    print(f"  Max absolute diff:  {max_diff:.10f}")
    print(f"  Mean absolute diff: {mean_diff:.10f}")

    # Debug: Check if weights actually match
    print(f"\nDebug: Weight comparison")
    weight_reconstructed = torch.cat([model.linear[i].weight.data for i in range(N)], dim=1)
    print(f"  Reconstructed weight shape: {weight_reconstructed.shape}")
    print(f"  Manual fused weight shape: {weight_fused_manual.shape}")
    weights_match = torch.allclose(weight_reconstructed, weight_fused_manual, atol=1e-6)
    print(f"  Weights match: {weights_match}")
    if not weights_match:
        print(f"  Max weight diff: {(weight_reconstructed - weight_fused_manual).abs().max():.10f}")

    # Debug: Check bias
    print(f"\nDebug: Bias comparison")
    bias_per_patch = model.linear[0].bias.data  # This is bias_fused / N
    bias_total = bias_per_patch * N  # Reconstruct total fused bias
    bias_expected_total = torch.divide(full_bias - bn_mean, torch.sqrt(bn_var + bn_eps))  # Fused bias before /N
    print(f"  Per-patch bias: range [{bias_per_patch.min():.4f}, {bias_per_patch.max():.4f}]")
    print(f"  Total bias (×{N}): range [{bias_total.min():.4f}, {bias_total.max():.4f}]")
    print(f"  Expected total fused bias: range [{bias_expected_total.min():.4f}, {bias_expected_total.max():.4f}]")
    biases_match = torch.allclose(bias_total, bias_expected_total, atol=1e-5)
    print(f"  Total biases match: {biases_match}")
    if not biases_match:
        print(f"  Max bias diff: {(bias_total - bias_expected_total).abs().max():.10f}")

    outputs_match = torch.allclose(out_unfused, out_fused, atol=1e-4)

    if outputs_match:
        print("\n✓ Fused and unfused outputs are mathematically equivalent!")
    else:
        print(f"\n✗ Outputs differ by more than tolerance!")
        print(f"   Relative error: {(max_diff / out_unfused.abs().mean()):.6f}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    all_checks = [
        ("Fused weights match manual computation", all_weights_match and all_biases_match),
        ("Forward pass equivalence", outputs_match),
    ]

    print()
    for check_name, passed in all_checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check_name}")

    all_passed = all(passed for _, passed in all_checks)
    print("\n" + "="*80)
    if all_passed:
        print("✓ BatchNorm fusion is MATHEMATICALLY CORRECT!")
        print("\nThe weight loading implementation correctly follows CryptoFace's")
        print("fusion approach from pcnn.py:78-85.")
    else:
        print("✗ Mathematical correctness verification FAILED!")
    print("="*80)


if __name__ == "__main__":
    test_fusion_mathematical_equivalence()
