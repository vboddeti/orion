"""
Test if HerPN fusion matches the unfused BatchNorm behavior.

This compares the unfused forward (using bn0, bn1, bn2) with the fused HerPN
to verify the fusion math is correct.
"""
import sys
import torch
import math

sys.path.append('/research/hal-vishnu/code/orion-fhe')

from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4
from face_recognition.models.weight_loader import load_checkpoint_for_config


def test_herpn_fusion():
    """Test if full layer1 fusion matches unfused behavior."""
    print(f"\n{'='*80}")
    print("Testing Full Layer1 Fusion (HerPN → Conv → HerPN → Conv → Shortcut)")
    print(f"{'='*80}\n")

    # Create test input (needs to match initial conv output shape)
    torch.manual_seed(42)
    x = torch.randn(1, 3, 32, 32)
    # Normalize to [-1, 1] as CryptoFace expects
    x = (x - x.min()) / (x.max() - x.min()) * 2 - 1

    print(f"Input shape: {x.shape}")
    print(f"Input range: [{x.min():.6f}, {x.max():.6f}]\n")

    # ========================================
    # UNFUSED: Full layer1 forward
    # ========================================
    print("UNFUSED (full layer1 with bn0, bn1, bn2):")

    model_unfused = CryptoFaceNet4()
    load_checkpoint_for_config(model_unfused, input_size=64, verbose=False)
    model_unfused.eval()

    backbone_unfused = model_unfused.nets[0]

    with torch.no_grad():
        # Initial conv
        out = backbone_unfused.conv(x)
        # Layer1 (unfused forward)
        out_unfused = backbone_unfused.layer1(out)

    print(f"  Output range: [{out_unfused.min():.6f}, {out_unfused.max():.6f}]")
    print(f"  Output mean: {out_unfused.mean():.6f}\n")

    # ========================================
    # FUSED: Full layer1 forward
    # ========================================
    print("FUSED (full layer1 with factored HerPN):")

    model_fused = CryptoFaceNet4()
    load_checkpoint_for_config(model_fused, input_size=64, verbose=False)
    model_fused.init_orion_params()  # Fuse the entire model
    model_fused.eval()

    backbone_fused = model_fused.nets[0]

    # Check that fusion happened
    print(f"  Fusion check: herpn1 exists = {backbone_fused.layer1.herpn1 is not None}")

    with torch.no_grad():
        # Initial conv
        out = backbone_fused.conv(x)
        # Layer1 (fused forward)
        out_fused = backbone_fused.layer1(out)

    print(f"  Output range: [{out_fused.min():.6f}, {out_fused.max():.6f}]")
    print(f"  Output mean: {out_fused.mean():.6f}\n")

    # ========================================
    # COMPARISON
    # ========================================
    print(f"{'='*80}")
    print("Comparison")
    print(f"{'='*80}\n")

    diff = (out_unfused - out_fused).abs()
    mae = diff.mean().item()
    max_err = diff.max().item()
    rel_err = (diff / (out_unfused.abs() + 1e-6)).mean().item()

    print(f"MAE: {mae:.6f}")
    print(f"Max error: {max_err:.6f}")
    print(f"Relative error: {rel_err:.4f}")

    if mae < 0.01:
        print(f"\n✓ Fusion is CORRECT (MAE < 0.01)")
        return True
    else:
        print(f"\n✗ Fusion is WRONG (MAE = {mae:.6f})")

        # Show a few sample comparisons
        print("\nSample comparisons (first channel, first pixel):")
        for i in range(min(5, x.shape[1])):
            unfused_val = out_unfused[0, i, 0, 0].item()
            fused_val = out_fused[0, i, 0, 0].item()
            print(f"  Channel {i}: unfused={unfused_val:10.6f}, fused={fused_val:10.6f}, diff={abs(unfused_val - fused_val):10.6f}")

        return False


if __name__ == "__main__":
    success = test_herpn_fusion()
    sys.exit(0 if success else 1)
