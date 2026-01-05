"""
Layer-by-layer comparison of unfused vs fused CryptoFace model.

This script traces through each layer to identify where outputs diverge.
"""
import sys
import torch
import torch.nn.functional as F

sys.path.append('/research/hal-vishnu/code/orion-fhe')

from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4
from face_recognition.models.weight_loader import load_checkpoint_for_config


def check_tensor(name, tensor, prefix=""):
    """Check tensor for NaN/Inf and print stats."""
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()

    status = ""
    if has_nan:
        status = "⚠️  NaN detected!"
    elif has_inf:
        status = "⚠️  Inf detected!"
    else:
        status = "✓"

    min_val = tensor.min().item() if not has_nan else float('nan')
    max_val = tensor.max().item() if not has_nan else float('nan')
    mean_val = tensor.mean().item() if not has_nan else float('nan')

    print(f"{prefix}{name:40s} {status:20s} shape={str(tuple(tensor.shape)):20s} "
          f"range=[{min_val:>10.4f}, {max_val:>10.4f}] mean={mean_val:>10.4f}")

    return has_nan or has_inf


def trace_backbone_layer_by_layer(backbone, x, label="Backbone", fused=False):
    """Trace through backbone layers one by one."""
    print(f"\n{'='*100}")
    print(f"{label} - Layer-by-Layer Trace (Fused={fused})")
    print(f"{'='*100}")

    check_tensor("Input", x)

    # Initial conv
    out = backbone.conv(x)
    if check_tensor("conv output", out):
        return None

    # Layer 1
    print("\n--- Layer 1 (HerPNConv) ---")

    if fused:
        # FUSED mode: Match the actual forward logic
        # Store input for shortcut (same as unfused: shortcut uses original x)
        layer_input = out

        # herpn1
        if hasattr(backbone.layer1.herpn1, 'weight0_raw'):
            w0 = backbone.layer1.herpn1.weight0_raw
            w1 = backbone.layer1.herpn1.weight1_raw
            w2 = backbone.layer1.herpn1.weight2_raw
            print(f"  herpn1 coefficients:")
            print(f"    w0 range: [{w0.min():.6f}, {w0.max():.6f}]")
            print(f"    w1 range: [{w1.min():.6f}, {w1.max():.6f}]")
            print(f"    w2 range: [{w2.min():.6f}, {w2.max():.6f}]")

        out = backbone.layer1.herpn1(out)
        if check_tensor("layer1.herpn1", out, "  "):
            return None

        # First conv
        out = backbone.layer1.conv1(out)
        if check_tensor("layer1.conv1", out, "  "):
            return None

        # herpn2
        if hasattr(backbone.layer1.herpn2, 'weight0_raw'):
            w0 = backbone.layer1.herpn2.weight0_raw
            w1 = backbone.layer1.herpn2.weight1_raw
            w2 = backbone.layer1.herpn2.weight2_raw
            print(f"  herpn2 coefficients:")
            print(f"    w0 range: [{w0.min():.6f}, {w0.max():.6f}]")
            print(f"    w1 range: [{w1.min():.6f}, {w1.max():.6f}]")
            print(f"    w2 range: [{w2.min():.6f}, {w2.max():.6f}]")

            # Check intermediate values
            print(f"  herpn2 intermediate values:")
            print(f"    input range: [{out.min():.6f}, {out.max():.6f}]")
            x_sq = out * out
            print(f"    x² range: [{x_sq.min():.6f}, {x_sq.max():.6f}]")
            w2_x_sq = w2 * x_sq
            print(f"    w2·x² range: [{w2_x_sq.min():.6f}, {w2_x_sq.max():.6f}]")
            w1_x = w1 * out
            print(f"    w1·x range: [{w1_x.min():.6f}, {w1_x.max():.6f}]")
            term_sum = w2_x_sq + w1_x + w0
            print(f"    w2·x² + w1·x + w0 range: [{term_sum.min():.6f}, {term_sum.max():.6f}]")

        out = backbone.layer1.herpn2(out)
        if check_tensor("layer1.herpn2", out, "  "):
            print("\n  ⚠️  DIVERGENCE DETECTED AT layer1.herpn2!")
            return None

        # Second conv
        out = backbone.layer1.conv2(out)
        if check_tensor("layer1.conv2", out, "  "):
            return None

        # Shortcut - use ORIGINAL layer input (same as unfused)
        shortcut = backbone.layer1.shortcut_scale(layer_input)
        if check_tensor("layer1.shortcut_scale", shortcut, "  "):
            return None

        if backbone.layer1.has_shortcut:
            shortcut = backbone.layer1.shortcut_conv(shortcut)
            if check_tensor("layer1.shortcut_conv", shortcut, "  "):
                return None
            shortcut = backbone.layer1.shortcut_bn(shortcut)
            if check_tensor("layer1.shortcut_bn", shortcut, "  "):
                return None

        out = out + shortcut
        if check_tensor("layer1.out (after shortcut)", out, "  "):
            return None

    else:
        # UNFUSED mode: Call layer forward (which uses bn0, bn1, bn2 internally)
        out = backbone.layer1(out)
        if check_tensor("layer1 output", out, "  "):
            return None

    # Layer 2
    print("\n--- Layer 2 (HerPNConv) ---")
    out = backbone.layer2(out)
    if check_tensor("layer2 output", out, "  "):
        return None

    # Layer 3
    print("\n--- Layer 3 (HerPNConv) ---")
    out = backbone.layer3(out)
    if check_tensor("layer3 output", out, "  "):
        return None

    # Layer 4
    print("\n--- Layer 4 (HerPNConv) ---")
    out = backbone.layer4(out)
    if check_tensor("layer4 output", out, "  "):
        return None

    # Layer 5
    print("\n--- Layer 5 (HerPNConv) ---")
    out = backbone.layer5(out)
    if check_tensor("layer5 output", out, "  "):
        return None

    # HerPNPool
    print("\n--- HerPNPool ---")
    out = backbone.herpnpool(out)
    if check_tensor("herpnpool output", out, "  "):
        return None

    # Final layers
    print("\n--- Final Layers ---")
    out = backbone.flatten(out)
    if check_tensor("flatten", out, "  "):
        return None

    out = backbone.bn(out)
    if check_tensor("bn (final)", out, "  "):
        return None

    print(f"\n{'='*100}")
    print(f"✓ {label} completed successfully!")
    print(f"{'='*100}\n")

    return out


def main():
    print(f"\n{'='*100}")
    print("Layer-by-Layer Comparison: Unfused vs Fused CryptoFaceNet4")
    print(f"{'='*100}")

    # Create input (normalized to [-1, 1])
    torch.manual_seed(42)
    x = torch.randn(1, 3, 64, 64)
    x = (x - x.min()) / (x.max() - x.min())  # [0, 1]
    x = 2 * x - 1  # [-1, 1]

    print(f"\nInput shape: {x.shape}")
    print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")

    # Extract first patch (top-left 32x32)
    patch = x[:, :, :32, :32]
    print(f"\nPatch shape: {patch.shape}")
    print(f"Patch range: [{patch.min():.4f}, {patch.max():.4f}]")

    # ========================================
    # Test 1: UNFUSED model
    # ========================================
    print("\n\n" + "="*100)
    print("TEST 1: UNFUSED MODEL")
    print("="*100)

    model_unfused = CryptoFaceNet4()
    load_checkpoint_for_config(model_unfused, input_size=64, verbose=False)
    model_unfused.eval()

    # Test backbone 0 (unfused)
    with torch.no_grad():
        out_unfused = trace_backbone_layer_by_layer(
            model_unfused.nets[0],
            patch,
            label="UNFUSED Backbone[0]",
            fused=False
        )

    if out_unfused is not None:
        print(f"\n✓ UNFUSED model output range: [{out_unfused.min():.4f}, {out_unfused.max():.4f}]")
    else:
        print(f"\n✗ UNFUSED model failed!")

    # ========================================
    # Test 2: FUSED model
    # ========================================
    print("\n\n" + "="*100)
    print("TEST 2: FUSED MODEL")
    print("="*100)

    model_fused = CryptoFaceNet4()
    load_checkpoint_for_config(model_fused, input_size=64, verbose=False)
    model_fused.init_orion_params()  # Fuse
    model_fused.eval()

    # Test backbone 0 (fused)
    with torch.no_grad():
        out_fused = trace_backbone_layer_by_layer(
            model_fused.nets[0],
            patch,
            label="FUSED Backbone[0]",
            fused=True
        )

    if out_fused is not None:
        print(f"\n✓ FUSED model output range: [{out_fused.min():.4f}, {out_fused.max():.4f}]")
    else:
        print(f"\n✗ FUSED model failed!")

    # ========================================
    # Comparison
    # ========================================
    print("\n\n" + "="*100)
    print("COMPARISON")
    print("="*100)

    if out_unfused is not None and out_fused is not None:
        diff = (out_unfused - out_fused).abs()
        mae = diff.mean().item()
        max_err = diff.max().item()

        print(f"\nMAE: {mae:.6f}")
        print(f"Max error: {max_err:.6f}")

        if mae < 1e-3:
            print(f"\n✓ Models match! (MAE < 1e-3)")
        else:
            print(f"\n⚠️  Models diverge! (MAE = {mae:.6f})")
    else:
        print("\n✗ Cannot compare - one or both models failed")


if __name__ == "__main__":
    main()
