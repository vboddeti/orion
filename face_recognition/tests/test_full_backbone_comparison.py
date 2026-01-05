"""
Test full backbone (all 5 layers) comparison between CryptoFace and ours.
"""
import sys
import torch

sys.path.append('/research/hal-vishnu/code/orion-fhe')
sys.path.append('/research/hal-vishnu/code/orion-fhe/CryptoFace')

import argparse
from CryptoFace.models import build_model
from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4
from face_recognition.models.weight_loader import load_checkpoint_for_config


def main():
    print(f"\n{'='*80}")
    print("Full Backbone Comparison: CryptoFace vs Ours")
    print(f"{'='*80}\n")

    # Create identical input
    torch.manual_seed(42)
    x = torch.randn(1, 3, 32, 32)
    x = (x - x.min()) / (x.max() - x.min()) * 2 - 1

    print(f"Input: {x.shape}, range=[{x.min():.4f}, {x.max():.4f}]\n")

    # ==================== CryptoFace ====================
    print("="*80)
    print("CryptoFace (UNFUSED)")
    print("="*80)

    args = argparse.Namespace()
    args.arch = 'cryptoface'
    args.input_size = 64
    args.patch_size = 32
    args.num_classes = 93431

    cf_model = build_model(args)
    ckpt = torch.load('/research/hal-vishnu/code/orion-fhe/face_recognition/checkpoints/backbone-64x64.ckpt',
                     map_location='cpu', weights_only=False)
    cf_model.load_state_dict(ckpt['backbone'])
    cf_model.eval()

    backbone_cf = cf_model.nets[0]

    with torch.no_grad():
        # Run through full backbone (unfused)
        out_cf = backbone_cf.conv(x)
        for layer in backbone_cf.layers:
            out_cf = layer(out_cf)
        out_cf = backbone_cf.herpnpool(out_cf)
        out_cf = backbone_cf.flatten(out_cf)
        out_cf_unfused = backbone_cf.bn(out_cf)

    print(f"Final output: [{out_cf_unfused.min():.4f}, {out_cf_unfused.max():.4f}], mean={out_cf_unfused.mean():.4f}\n")

    # ==================== CryptoFace FUSED ====================
    print("="*80)
    print("CryptoFace (FUSED)")
    print("="*80)

    cf_model.fuse()

    with torch.no_grad():
        # Run through full backbone (fused)
        out_cf_fused = backbone_cf.conv(x)
        for layer in backbone_cf.layers:
            out_cf_fused = layer.forward_fuse(out_cf_fused)
        # HerPNPool fused
        out_cf_fused = backbone_cf.herpnpool.forward_fuse(out_cf_fused)
        out_cf_fused = backbone_cf.flatten(out_cf_fused)
        out_cf_fused = backbone_cf.bn(out_cf_fused)

    print(f"Final output: [{out_cf_fused.min():.4f}, {out_cf_fused.max():.4f}], mean={out_cf_fused.mean():.4f}\n")

    # ==================== Our Implementation ====================
    print("="*80)
    print("Our Implementation (UNFUSED)")
    print("="*80)

    our_model = CryptoFaceNet4()
    load_checkpoint_for_config(our_model, input_size=64, verbose=False)
    our_model.eval()

    backbone_ours = our_model.nets[0]

    with torch.no_grad():
        out_ours_unfused = backbone_ours(x)

    print(f"Final output: [{out_ours_unfused.min():.4f}, {out_ours_unfused.max():.4f}], mean={out_ours_unfused.mean():.4f}\n")

    # ==================== Our Implementation FUSED ====================
    print("="*80)
    print("Our Implementation (FUSED)")
    print("="*80)

    our_model_fused = CryptoFaceNet4()
    load_checkpoint_for_config(our_model_fused, input_size=64, verbose=False)
    our_model_fused.init_orion_params()
    our_model_fused.eval()

    backbone_ours_fused = our_model_fused.nets[0]

    with torch.no_grad():
        out_ours_fused = backbone_ours_fused(x)

    print(f"Final output: [{out_ours_fused.min():.4f}, {out_ours_fused.max():.4f}], mean={out_ours_fused.mean():.4f}\n")

    # ==================== Comparison ====================
    print("="*80)
    print("COMPARISON")
    print("="*80 + "\n")

    # Compare unfused
    diff_unfused = (out_cf_unfused - out_ours_unfused).abs()
    print(f"Unfused - CryptoFace vs Ours:")
    print(f"  MAE: {diff_unfused.mean():.6f}")
    print(f"  Max diff: {diff_unfused.max():.6f}")

    # Compare fused
    diff_fused = (out_cf_fused - out_ours_fused).abs()
    print(f"\nFused - CryptoFace vs Ours:")
    print(f"  MAE: {diff_fused.mean():.6f}")
    print(f"  Max diff: {diff_fused.max():.6f}")

    # CryptoFace: unfused vs fused
    diff_cf = (out_cf_unfused - out_cf_fused).abs()
    print(f"\nCryptoFace - Unfused vs Fused:")
    print(f"  MAE: {diff_cf.mean():.6f}")
    print(f"  Max diff: {diff_cf.max():.6f}")

    # Ours: unfused vs fused
    diff_ours = (out_ours_unfused - out_ours_fused).abs()
    print(f"\nOurs - Unfused vs Fused:")
    print(f"  MAE: {diff_ours.mean():.6f}")
    print(f"  Max diff: {diff_ours.max():.6f}")

    # Success check
    print(f"\n{'='*80}")
    if diff_unfused.mean() < 0.01 and diff_fused.mean() < 0.01:
        print("✓ SUCCESS: Both unfused and fused match CryptoFace!")
    else:
        print("✗ FAILED: Outputs diverge from CryptoFace")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
