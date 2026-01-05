"""
Compare CryptoFace fused vs our fused implementation layer-by-layer.
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
    print("Fused Layer-by-Layer Comparison: CryptoFace vs Ours")
    print(f"{'='*80}\n")

    # Create identical input
    torch.manual_seed(42)
    x = torch.randn(1, 3, 32, 32)
    x = (x - x.min()) / (x.max() - x.min()) * 2 - 1

    print(f"Input: {x.shape}, range=[{x.min():.4f}, {x.max():.4f}]\n")

    # ==================== CryptoFace ====================
    args = argparse.Namespace()
    args.arch = 'cryptoface'
    args.input_size = 64
    args.patch_size = 32
    args.num_classes = 93431

    cf_model = build_model(args)
    ckpt = torch.load('/research/hal-vishnu/code/orion-fhe/face_recognition/checkpoints/backbone-64x64.ckpt',
                     map_location='cpu', weights_only=False)
    cf_model.load_state_dict(ckpt['backbone'])
    cf_model.fuse()
    cf_model.eval()

    backbone_cf = cf_model.nets[0]

    # ==================== Our Implementation ====================
    our_model = CryptoFaceNet4()
    load_checkpoint_for_config(our_model, input_size=64, verbose=False)
    our_model.init_orion_params()
    our_model.eval()

    backbone_ours = our_model.nets[0]

    # ==================== Layer-by-Layer Comparison ====================
    print("="*80)
    print("Layer-by-Layer Comparison (FUSED)")
    print("="*80 + "\n")

    with torch.no_grad():
        # Conv layer
        out_cf = backbone_cf.conv(x)
        out_ours = backbone_ours.conv(x)
        diff = (out_cf - out_ours).abs()
        print(f"conv:")
        print(f"  CryptoFace: [{out_cf.min():.4f}, {out_cf.max():.4f}]")
        print(f"  Ours:       [{out_ours.min():.4f}, {out_ours.max():.4f}]")
        print(f"  MAE: {diff.mean():.6f}, Max: {diff.max():.6f}\n")

        # Layer 1
        out_cf = backbone_cf.layers[0].forward_fuse(out_cf)
        out_ours = backbone_ours.layer1(out_ours)
        diff = (out_cf - out_ours).abs()
        print(f"layer1:")
        print(f"  CryptoFace: [{out_cf.min():.4f}, {out_cf.max():.4f}]")
        print(f"  Ours:       [{out_ours.min():.4f}, {out_ours.max():.4f}]")
        print(f"  MAE: {diff.mean():.6f}, Max: {diff.max():.6f}\n")

        if diff.mean() > 0.001:
            print("⚠️  Divergence detected at layer1!\n")
            return

        # Layer 2
        out_cf = backbone_cf.layers[1].forward_fuse(out_cf)
        out_ours = backbone_ours.layer2(out_ours)
        diff = (out_cf - out_ours).abs()
        print(f"layer2:")
        print(f"  CryptoFace: [{out_cf.min():.4f}, {out_cf.max():.4f}]")
        print(f"  Ours:       [{out_ours.min():.4f}, {out_ours.max():.4f}]")
        print(f"  MAE: {diff.mean():.6f}, Max: {diff.max():.6f}\n")

        if diff.mean() > 0.001:
            print("⚠️  Divergence detected at layer2!\n")
            return

        # Layer 3
        out_cf = backbone_cf.layers[2].forward_fuse(out_cf)
        out_ours = backbone_ours.layer3(out_ours)
        diff = (out_cf - out_ours).abs()
        print(f"layer3:")
        print(f"  CryptoFace: [{out_cf.min():.4f}, {out_cf.max():.4f}]")
        print(f"  Ours:       [{out_ours.min():.4f}, {out_ours.max():.4f}]")
        print(f"  MAE: {diff.mean():.6f}, Max: {diff.max():.6f}\n")

        if diff.mean() > 0.001:
            print("⚠️  Divergence detected at layer3!\n")
            return

        # Layer 4
        out_cf = backbone_cf.layers[3].forward_fuse(out_cf)
        out_ours = backbone_ours.layer4(out_ours)
        diff = (out_cf - out_ours).abs()
        print(f"layer4:")
        print(f"  CryptoFace: [{out_cf.min():.4f}, {out_cf.max():.4f}]")
        print(f"  Ours:       [{out_ours.min():.4f}, {out_ours.max():.4f}]")
        print(f"  MAE: {diff.mean():.6f}, Max: {diff.max():.6f}\n")

        if diff.mean() > 0.001:
            print("⚠️  Divergence detected at layer4!\n")
            return

        # Layer 5
        out_cf = backbone_cf.layers[4].forward_fuse(out_cf)
        out_ours = backbone_ours.layer5(out_ours)
        diff = (out_cf - out_ours).abs()
        print(f"layer5:")
        print(f"  CryptoFace: [{out_cf.min():.4f}, {out_cf.max():.4f}]")
        print(f"  Ours:       [{out_ours.min():.4f}, {out_ours.max():.4f}]")
        print(f"  MAE: {diff.mean():.6f}, Max: {diff.max():.6f}\n")

        if diff.mean() > 0.001:
            print("⚠️  Divergence detected at layer5!\n")
            return

        # HerPNPool
        out_cf = backbone_cf.herpnpool.forward_fuse(out_cf)
        out_ours = backbone_ours.herpnpool(out_ours)
        diff = (out_cf - out_ours).abs()
        print(f"herpnpool:")
        print(f"  CryptoFace: [{out_cf.min():.4f}, {out_cf.max():.4f}]")
        print(f"  Ours:       [{out_ours.min():.4f}, {out_ours.max():.4f}]")
        print(f"  MAE: {diff.mean():.6f}, Max: {diff.max():.6f}\n")

        if diff.mean() > 0.001:
            print("⚠️  Divergence detected at herpnpool!\n")
            return

        # Flatten
        out_cf = backbone_cf.flatten(out_cf)
        out_ours = backbone_ours.flatten(out_ours)
        diff = (out_cf - out_ours).abs()
        print(f"flatten:")
        print(f"  MAE: {diff.mean():.6f}, Max: {diff.max():.6f}\n")

        # Final BN
        out_cf = backbone_cf.bn(out_cf)
        out_ours = backbone_ours.bn(out_ours)
        diff = (out_cf - out_ours).abs()
        print(f"bn (final):")
        print(f"  CryptoFace: [{out_cf.min():.4f}, {out_cf.max():.4f}]")
        print(f"  Ours:       [{out_ours.min():.4f}, {out_ours.max():.4f}]")
        print(f"  MAE: {diff.mean():.6f}, Max: {diff.max():.6f}\n")

        if diff.mean() < 0.01:
            print("✓ All layers match CryptoFace!")
        else:
            print("✗ Final output diverges from CryptoFace")


if __name__ == "__main__":
    main()
