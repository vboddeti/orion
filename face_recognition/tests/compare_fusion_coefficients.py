"""
Compare fused coefficients between CryptoFace and our implementation.
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
    print("Comparing Fused Coefficients: CryptoFace vs Ours")
    print(f"{'='*80}\n")

    # CryptoFace fusion
    print("Loading CryptoFace model...")
    args = argparse.Namespace()
    args.arch = 'cryptoface'
    args.input_size = 64
    args.patch_size = 32
    args.num_classes = 93431

    backbone_cf = build_model(args)
    ckpt = torch.load('/research/hal-vishnu/code/orion-fhe/face_recognition/checkpoints/backbone-64x64.ckpt',
                     map_location='cpu', weights_only=False)
    backbone_cf.load_state_dict(ckpt['backbone'])
    backbone_cf.fuse()

    layer1_cf = backbone_cf.nets[0].layers[0]

    print('✓ CryptoFace fused\n')

    # Our fusion
    print("Loading our model...")
    model_ours = CryptoFaceNet4()
    load_checkpoint_for_config(model_ours, input_size=64, verbose=False)
    model_ours.init_orion_params()

    layer1_ours = model_ours.nets[0].layer1

    print('✓ Our model fused\n')

    # Compare coefficients
    print(f"{'='*80}")
    print("Coefficient Comparison for layer1.herpn1")
    print(f"{'='*80}\n")

    print(f"{'Coefficient':<15} {'CryptoFace Range':<35} {'Our Range':<35}")
    print(f"{'-'*85}")

    cf_a0_min, cf_a0_max = layer1_cf.a0.min().item(), layer1_cf.a0.max().item()
    cf_a1_min, cf_a1_max = layer1_cf.a1.min().item(), layer1_cf.a1.max().item()
    cf_a2_min, cf_a2_max = layer1_cf.a2.min().item(), layer1_cf.a2.max().item()

    our_a0_min, our_a0_max = layer1_ours.herpn1.weight0_raw.min().item(), layer1_ours.herpn1.weight0_raw.max().item()
    our_a1_min, our_a1_max = layer1_ours.herpn1.weight1_raw.min().item(), layer1_ours.herpn1.weight1_raw.max().item()
    our_a2_min, our_a2_max = layer1_ours.herpn1.scale_factor.min().item(), layer1_ours.herpn1.scale_factor.max().item()

    print(f"a0 (const)     [{cf_a0_min:>10.4f}, {cf_a0_max:>10.4f}]       [{our_a0_min:>10.4f}, {our_a0_max:>10.4f}]")
    print(f"a1 (linear)    [{cf_a1_min:>10.4f}, {cf_a1_max:>10.4f}]       [{our_a1_min:>10.4f}, {our_a1_max:>10.4f}]")
    print(f"a2 (quad/w2)   [{cf_a2_min:>10.4f}, {cf_a2_max:>10.4f}]       [{our_a2_min:>10.4f}, {our_a2_max:>10.4f}]")

    # Check if they match
    print(f"\n{'='*80}")
    print("Match Analysis")
    print(f"{'='*80}\n")

    a0_match = torch.allclose(layer1_cf.a0, layer1_ours.herpn1.weight0_raw, atol=1e-5)
    a1_match = torch.allclose(layer1_cf.a1, layer1_ours.herpn1.weight1_raw, atol=1e-5)
    a2_match = torch.allclose(layer1_cf.a2, layer1_ours.herpn1.scale_factor, atol=1e-5)

    print(f"a0 matches: {a0_match}")
    print(f"a1 matches: {a1_match}")
    print(f"a2 matches: {a2_match}")

    if a0_match and a1_match and a2_match:
        print(f"\n✓ All coefficients match!")
    else:
        print(f"\n✗ Coefficients differ!")

        # Show max difference
        if not a0_match:
            diff_a0 = (layer1_cf.a0 - layer1_ours.herpn1.weight0_raw).abs().max().item()
            print(f"  Max a0 difference: {diff_a0:.6f}")
        if not a1_match:
            diff_a1 = (layer1_cf.a1 - layer1_ours.herpn1.weight1_raw).abs().max().item()
            print(f"  Max a1 difference: {diff_a1:.6f}")
        if not a2_match:
            diff_a2 = (layer1_cf.a2 - layer1_ours.herpn1.scale_factor).abs().max().item()
            print(f"  Max a2 difference: {diff_a2:.6f}")


if __name__ == "__main__":
    main()
