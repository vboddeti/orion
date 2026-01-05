"""
Compare conv1 weights between CryptoFace and Orion after fusion.
"""
import sys
import torch
import numpy as np

sys.path.append('/research/hal-vishnu/code/orion-fhe')
sys.path.append('/research/hal-vishnu/code/orion-fhe/CryptoFace')

from CryptoFace.models import PatchCNN as CryptoFacePatchCNN
from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4
from face_recognition.models.weight_loader import load_checkpoint_for_config


def compare_weights(name, cf_weight, orion_weight):
    """Compare two weight tensors."""
    print(f"\n{name}:")
    print(f"  CryptoFace shape: {cf_weight.shape}")
    print(f"  Orion shape:      {orion_weight.shape}")
    print(f"  CryptoFace range: [{cf_weight.min():.6f}, {cf_weight.max():.6f}]")
    print(f"  Orion range:      [{orion_weight.min():.6f}, {orion_weight.max():.6f}]")

    diff = (cf_weight - orion_weight).abs()
    rel_diff = diff / (cf_weight.abs() + 1e-8)

    print(f"  Abs diff: max={diff.max():.6f}, mean={diff.mean():.6f}")
    print(f"  Rel diff: max={rel_diff.max():.6f}, mean={rel_diff.mean():.6f}")

    if diff.max() < 1e-6:
        print(f"  ✓ Weights MATCH")
    else:
        print(f"  ✗ Weights DIFFER")

        # Sample a few weights to see the difference
        print(f"\n  Sample weights (first 3):")
        for i in range(min(3, cf_weight.shape[0])):
            print(f"    Channel {i}:")
            print(f"      CryptoFace: {cf_weight[i, 0, 0, 0]:.6f}")
            print(f"      Orion:      {orion_weight[i, 0, 0, 0]:.6f}")
            print(f"      Diff:       {(cf_weight[i, 0, 0, 0] - orion_weight[i, 0, 0, 0]).abs():.6f}")


def main():
    patch_idx = 0

    print(f"\n{'='*80}")
    print(f"CONV WEIGHT COMPARISON (Patch {patch_idx})")
    print(f"{'='*80}")

    # Load CryptoFace model
    print(f"\nLoading CryptoFace model...")
    cryptoface_model = CryptoFacePatchCNN(input_size=64, patch_size=32)
    ckpt = torch.load("face_recognition/checkpoints/backbone-64x64.ckpt",
                      map_location='cpu', weights_only=False)
    cryptoface_model.load_state_dict(ckpt['backbone'], strict=False)
    cryptoface_model.eval()
    cryptoface_model.fuse()

    cf_backbone = cryptoface_model.nets[patch_idx]
    cf_layer1 = cf_backbone.layers[0]

    # Load Orion model
    print(f"\nLoading Orion model...")
    orion_model = CryptoFaceNet4()
    load_checkpoint_for_config(orion_model, input_size=64, verbose=False)
    orion_model.init_orion_params()
    orion_model.eval()

    orion_backbone = orion_model.nets[patch_idx]
    orion_layer1 = orion_backbone.layer1

    print(f"\n{'='*80}")
    print(f"Layer 1 Weight Comparison")
    print(f"{'='*80}")

    # Compare HerPN1 coefficients
    print(f"\n--- HerPN1 Coefficients ---")
    print(f"CryptoFace a0: {cf_layer1.a0[0, 0, 0]:.6f}")
    print(f"Orion weight0: {orion_layer1.herpn1.weight0_raw[0, 0, 0]:.6f}")
    print(f"Diff:          {(cf_layer1.a0[0, 0, 0] - orion_layer1.herpn1.weight0_raw[0, 0, 0]).abs():.6f}")

    print(f"\nCryptoFace a1: {cf_layer1.a1[0, 0, 0]:.6f}")
    print(f"Orion weight1: {orion_layer1.herpn1.weight1_raw[0, 0, 0]:.6f}")
    print(f"Diff:          {(cf_layer1.a1[0, 0, 0] - orion_layer1.herpn1.weight1_raw[0, 0, 0]).abs():.6f}")

    print(f"\nCryptoFace a2: {cf_layer1.a2[0, 0, 0]:.6f}")
    print(f"Orion scale:   {orion_layer1.herpn1.scale_factor[0, 0, 0]:.6f}")
    print(f"Diff:          {(cf_layer1.a2[0, 0, 0] - orion_layer1.herpn1.scale_factor[0, 0, 0]).abs():.6f}")

    # Compare Conv1 weights
    print(f"\n--- Conv1 Weights ---")
    print(f"CryptoFace uses weight1 (pre-scaled by a2)")
    print(f"Orion modified conv1.weight in-place (scaled by scale_factor)")

    compare_weights("Conv1 Weight", cf_layer1.weight1, orion_layer1.conv1.weight)

    # Also check if the original conv1 weight was the same before scaling
    print(f"\n--- Checking Original Conv1 Weight (Before Scaling) ---")
    print(f"CryptoFace: weight1 / a2 should equal original conv1.weight")
    cf_original = cf_layer1.weight1 / cf_layer1.a2
    orion_original = orion_layer1.conv1.weight / orion_layer1.herpn1.scale_factor
    compare_weights("Original Conv1 Weight (unscaled)", cf_original, orion_original)

    # Compare HerPN2 and Conv2 as well
    print(f"\n{'='*80}")
    print(f"HerPN2 and Conv2")
    print(f"{'='*80}")

    print(f"\n--- HerPN2 Coefficients ---")
    print(f"CryptoFace b0: {cf_layer1.b0[0, 0, 0]:.6f}")
    print(f"Orion weight0: {orion_layer1.herpn2.weight0_raw[0, 0, 0]:.6f}")
    print(f"Diff:          {(cf_layer1.b0[0, 0, 0] - orion_layer1.herpn2.weight0_raw[0, 0, 0]).abs():.6f}")

    print(f"\nCryptoFace b1: {cf_layer1.b1[0, 0, 0]:.6f}")
    print(f"Orion weight1: {orion_layer1.herpn2.weight1_raw[0, 0, 0]:.6f}")
    print(f"Diff:          {(cf_layer1.b1[0, 0, 0] - orion_layer1.herpn2.weight1_raw[0, 0, 0]).abs():.6f}")

    print(f"\nNote: CryptoFace bakes b2 into weight2, while Orion stores it as scale_factor")
    print(f"      So we check if weight2 == conv2.weight when both are unscaled by their scale factors")

    print(f"\n--- Conv2 Weights ---")
    compare_weights("Conv2 Weight", cf_layer1.weight2, orion_layer1.conv2.weight)

    # Check Conv bias
    print(f"\n--- Conv Bias Check ---")
    print(f"Conv1 has bias: CryptoFace={cf_layer1.conv1.bias is not None}, Orion={orion_layer1.conv1.bias is not None}")
    print(f"Conv2 has bias: CryptoFace={cf_layer1.conv2.bias is not None}, Orion={orion_layer1.conv2.bias is not None}")


if __name__ == "__main__":
    main()
