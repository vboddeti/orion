"""
Debug script to trace NaN issue through a single patch.
"""
import sys
import torch
import numpy as np

sys.path.append('/research/hal-vishnu/code/orion-fhe')
sys.path.append('/research/hal-vishnu/code/orion-fhe/CryptoFace')

from CryptoFace.models import PatchCNN as CryptoFacePatchCNN
from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4
from face_recognition.models.weight_loader import load_checkpoint_for_config


def debug_orion_backbone():
    """Debug Orion backbone to find NaN source."""
    print("="*80)
    print("DEBUG: Orion Backbone Forward Pass")
    print("="*80)

    # Load Orion model
    model = CryptoFaceNet4()
    load_checkpoint_for_config(model, input_size=64, verbose=False)

    # Fuse BatchNorm
    print("\nCalling init_orion_params()...")
    model.init_orion_params()
    model.eval()

    # Generate test input (single patch)
    torch.manual_seed(42)
    patch = torch.randn(1, 3, 32, 32) * 0.3
    patch = torch.clamp(patch, -1, 1)

    print(f"\nInput patch shape: {patch.shape}")
    print(f"Input range: [{patch.min():.4f}, {patch.max():.4f}]")
    print(f"Input mean: {patch.mean():.4f}, std: {patch.std():.4f}")

    # Process through first backbone
    backbone = model.nets[0]

    # Trace through layers
    with torch.no_grad():
        print("\n" + "-"*80)
        print("Layer 1: Initial Conv")
        print("-"*80)
        out = backbone.conv(patch)
        print(f"  Output shape: {out.shape}")
        print(f"  Output range: [{out.min():.4f}, {out.max():.4f}]")
        print(f"  Has NaN: {torch.isnan(out).any()}")

        print("\n" + "-"*80)
        print("Layer 2: layer1 (HerPNConv)")
        print("-"*80)
        out = backbone.layer1(out)
        print(f"  Output shape: {out.shape}")
        print(f"  Output range: [{out.min():.4f}, {out.max():.4f}]")
        print(f"  Has NaN: {torch.isnan(out).any()}")

        if torch.isnan(out).any():
            print("\n*** NaN detected in layer1! ***")
            print("Checking layer1 components...")

            # Re-run with intermediate checks
            test_in = backbone.conv(patch)

            # Check if herpn1 exists
            if hasattr(backbone.layer1, 'herpn1') and backbone.layer1.herpn1 is not None:
                print("\nChecking herpn1...")
                herpn1_out = backbone.layer1.herpn1(test_in)
                print(f"  herpn1 output range: [{herpn1_out.min():.4f}, {herpn1_out.max():.4f}]")
                print(f"  herpn1 has NaN: {torch.isnan(herpn1_out).any()}")

                if not torch.isnan(herpn1_out).any():
                    print("\nChecking conv1...")
                    conv1_out = backbone.layer1.conv1(herpn1_out)
                    print(f"  conv1 output range: [{conv1_out.min():.4f}, {conv1_out.max():.4f}]")
                    print(f"  conv1 has NaN: {torch.isnan(conv1_out).any()}")

                    if not torch.isnan(conv1_out).any():
                        print("\nChecking herpn2...")
                        herpn2_out = backbone.layer1.herpn2(conv1_out)
                        print(f"  herpn2 output range: [{herpn2_out.min():.4f}, {herpn2_out.max():.4f}]")
                        print(f"  herpn2 has NaN: {torch.isnan(herpn2_out).any()}")
            else:
                print("\nherpn1 is None! Using unfused BatchNorm path")
                print("This should not happen after init_orion_params()")

            return

        print("\n" + "-"*80)
        print("Layer 3: layer2 (HerPNConv, stride=2)")
        print("-"*80)
        out = backbone.layer2(out)
        print(f"  Output shape: {out.shape}")
        print(f"  Output range: [{out.min():.4f}, {out.max():.4f}]")
        print(f"  Has NaN: {torch.isnan(out).any()}")

        if torch.isnan(out).any():
            print("\n*** NaN detected in layer2! ***")
            return

        print("\n" + "-"*80)
        print("Layer 4: layer3 (HerPNConv)")
        print("-"*80)
        out = backbone.layer3(out)
        print(f"  Output shape: {out.shape}")
        print(f"  Output range: [{out.min():.4f}, {out.max():.4f}]")
        print(f"  Has NaN: {torch.isnan(out).any()}")

        if torch.isnan(out).any():
            print("\n*** NaN detected in layer3! ***")
            return

        print("\n" + "-"*80)
        print("Layer 5: layer4 (HerPNConv, stride=2)")
        print("-"*80)
        out = backbone.layer4(out)
        print(f"  Output shape: {out.shape}")
        print(f"  Output range: [{out.min():.4f}, {out.max():.4f}]")
        print(f"  Has NaN: {torch.isnan(out).any()}")

        if torch.isnan(out).any():
            print("\n*** NaN detected in layer4! ***")
            return

        print("\n" + "-"*80)
        print("Layer 6: layer5 (HerPNConv)")
        print("-"*80)
        out = backbone.layer5(out)
        print(f"  Output shape: {out.shape}")
        print(f"  Output range: [{out.min():.4f}, {out.max():.4f}]")
        print(f"  Has NaN: {torch.isnan(out).any()}")

        if torch.isnan(out).any():
            print("\n*** NaN detected in layer5! ***")
            return

        print("\n" + "-"*80)
        print("Layer 7: herpnpool")
        print("-"*80)
        out = backbone.herpnpool(out)
        print(f"  Output shape: {out.shape}")
        print(f"  Output range: [{out.min():.4f}, {out.max():.4f}]")
        print(f"  Has NaN: {torch.isnan(out).any()}")

        if torch.isnan(out).any():
            print("\n*** NaN detected in herpnpool! ***")
            return

        print("\n" + "-"*80)
        print("Layer 8: flatten")
        print("-"*80)
        out = backbone.flatten(out)
        print(f"  Output shape: {out.shape}")
        print(f"  Output range: [{out.min():.4f}, {out.max():.4f}]")
        print(f"  Has NaN: {torch.isnan(out).any()}")

        print("\n" + "-"*80)
        print("Layer 9: final bn")
        print("-"*80)
        out = backbone.bn(out)
        print(f"  Output shape: {out.shape}")
        print(f"  Output range: [{out.min():.4f}, {out.max():.4f}]")
        print(f"  Has NaN: {torch.isnan(out).any()}")

        if torch.isnan(out).any():
            print("\n*** NaN detected in final bn! ***")
            return

        if torch.isnan(out).any():
            print("\n*** NaN detected in final bn! ***")
            return

        print("\n" + "-"*80)
        print("Layer 10: linear[0]")
        print("-"*80)
        linear_out = model.linear[0](out)
        print(f"  Output shape: {linear_out.shape}")
        print(f"  Output range: [{linear_out.min():.4f}, {linear_out.max():.4f}]")
        print(f"  Has NaN: {torch.isnan(linear_out).any()}")

    print("\n" + "="*80)
    if torch.isnan(linear_out).any():
        print("✗ NaN detected in linear layer!")
    else:
        print("✓ No NaN detected through full pipeline!")
    print("="*80)


if __name__ == "__main__":
    debug_orion_backbone()
