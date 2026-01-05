"""
Test backbone WITHOUT final pooling/flatten/batchnorm layers.

This tests just the conv + HerPN layers to isolate the FHE issue.
"""
import sys
import torch

sys.path.append('/research/hal-vishnu/code/orion-fhe')

import orion
import orion.nn as on
from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4
from face_recognition.models.weight_loader import load_checkpoint_for_config


class BackboneWithoutPooling(on.Module):
    """Backbone with just conv layers and HerPN (no pooling/flatten/bn at end)."""

    def __init__(self, backbone):
        super().__init__()
        # Copy just the conv and HerPN layers
        self.conv = backbone.conv
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.layer5 = backbone.layer5

    def forward(self, x):
        out = self.conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out


def main():
    print(f"\n{'='*80}")
    print("Testing Backbone WITHOUT Pooling (Conv + HerPN only)")
    print(f"{'='*80}")

    # Load full model
    print("\nLoading model...")
    full_model = CryptoFaceNet4()
    load_checkpoint_for_config(full_model, input_size=64, verbose=False)
    full_model.init_orion_params()

    # Create simplified backbone
    backbone = full_model.nets[0]
    simple_backbone = BackboneWithoutPooling(backbone)

    print("✓ Created simplified backbone (no pooling/flatten/bn)")

    # Test input
    torch.manual_seed(42)
    test_patch = torch.randn(1, 3, 32, 32)
    print(f"✓ Test input: {test_patch.shape}")

    # Cleartext inference
    print("\nCleartext inference...")
    simple_backbone.eval()
    with torch.no_grad():
        cleartext_out = simple_backbone(test_patch)
    print(f"✓ Cleartext output: {cleartext_out.shape}")
    print(f"  Range: [{cleartext_out.min():.6f}, {cleartext_out.max():.6f}]")

    # FHE setup
    print("\nInitializing FHE...")
    orion.init_scheme("configs/cryptoface_net4.yml")

    print("Tracing and compiling...")
    orion.fit(simple_backbone, test_patch)
    input_level = orion.compile(simple_backbone)
    print(f"✓ Compiled, input_level={input_level}")

    # FHE inference
    print("\nFHE inference...")
    vec_ptxt = orion.encode(test_patch, input_level)
    vec_ctxt = orion.encrypt(vec_ptxt)

    simple_backbone.he()
    out_ctxt = simple_backbone(vec_ctxt)

    fhe_out = out_ctxt.decrypt().decode()
    print(f"✓ FHE output: {fhe_out.shape}")
    print(f"  Range: [{fhe_out.min():.6f}, {fhe_out.max():.6f}]")

    # Compare
    print(f"\n{'='*80}")
    print("Comparison")
    print(f"{'='*80}")

    if cleartext_out.shape != fhe_out.shape:
        print(f"✗ Shape mismatch: {cleartext_out.shape} vs {fhe_out.shape}")
        return

    diff = (cleartext_out - fhe_out).abs()
    mae = diff.mean().item()
    max_err = diff.max().item()

    print(f"MAE: {mae:.6f}")
    print(f"Max error: {max_err:.6f}")

    if mae < 1.0:
        print(f"\n✓ TEST PASSED (MAE < 1.0)")
    else:
        print(f"\n✗ TEST FAILED (MAE >= 1.0)")


if __name__ == "__main__":
    main()
