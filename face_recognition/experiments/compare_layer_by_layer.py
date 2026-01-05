"""
Layer-by-layer comparison between CryptoFace and Orion models.

This script compares intermediate outputs to find where they diverge.
"""
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

sys.path.append('/research/hal-vishnu/code/orion-fhe')
sys.path.append('/research/hal-vishnu/code/orion-fhe/CryptoFace')

from CryptoFace.models import PatchCNN as CryptoFacePatchCNN
from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4
from face_recognition.models.weight_loader import load_checkpoint_for_config


def load_lfw_image(image_path, input_size=64):
    """Load and preprocess LFW image."""
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(input_size, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    img_tensor = transform(img)
    return img_tensor.unsqueeze(0)


def compare_tensors(name, cf_output, orion_output, verbose=True):
    """Compare two tensors and print statistics."""
    cf_has_nan = torch.isnan(cf_output).any().item()
    orion_has_nan = torch.isnan(orion_output).any().item()

    if verbose:
        print(f"\n{name}:")
        print(f"  CryptoFace: shape={cf_output.shape}, nan={cf_has_nan}, "
              f"range=[{cf_output.min():.4f}, {cf_output.max():.4f}]")
        print(f"  Orion:      shape={orion_output.shape}, nan={orion_has_nan}, "
              f"range=[{orion_output.min():.4f}, {orion_output.max():.4f}]")

        if not cf_has_nan and not orion_has_nan:
            diff = (cf_output - orion_output).abs()
            rel_diff = diff / (cf_output.abs() + 1e-8)
            print(f"  Difference: max_abs={diff.max():.6f}, mean_abs={diff.mean():.6f}")
            print(f"              max_rel={rel_diff.max():.6f}, mean_rel={rel_diff.mean():.6f}")

            # Check if they match
            matches = diff.max() < 1e-4
            if matches:
                print(f"  ✓ Outputs MATCH (within tolerance)")
            else:
                print(f"  ✗ Outputs DIFFER")

            return matches
        elif orion_has_nan and not cf_has_nan:
            print(f"  ✗ DIVERGENCE: Orion has NaN, CryptoFace doesn't!")
            return False
        else:
            print(f"  ⚠ Both have NaN")
            return False

    return False


def compare_single_patch_backbone(patch_idx=0, verbose=True):
    """Compare a single patch backbone between CryptoFace and Orion."""

    print(f"\n{'='*80}")
    print(f"LAYER-BY-LAYER COMPARISON (Patch {patch_idx})")
    print(f"{'='*80}")

    # Load real image
    lfw_dir = Path("/research/hal-datastage/datasets/original/LFW/lfw-mtcnn-aligned")
    person_dirs = sorted([d for d in lfw_dir.iterdir() if d.is_dir()])
    first_person = person_dirs[0]
    images = sorted(list(first_person.glob("*.jpg")) + list(first_person.glob("*.png")))
    image_path = images[0]

    print(f"\nUsing image: {image_path}")
    img = load_lfw_image(image_path, input_size=64)

    # Extract single patch
    patches = []
    for h in range(2):
        for w in range(2):
            patch = img[:, :, h*32:(h+1)*32, w*32:(w+1)*32]
            patches.append(patch)

    patch = patches[patch_idx]
    print(f"Patch shape: {patch.shape}, range: [{patch.min():.4f}, {patch.max():.4f}]")

    # Load CryptoFace model
    print(f"\n{'='*80}")
    print(f"Loading CryptoFace model")
    print(f"{'='*80}")

    cryptoface_model = CryptoFacePatchCNN(input_size=64, patch_size=32)
    ckpt = torch.load("face_recognition/checkpoints/backbone-64x64.ckpt",
                      map_location='cpu', weights_only=False)
    cryptoface_model.load_state_dict(ckpt['backbone'], strict=False)
    cryptoface_model.eval()
    cryptoface_model.fuse()

    cf_backbone = cryptoface_model.nets[patch_idx]

    # Load Orion model
    print(f"\n{'='*80}")
    print(f"Loading Orion model")
    print(f"{'='*80}")

    orion_model = CryptoFaceNet4()
    load_checkpoint_for_config(orion_model, input_size=64, verbose=False)
    orion_model.init_orion_params()
    orion_model.eval()

    orion_backbone = orion_model.nets[patch_idx]

    # Compare layer by layer
    print(f"\n{'='*80}")
    print(f"Layer-by-layer comparison")
    print(f"{'='*80}")

    with torch.no_grad():
        # Input
        compare_tensors("Input patch", patch, patch, verbose=True)

        # Initial conv
        cf_out = cf_backbone.conv(patch)
        orion_out = orion_backbone.conv(patch)
        match = compare_tensors("After conv", cf_out, orion_out, verbose=True)

        if not match:
            print(f"\n⚠ First divergence at: conv layer")
            return

        # Layer 1 (layers[0] in CryptoFace, layer1 in Orion)
        cf_out = cf_backbone.layers[0].forward_fuse(cf_out)
        orion_out = orion_backbone.layer1(orion_out)
        match = compare_tensors("After layer1", cf_out, orion_out, verbose=True)

        if not match:
            print(f"\n⚠ First divergence at: layer1")
            # Deep dive into layer1
            print(f"\nDEEP DIVE INTO LAYER1:")
            print(f"="*80)
            deep_dive_herpnconv(cf_backbone.layers[0], orion_backbone.layer1,
                               cf_backbone.conv(patch), orion_backbone.conv(patch))
            return

        # Layer 2 (layers[1] in CryptoFace, layer2 in Orion) - has shortcut
        cf_out = cf_backbone.layers[1].forward_fuse(cf_out)
        orion_out = orion_backbone.layer2(orion_out)
        match = compare_tensors("After layer2 (stride=2, shortcut)", cf_out, orion_out, verbose=True)

        if not match:
            print(f"\n⚠ First divergence at: layer2")
            return

        # Layer 3 (layers[2] in CryptoFace, layer3 in Orion)
        cf_out = cf_backbone.layers[2].forward_fuse(cf_out)
        orion_out = orion_backbone.layer3(orion_out)
        match = compare_tensors("After layer3", cf_out, orion_out, verbose=True)

        if not match:
            print(f"\n⚠ First divergence at: layer3")
            return

        # Layer 4 (layers[3] in CryptoFace, layer4 in Orion) - has shortcut
        cf_out = cf_backbone.layers[3].forward_fuse(cf_out)
        orion_out = orion_backbone.layer4(orion_out)
        match = compare_tensors("After layer4 (stride=2, shortcut)", cf_out, orion_out, verbose=True)

        if not match:
            print(f"\n⚠ First divergence at: layer4")
            return

        # Layer 5 (layers[4] in CryptoFace, layer5 in Orion)
        cf_out = cf_backbone.layers[4].forward_fuse(cf_out)
        orion_out = orion_backbone.layer5(orion_out)
        match = compare_tensors("After layer5", cf_out, orion_out, verbose=True)

        if not match:
            print(f"\n⚠ First divergence at: layer5")
            return

        # HerPNPool
        cf_out = cf_backbone.herpnpool.forward_fuse(cf_out)
        orion_out = orion_backbone.herpnpool(orion_out)
        match = compare_tensors("After herpnpool", cf_out, orion_out, verbose=True)

        if not match:
            print(f"\n⚠ First divergence at: herpnpool")
            return

        # Flatten
        cf_out = cf_backbone.flatten(cf_out)
        orion_out = orion_backbone.flatten(orion_out)
        match = compare_tensors("After flatten", cf_out, orion_out, verbose=True)

        if not match:
            print(f"\n⚠ First divergence at: flatten")
            return

        # Final BatchNorm
        cf_out = cf_backbone.bn(cf_out)
        orion_out = orion_backbone.bn(orion_out)
        match = compare_tensors("After final bn", cf_out, orion_out, verbose=True)

        if not match:
            print(f"\n⚠ First divergence at: final bn")
            return

        print(f"\n{'='*80}")
        print(f"✓ ALL LAYERS MATCH!")
        print(f"{'='*80}")


def deep_dive_herpnconv(cf_layer, orion_layer, cf_input, orion_input):
    """Deep dive into a HerPNConv layer to find divergence."""
    print(f"\nComparing HerPNConv internal operations:")

    with torch.no_grad():
        # CryptoFace forward_fuse implementation
        # x = torch.square(x) + self.a1 * x + self.a0
        # out = F.conv2d(x, self.weight1, stride=self.conv1.stride, padding=self.conv1.padding)
        # out = torch.square(out) + self.b1 * out + self.b0
        # out = F.conv2d(out, self.weight2, stride=self.conv2.stride, padding=self.conv2.padding)
        # out += self.shortcut(x * self.a2)

        # Step 1: First HerPN
        cf_x = torch.square(cf_input) + cf_layer.a1 * cf_input + cf_layer.a0
        orion_x = orion_layer.herpn1(orion_input)
        compare_tensors("  HerPN1 output", cf_x, orion_x, verbose=True)

        # Step 2: First Conv
        cf_out = torch.nn.functional.conv2d(cf_x, cf_layer.weight1,
                                            stride=cf_layer.conv1.stride,
                                            padding=cf_layer.conv1.padding)
        orion_out = orion_layer.conv1(orion_x)
        compare_tensors("  Conv1 output", cf_out, orion_out, verbose=True)

        # Step 3: Second HerPN
        cf_out2 = torch.square(cf_out) + cf_layer.b1 * cf_out + cf_layer.b0
        orion_out2 = orion_layer.herpn2(orion_out)
        compare_tensors("  HerPN2 output", cf_out2, orion_out2, verbose=True)

        # Step 4: Second Conv
        cf_out3 = torch.nn.functional.conv2d(cf_out2, cf_layer.weight2,
                                             stride=cf_layer.conv2.stride,
                                             padding=cf_layer.conv2.padding)
        orion_out3 = orion_layer.conv2(orion_out2)
        compare_tensors("  Conv2 output", cf_out3, orion_out3, verbose=True)

        # Step 5: Shortcut
        # CryptoFace: shortcut = input_to_shortcut * scale_factor
        # where input_to_shortcut is either identity or conv+bn depending on has_shortcut
        if cf_layer.shortcut is not None and len(cf_layer.shortcut) > 0:
            cf_shortcut = cf_layer.shortcut(cf_x * cf_layer.a2)
            print(f"\n  CryptoFace has conv shortcut")
        else:
            cf_shortcut = cf_x * cf_layer.a2
            print(f"\n  CryptoFace identity shortcut (scaled)")

        # Orion: shortcut = herpn1(input) * scale_factor
        # then optionally apply conv+bn if has_shortcut
        orion_herpn1_for_shortcut = orion_layer.herpn1(orion_input)
        orion_shortcut_scaled = orion_herpn1_for_shortcut * orion_layer.herpn1.scale_factor

        compare_tensors("  Shortcut after HerPN+scale", cf_x * cf_layer.a2, orion_shortcut_scaled, verbose=True)

        if orion_layer.has_shortcut:
            orion_shortcut = orion_layer.shortcut_conv(orion_shortcut_scaled)
            orion_shortcut = orion_layer.shortcut_bn(orion_shortcut)
            print(f"  Orion has conv+bn shortcut")
        else:
            orion_shortcut = orion_shortcut_scaled
            print(f"  Orion identity shortcut")

        compare_tensors("  Shortcut final output", cf_shortcut, orion_shortcut, verbose=True)

        # Step 6: Final addition
        cf_final = cf_out3 + cf_shortcut
        orion_final = orion_out3 + orion_shortcut
        compare_tensors("  Final output (after add)", cf_final, orion_final, verbose=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Layer-by-layer comparison")
    parser.add_argument("--patch", type=int, default=0, help="Patch index to compare")

    args = parser.parse_args()

    compare_single_patch_backbone(patch_idx=args.patch, verbose=True)
