"""
Compare CryptoFace fused model vs Orion model AFTER full compilation.

This verifies that the compilation steps (fit + compile) don't change cleartext behavior.

Workflow:
- CryptoFace: load → fuse
- Orion: load → init_orion_params → fit → compile
- Compare: cleartext outputs layer-by-layer
"""
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

sys.path.append('/research/hal-vishnu/code/orion-fhe')
sys.path.append('/research/hal-vishnu/code/orion-fhe/CryptoFace')

import orion
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

            # Check if they match (more lenient tolerance after compile)
            matches = diff.max() < 1e-3
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


def compare_after_full_compile(patch_idx=0, verbose=True):
    """
    Compare CryptoFace vs Orion AFTER Orion has been fully compiled.

    CryptoFace: load → fuse
    Orion: load → init_orion_params → fit → compile
    """

    print(f"\n{'='*80}")
    print(f"COMPARISON AFTER FULL COMPILATION (Patch {patch_idx})")
    print(f"{'='*80}")

    # Load real image
    lfw_dir = Path("/research/hal-datastage/datasets/original/LFW/lfw-mtcnn-aligned")
    person_dirs = sorted([d for d in lfw_dir.iterdir() if d.is_dir()])
    first_person = person_dirs[0]
    images = sorted(list(first_person.glob("*.jpg")) + list(first_person.glob("*.png")))
    image_path = images[0]

    print(f"\nUsing image: {image_path}")
    full_img = load_lfw_image(image_path, input_size=64)

    # Extract single patch
    patches = []
    for h in range(2):
        for w in range(2):
            patch = full_img[:, :, h*32:(h+1)*32, w*32:(w+1)*32]
            patches.append(patch)

    patch = patches[patch_idx]
    print(f"Patch shape: {patch.shape}, range: [{patch.min():.4f}, {patch.max():.4f}]")

    # =========================================================================
    # CRYPTOFACE: Load and Fuse
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"CryptoFace Preprocessing")
    print(f"{'='*80}")

    cryptoface_model = CryptoFacePatchCNN(input_size=64, patch_size=32)
    ckpt = torch.load("face_recognition/checkpoints/backbone-64x64.ckpt",
                      map_location='cpu', weights_only=False)
    cryptoface_model.load_state_dict(ckpt['backbone'], strict=False)
    cryptoface_model.eval()

    print("Step 1: Load checkpoint ✓")

    cryptoface_model.fuse()
    print("Step 2: Fuse BatchNorm ✓")

    cf_backbone = cryptoface_model.nets[patch_idx]

    # =========================================================================
    # ORION: Load → Fuse → Fit → Compile
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"Orion Preprocessing (Full Pipeline)")
    print(f"{'='*80}")

    orion_model = CryptoFaceNet4()
    load_checkpoint_for_config(orion_model, input_size=64, verbose=False)
    orion_model.eval()
    print("Step 1: Load checkpoint ✓")

    orion_model.init_orion_params()
    print("Step 2: Fuse BatchNorm (init_orion_params) ✓")

    orion_backbone = orion_model.nets[patch_idx]

    # Initialize CKKS scheme
    print("\nStep 3: Initialize CKKS scheme...")
    orion.init_scheme("configs/cryptoface_net4.yml")
    print("         ✓ Scheme initialized")

    # Fit: Trace model
    print("\nStep 4: Fit (trace model)...")
    orion.fit(orion_backbone, patch)
    print("         ✓ Model traced")

    # Compile: Generate FHE parameters
    print("\nStep 5: Compile (generate FHE parameters)...")
    input_level = orion.compile(orion_backbone)
    print(f"         ✓ Model compiled (input_level={input_level})")

    print(f"\n{'='*80}")
    print(f"Both models ready. Now comparing cleartext outputs...")
    print(f"{'='*80}")

    # =========================================================================
    # Compare Layer-by-Layer in Cleartext Mode
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"Layer-by-layer cleartext comparison")
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
            return False

        # Layer 1
        cf_out = cf_backbone.layers[0].forward_fuse(cf_out)
        orion_out = orion_backbone.layer1(orion_out)
        match = compare_tensors("After layer1", cf_out, orion_out, verbose=True)
        if not match:
            print(f"\n⚠ First divergence at: layer1")
            return False

        # Layer 2 (has shortcut)
        cf_out = cf_backbone.layers[1].forward_fuse(cf_out)
        orion_out = orion_backbone.layer2(orion_out)
        match = compare_tensors("After layer2 (stride=2, shortcut)", cf_out, orion_out, verbose=True)
        if not match:
            print(f"\n⚠ First divergence at: layer2")
            return False

        # Layer 3
        cf_out = cf_backbone.layers[2].forward_fuse(cf_out)
        orion_out = orion_backbone.layer3(orion_out)
        match = compare_tensors("After layer3", cf_out, orion_out, verbose=True)
        if not match:
            print(f"\n⚠ First divergence at: layer3")
            return False

        # Layer 4 (has shortcut)
        cf_out = cf_backbone.layers[3].forward_fuse(cf_out)
        orion_out = orion_backbone.layer4(orion_out)
        match = compare_tensors("After layer4 (stride=2, shortcut)", cf_out, orion_out, verbose=True)
        if not match:
            print(f"\n⚠ First divergence at: layer4")
            return False

        # Layer 5
        cf_out = cf_backbone.layers[4].forward_fuse(cf_out)
        orion_out = orion_backbone.layer5(orion_out)
        match = compare_tensors("After layer5", cf_out, orion_out, verbose=True)
        if not match:
            print(f"\n⚠ First divergence at: layer5")
            return False

        # HerPNPool
        cf_out = cf_backbone.herpnpool.forward_fuse(cf_out)
        orion_out = orion_backbone.herpnpool(orion_out)
        match = compare_tensors("After herpnpool", cf_out, orion_out, verbose=True)
        if not match:
            print(f"\n⚠ First divergence at: herpnpool")
            return False

        # Flatten
        cf_out = cf_backbone.flatten(cf_out)
        orion_out = orion_backbone.flatten(orion_out)
        match = compare_tensors("After flatten", cf_out, orion_out, verbose=True)
        if not match:
            print(f"\n⚠ First divergence at: flatten")
            return False

        # Final BatchNorm
        cf_out = cf_backbone.bn(cf_out)
        orion_out = orion_backbone.bn(orion_out)
        match = compare_tensors("After final bn", cf_out, orion_out, verbose=True)
        if not match:
            print(f"\n⚠ First divergence at: final bn")
            return False

        print(f"\n{'='*80}")
        print(f"✓ ALL LAYERS MATCH AFTER FULL COMPILATION!")
        print(f"{'='*80}")
        print(f"\nConclusion: Compilation (fit + compile) does NOT change cleartext behavior.")
        print(f"            Orion model matches CryptoFace after all preprocessing steps.")
        return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare after full compilation")
    parser.add_argument("--patch", type=int, default=0, help="Patch index to compare")

    args = parser.parse_args()

    success = compare_after_full_compile(patch_idx=args.patch, verbose=True)
    sys.exit(0 if success else 1)
