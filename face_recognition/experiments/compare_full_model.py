"""
Compare full model outputs (all patches + linear layers + final normalization).
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


def compare_tensors(name, cf_output, orion_output):
    """Compare two tensors and print statistics."""
    print(f"\n{name}:")
    print(f"  CryptoFace: shape={cf_output.shape}, range=[{cf_output.min():.6f}, {cf_output.max():.6f}]")
    print(f"  Orion:      shape={orion_output.shape}, range=[{orion_output.min():.6f}, {orion_output.max():.6f}]")

    diff = (cf_output - orion_output).abs()
    rel_diff = diff / (cf_output.abs() + 1e-8)

    print(f"  Difference: max_abs={diff.max():.8f}, mean_abs={diff.mean():.8f}")
    print(f"              max_rel={rel_diff.max():.8f}, mean_rel={rel_diff.mean():.8f}")

    # Tolerance: 1e-4 for max abs error (reasonable for float32 precision)
    if diff.max() < 1e-4:
        print(f"  ✓ Outputs MATCH (within numerical precision)")
        return True
    else:
        print(f"  ✗ Outputs DIFFER")
        return False


def main():
    print(f"\n{'='*80}")
    print(f"FULL MODEL COMPARISON")
    print(f"{'='*80}")

    # Load real image
    lfw_dir = Path("/research/hal-datastage/datasets/original/LFW/lfw-mtcnn-aligned")
    person_dirs = sorted([d for d in lfw_dir.iterdir() if d.is_dir()])
    first_person = person_dirs[0]
    images = sorted(list(first_person.glob("*.jpg")) + list(first_person.glob("*.png")))
    image_path = images[0]

    print(f"\nUsing image: {image_path}")
    img = load_lfw_image(image_path, input_size=64)
    print(f"Image shape: {img.shape}, range: [{img.min():.4f}, {img.max():.4f}]")

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

    # Load Orion model
    print(f"\n{'='*80}")
    print(f"Loading Orion model")
    print(f"{'='*80}")

    orion_model = CryptoFaceNet4()
    load_checkpoint_for_config(orion_model, input_size=64, verbose=False)
    orion_model.init_orion_params()
    orion_model.eval()

    # Forward pass
    print(f"\n{'='*80}")
    print(f"Forward Pass")
    print(f"{'='*80}")

    with torch.no_grad():
        cf_out = cryptoface_model.forward_fuse(img)
        orion_out = orion_model(img)

    # Compare outputs
    match = compare_tensors("Final embeddings", cf_out, orion_out)

    if match:
        print(f"\n{'='*80}")
        print(f"✓ FULL MODEL OUTPUTS MATCH!")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print(f"✗ FULL MODEL OUTPUTS DIFFER")
        print(f"{'='*80}")

        # Additional debugging: compare backbone outputs
        print(f"\nDebug: Comparing individual patch backbone outputs...")

        # Extract patches
        patches = []
        for h in range(2):
            for w in range(2):
                patch = img[:, :, h*32:(h+1)*32, w*32:(w+1)*32]
                patches.append(patch)

        for i in range(4):
            cf_backbone_out = cryptoface_model.nets[i].forward_fuse(patches[i])
            orion_backbone_out = orion_model.nets[i](patches[i])
            compare_tensors(f"  Patch {i} backbone output", cf_backbone_out, orion_backbone_out)


if __name__ == "__main__":
    main()
