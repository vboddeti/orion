"""
Verify weight loading with real LFW face images.
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
    """
    Load and preprocess LFW image using CryptoFace preprocessing.

    CryptoFace preprocessing (from datasets.py):
    - Resize to input_size (if not 112)
    - Normalize with mean=0.5, std=0.5 (transforms [0,1] → [-1,1])
    """
    # Load image
    img = Image.open(image_path).convert('RGB')

    # CryptoFace preprocessing
    transform = transforms.Compose([
        transforms.Resize(input_size, antialias=True),
        transforms.ToTensor(),  # [0, 1]
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]
    ])

    img_tensor = transform(img)
    return img_tensor.unsqueeze(0)  # Add batch dimension


def verify_with_real_image(input_size=64, verbose=True):
    """Verify models with real LFW image."""

    if verbose:
        print(f"\n{'='*80}")
        print(f"VERIFICATION: Real LFW Image Test ({input_size}×{input_size})")
        print(f"{'='*80}")

    # Find a real LFW image
    lfw_dir = Path("/research/hal-datastage/datasets/original/LFW/lfw-mtcnn-aligned")

    # Get first image from first person
    person_dirs = sorted([d for d in lfw_dir.iterdir() if d.is_dir()])
    if not person_dirs:
        raise FileNotFoundError(f"No person directories found in {lfw_dir}")

    first_person = person_dirs[0]
    images = sorted(list(first_person.glob("*.jpg")) + list(first_person.glob("*.png")))

    if not images:
        raise FileNotFoundError(f"No images found in {first_person}")

    image_path = images[0]

    if verbose:
        print(f"\nUsing image: {image_path}")
        print(f"Person: {first_person.name}")

    # Load and preprocess image
    if verbose:
        print(f"\n{'='*80}")
        print(f"Step 1: Loading and preprocessing image...")
        print(f"{'='*80}")

    img = load_lfw_image(image_path, input_size=input_size)

    if verbose:
        print(f"Image shape: {img.shape}")
        print(f"Image range: [{img.min():.4f}, {img.max():.4f}]")
        print(f"Image mean: {img.mean():.4f}, std: {img.std():.4f}")

    # Determine model variant
    checkpoint_map = {
        64: ("backbone-64x64.ckpt", CryptoFaceNet4, 4),
        96: ("backbone-96x96.ckpt", None, 9),  # Not implemented yet
        128: ("backbone-128x128.ckpt", None, 16),  # Not implemented yet
    }

    if input_size not in checkpoint_map:
        raise ValueError(f"Invalid input_size: {input_size}")

    checkpoint_name, orion_model_fn, num_patches = checkpoint_map[input_size]
    checkpoint_path = f"face_recognition/checkpoints/{checkpoint_name}"

    # Step 2: Load CryptoFace model
    if verbose:
        print(f"\n{'='*80}")
        print(f"Step 2: Loading CryptoFace original model...")
        print(f"{'='*80}")

    cryptoface_model = CryptoFacePatchCNN(input_size=input_size, patch_size=32)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cryptoface_model.load_state_dict(ckpt['backbone'], strict=False)
    cryptoface_model.eval()
    cryptoface_model.fuse()

    if verbose:
        print(f"✓ Loaded and fused CryptoFace model: {num_patches} patches")

    # Step 3: Load Orion model
    if verbose:
        print(f"\n{'='*80}")
        print(f"Step 3: Loading Orion CryptoFacePCNN model...")
        print(f"{'='*80}")

    orion_model = orion_model_fn()
    load_checkpoint_for_config(orion_model, input_size=input_size, verbose=False)
    orion_model.init_orion_params()
    orion_model.eval()

    if verbose:
        print(f"✓ Loaded and fused Orion model: {num_patches} patches")

    # Step 4: Run CryptoFace model
    if verbose:
        print(f"\n{'='*80}")
        print(f"Step 4: Running CryptoFace model...")
        print(f"{'='*80}")

    with torch.no_grad():
        cryptoface_output = cryptoface_model.forward_fuse(img)

    if verbose:
        print(f"CryptoFace output shape: {cryptoface_output.shape}")
        print(f"CryptoFace output range: [{cryptoface_output.min():.6f}, {cryptoface_output.max():.6f}]")
        print(f"CryptoFace output mean: {cryptoface_output.mean():.6f}")
        print(f"CryptoFace output std:  {cryptoface_output.std():.6f}")
        print(f"CryptoFace has NaN: {torch.isnan(cryptoface_output).any()}")

    # Step 5: Run Orion model
    if verbose:
        print(f"\n{'='*80}")
        print(f"Step 5: Running Orion model...")
        print(f"{'='*80}")

    with torch.no_grad():
        orion_output = orion_model(img)

    if verbose:
        print(f"Orion output shape: {orion_output.shape}")
        print(f"Orion output range: [{orion_output.min():.6f}, {orion_output.max():.6f}]")
        print(f"Orion output mean: {orion_output.mean():.6f}")
        print(f"Orion output std:  {orion_output.std():.6f}")
        print(f"Orion has NaN: {torch.isnan(orion_output).any()}")

    # Step 6: Compare outputs
    if verbose:
        print(f"\n{'='*80}")
        print(f"Step 6: Comparing outputs...")
        print(f"{'='*80}")

    cryptoface_np = cryptoface_output.numpy()
    orion_np = orion_output.numpy()

    abs_diff = np.abs(cryptoface_np - orion_np)
    rel_diff = abs_diff / (np.abs(cryptoface_np) + 1e-8)

    max_abs_diff = abs_diff.max()
    mean_abs_diff = abs_diff.mean()
    max_rel_diff = rel_diff.max()
    mean_rel_diff = rel_diff.mean()

    # Cosine similarity
    cos_sim = np.dot(cryptoface_np[0], orion_np[0]) / (
        np.linalg.norm(cryptoface_np[0]) * np.linalg.norm(orion_np[0])
    )

    if verbose:
        print(f"\nAbsolute Differences:")
        print(f"  Max:  {max_abs_diff:.8f}")
        print(f"  Mean: {mean_abs_diff:.8f}")
        print(f"\nRelative Differences:")
        print(f"  Max:  {max_rel_diff:.8f}")
        print(f"  Mean: {mean_rel_diff:.8f}")
        print(f"\nCosine Similarity: {cos_sim:.10f}")

    # Determine if outputs match
    tolerance_abs = 1e-4
    tolerance_cosine = 0.9999

    abs_match = max_abs_diff < tolerance_abs
    cosine_match = cos_sim > tolerance_cosine

    if verbose:
        print(f"\n{'='*80}")
        print(f"VERIFICATION RESULT")
        print(f"{'='*80}")
        print(f"Absolute difference < {tolerance_abs}: {abs_match} {'✓' if abs_match else '✗'}")
        print(f"Cosine similarity > {tolerance_cosine}: {cosine_match} {'✓' if cosine_match else '✗'}")

        if abs_match and cosine_match:
            print(f"\n{'='*80}")
            print(f"✓✓✓ VERIFICATION PASSED! ✓✓✓")
            print(f"{'='*80}")
            print(f"Orion model produces identical outputs to CryptoFace model!")
        else:
            print(f"\n{'='*80}")
            print(f"⚠ VERIFICATION FAILED")
            print(f"{'='*80}")
            print(f"Outputs differ beyond tolerance.")

    return abs_match and cosine_match


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify with real LFW image")
    parser.add_argument("--input-size", type=int, default=64, choices=[64, 96, 128])

    args = parser.parse_args()

    passed = verify_with_real_image(input_size=args.input_size, verbose=True)
    sys.exit(0 if passed else 1)
