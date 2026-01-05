"""
Simple end-to-end comparison: Plaintext vs FHE inference for a single patch backbone.

This avoids intermediate decryptions and only compares the final output.
"""
import sys
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms

sys.path.append('/research/hal-vishnu/code/orion-fhe')

import orion
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


def main():
    patch_idx = 0

    print(f"\n{'='*80}")
    print(f"PLAINTEXT VS FHE COMPARISON (Patch {patch_idx} Backbone)")
    print(f"{'='*80}")

    # Check config
    config_path = Path("configs/cryptoface_net4.yml")
    if not config_path.exists():
        print(f"\n✗ Config not found: {config_path}")
        return

    # Load test image
    lfw_dir = Path("/research/hal-datastage/datasets/original/LFW/lfw-mtcnn-aligned")
    person_dirs = sorted([d for d in lfw_dir.iterdir() if d.is_dir()])
    first_person = person_dirs[0]
    images = sorted(list(first_person.glob("*.jpg")) + list(first_person.glob("*.png")))
    image_path = images[0]

    print(f"\nTest image: {image_path}")
    img = load_lfw_image(image_path, input_size=64)

    # Extract single patch
    patches = []
    for h in range(2):
        for w in range(2):
            patch = img[:, :, h*32:(h+1)*32, w*32:(w+1)*32]
            patches.append(patch)

    patch = patches[patch_idx]
    print(f"Patch shape: {patch.shape}, range: [{patch.min():.4f}, {patch.max():.4f}]")

    # Initialize FHE scheme
    print(f"\n{'='*80}")
    print(f"Initializing FHE Scheme")
    print(f"{'='*80}")
    orion.init_scheme(str(config_path))
    print(f"✓ Scheme initialized")

    # Load model with pre-trained weights
    print(f"\n{'='*80}")
    print(f"Loading Model")
    print(f"{'='*80}")
    model = CryptoFaceNet4()
    load_checkpoint_for_config(model, input_size=64, verbose=False)

    # Warmup
    print(f"\nWarmup: Collecting BatchNorm statistics...")
    model.eval()
    with torch.no_grad():
        for _ in range(20):
            _ = model(torch.randn(4, 3, 64, 64))
    print(f"✓ Statistics collected")

    # Fuse operations
    print(f"\nFusing HerPN operations...")
    model.init_orion_params()
    print(f"✓ HerPN fused")

    # Get backbone for this patch
    backbone = model.nets[patch_idx]

    # Run plaintext inference
    print(f"\n{'='*80}")
    print(f"Plaintext Inference")
    print(f"{'='*80}")

    backbone.eval()
    with torch.no_grad():
        plaintext_out = backbone(patch)

    print(f"✓ Plaintext output shape: {plaintext_out.shape}")
    print(f"  Range: [{plaintext_out.min():.6f}, {plaintext_out.max():.6f}]")

    # Fit and compile for FHE
    print(f"\n{'='*80}")
    print(f"Fitting and Compiling for FHE")
    print(f"{'='*80}")

    orion.fit(backbone, patch)
    print(f"✓ Model traced")

    input_level = orion.compile(backbone)
    print(f"✓ Model compiled, input_level = {input_level}")

    # Encode and encrypt
    print(f"\n{'='*80}")
    print(f"Encrypting Input")
    print(f"{'='*80}")

    vec_ptxt = orion.encode(patch, input_level)
    print(f"✓ Input encoded")

    vec_ctxt = orion.encrypt(vec_ptxt)
    print(f"✓ Input encrypted")

    # Run FHE inference
    print(f"\n{'='*80}")
    print(f"FHE Inference")
    print(f"{'='*80}")

    backbone.he()  # Switch to FHE mode
    out_ctxt = backbone(vec_ctxt)
    print(f"✓ FHE inference completed")

    # Decrypt and decode
    print(f"\n{'='*80}")
    print(f"Decrypting Output")
    print(f"{'='*80}")

    fhe_out = out_ctxt.decrypt().decode()
    print(f"✓ Output decrypted and decoded")
    print(f"  FHE output shape: {fhe_out.shape}")
    print(f"  Range: [{fhe_out.min():.6f}, {fhe_out.max():.6f}]")

    # Compare results
    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")

    diff = (plaintext_out - fhe_out).abs()
    rel_diff = diff / (plaintext_out.abs() + 1e-8)

    mae = diff.mean().item()
    max_abs_error = diff.max().item()
    mean_rel_error = rel_diff.mean().item()

    print(f"\nError metrics:")
    print(f"  MAE (Mean Absolute Error):    {mae:.6f}")
    print(f"  Max Absolute Error:           {max_abs_error:.6f}")
    print(f"  Mean Relative Error:          {mean_rel_error:.6f} ({mean_rel_error*100:.4f}%)")

    # Success criteria
    if mae < 1.0:
        print(f"\n{'='*80}")
        print(f"✓ FHE INFERENCE SUCCESSFUL (MAE < 1.0)")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print(f"✗ FHE INFERENCE FAILED (MAE >= 1.0)")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
