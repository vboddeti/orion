"""
Compare w1_fhe encoding between layer1_herpn2 (gap=1) and layer2_herpn2 (gap=2)
to identify why hybrid packing causes errors.
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
    print("\n" + "="*80)
    print("Comparing w1_fhe encoding: gap=1 vs gap=2")
    print("="*80 + "\n")

    # Load image
    lfw_dir = Path("/research/hal-datastage/datasets/original/LFW/lfw-mtcnn-aligned")
    person_dirs = sorted([d for d in lfw_dir.iterdir() if d.is_dir()])
    first_person = person_dirs[0]
    images = sorted(list(first_person.glob("*.jpg")) + list(first_person.glob("*.png")))
    image_path = images[0]

    full_img = load_lfw_image(image_path, input_size=64)
    patch = full_img[:, :, :32, :32]

    # Load model
    print("Loading model...")
    fhe_model = CryptoFaceNet4()
    load_checkpoint_for_config(fhe_model, input_size=64, verbose=False)
    full_backbone = fhe_model.nets[0]
    full_backbone.eval()
    full_backbone.init_orion_params()

    # Create partial backbone (only layer1 and layer2 to speed up compilation)
    import orion.nn as on

    class PartialBackbone(on.Module):
        def __init__(self, backbone):
            super().__init__()
            self.conv = backbone.conv
            self.layer1 = backbone.layer1
            self.layer2 = backbone.layer2

        def forward(self, x):
            x = self.conv(x)
            x = self.layer1(x)
            x = self.layer2(x)
            return x

    backbone = PartialBackbone(full_backbone)
    backbone.eval()

    # Initialize FHE
    print("Initializing CKKS scheme...")
    orion.init_scheme("configs/pcnn-backbone.yml")

    # Fit and compile
    print("Fitting and compiling (only through layer2)...")
    orion.fit(backbone, patch)
    input_level = orion.compile(backbone)
    print(f"✓ Compiled (input_level={input_level})\n")

    # ==================================================================
    # Compare layer1_herpn2 and layer2_herpn2
    # ==================================================================
    print("="*80)
    print("Layer1 HerPN2 (gap=1)")
    print("="*80)

    herpn1 = backbone.layer1.herpn2
    print(f"Cleartext w1 shape: {herpn1.w1.shape}")
    print(f"Cleartext w1 range: [{herpn1.w1.min():.6f}, {herpn1.w1.max():.6f}]")
    print(f"Cleartext w1 values (first 5): {herpn1.w1.squeeze()[:5].tolist()}")

    if hasattr(herpn1, 'w1_fhe'):
        print(f"\nw1_fhe exists: True")
        print(f"w1_fhe type: {type(herpn1.w1_fhe)}")
        print(f"w1_fhe shape (on_shape): {herpn1.w1_fhe.on_shape}")
        print(f"w1_fhe shape (shape): {herpn1.w1_fhe.shape}")

        # Decode to see actual values
        w1_decoded = herpn1.w1_fhe.decode()
        print(f"w1_fhe decoded shape: {w1_decoded.shape}")
        print(f"w1_fhe decoded range: [{w1_decoded.min():.6f}, {w1_decoded.max():.6f}]")
        print(f"w1_fhe decoded values (first 5): {w1_decoded.flatten()[:5].tolist()}")

        # Note: w1_fhe is broadcast, so direct comparison needs reshaping
        print(f"\nNote: w1_fhe is broadcast from {herpn1.w1.shape} to {w1_decoded.shape}")
    else:
        print("\nw1_fhe not found!")

    print("\n" + "="*80)
    print("Layer2 HerPN2 (gap=2)")
    print("="*80)

    herpn2 = backbone.layer2.herpn2
    print(f"Cleartext w1 shape: {herpn2.w1.shape}")
    print(f"Cleartext w1 range: [{herpn2.w1.min():.6f}, {herpn2.w1.max():.6f}]")
    print(f"Cleartext w1 values (first 5): {herpn2.w1.squeeze()[:5].tolist()}")

    if hasattr(herpn2, 'w1_fhe'):
        print(f"\nw1_fhe exists: True")
        print(f"w1_fhe type: {type(herpn2.w1_fhe)}")
        print(f"w1_fhe shape (on_shape): {herpn2.w1_fhe.on_shape}")
        print(f"w1_fhe shape (shape): {herpn2.w1_fhe.shape}")

        # Decode to see actual values
        w1_decoded = herpn2.w1_fhe.decode()
        print(f"w1_fhe decoded shape: {w1_decoded.shape}")
        print(f"w1_fhe decoded range: [{w1_decoded.min():.6f}, {w1_decoded.max():.6f}]")
        print(f"w1_fhe decoded values (first 5): {w1_decoded.flatten()[:5].tolist()}")

        # Note: w1_fhe is broadcast, so we can't directly compare flattened tensors
        print(f"\nNote: w1_fhe is broadcast from {herpn2.w1.shape} to {w1_decoded.shape}")
    else:
        print("\nw1_fhe not found!")

    # ==================================================================
    # Check input/output shapes
    # ==================================================================
    print("\n" + "="*80)
    print("Shape Analysis")
    print("="*80)

    print("\nLayer1 HerPN2:")
    print(f"  Input clear shape: (1, 16, 32, 32)")
    print(f"  Input FHE shape: (1, 16, 32, 32)")
    print(f"  Gap: 1")
    print(f"  w1 shape: {herpn1.w1.shape}")
    print(f"  w1_fhe on_shape: {herpn1.w1_fhe.on_shape if hasattr(herpn1, 'w1_fhe') else 'N/A'}")

    print("\nLayer2 HerPN2:")
    print(f"  Input clear shape: (1, 32, 16, 16)")
    print(f"  Input FHE shape: (1, 8, 32, 32)")
    print(f"  Gap: 2")
    print(f"  w1 shape: {herpn2.w1.shape}")
    print(f"  w1_fhe on_shape: {herpn2.w1_fhe.on_shape if hasattr(herpn2, 'w1_fhe') else 'N/A'}")

    print("\n" + "="*80)
    print("HYPOTHESIS:")
    print("="*80)
    print("""
For gap=2, the input is in packed format (1, 8, 32, 32) representing cleartext (1, 32, 16, 16).
The weight w1 has shape (32, 1, 1) in cleartext.

When multiplying ciphertext * plaintext with gap=2:
  - The plaintext weight needs to be properly broadcast/packed to match the packed ciphertext
  - If w1_fhe is encoded assuming gap=1 (unpacked), it won't properly align with the packed ciphertext
  - This would cause incorrect multiplication results

Expected:
  - w1_fhe for gap=2 should have on_shape accounting for the packing
  - OR w1_fhe should be broadcast differently for packed vs unpacked tensors
    """)


if __name__ == "__main__":
    main()
