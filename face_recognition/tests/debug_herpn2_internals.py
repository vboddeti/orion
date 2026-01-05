"""
Debug HerPN2 operation in layer2 by decrypting all intermediate computations.

This script identifies where the FHE divergence occurs within the HerPN operation:
- Input x
- Quadratic term: x²
- Weighted quadratic: w2·x²
- Linear term: w1·x
- Constant term: w0
- Final output: w2·x² + w1·x + w0
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms

sys.path.append('/research/hal-vishnu/code/orion-fhe')

import orion
from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4
from face_recognition.models.weight_loader import load_checkpoint_for_config


def unpack_fhe_tensor(packed_tensor, gap, original_shape):
    """Unpack a hybrid-packed FHE tensor back to original cleartext shape."""
    if gap == 1:
        return packed_tensor

    unpacked = F.pixel_unshuffle(packed_tensor, gap)
    N, C_orig, H_orig, W_orig = original_shape
    unpacked = unpacked[:, :C_orig, :H_orig, :W_orig]
    return unpacked


def calculate_gap(cleartext_shape, packed_shape):
    """Calculate packing gap from cleartext and packed shapes."""
    clear_H = cleartext_shape[2]
    packed_H = packed_shape[2]
    return packed_H // clear_H if clear_H > 0 else 1


def compare_tensors(name, cleartext, fhe_packed, gap, indent=0):
    """Compare cleartext vs FHE tensors after unpacking."""
    prefix = "  " * indent

    fhe_unpacked = unpack_fhe_tensor(fhe_packed, gap, cleartext.shape)

    diff = (cleartext - fhe_unpacked).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    clear_max = cleartext.abs().max().item()
    rel_error = (max_diff / clear_max * 100) if clear_max > 0 else 0

    print(f"{prefix}{name}:")
    print(f"{prefix}  Cleartext:     range=[{cleartext.min():.6f}, {cleartext.max():.6f}]")
    print(f"{prefix}  FHE (unpack):  range=[{fhe_unpacked.min():.6f}, {fhe_unpacked.max():.6f}]")
    print(f"{prefix}  Difference:    max={max_diff:.6f}, mean={mean_diff:.6f}")
    print(f"{prefix}  Relative err:  {rel_error:.2f}%")

    if rel_error > 5.0:
        print(f"{prefix}  ⚠️  WARNING: Large divergence!")
    elif rel_error < 0.1:
        print(f"{prefix}  ✓ Match")

    return rel_error


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


def debug_herpn2_internals():
    """Debug layer2 HerPN2 operation step by step."""

    print(f"\n{'='*80}")
    print(f"Layer2 HerPN2 Internal Debug")
    print(f"{'='*80}\n")

    # Load image
    lfw_dir = Path("/research/hal-datastage/datasets/original/LFW/lfw-mtcnn-aligned")
    person_dirs = sorted([d for d in lfw_dir.iterdir() if d.is_dir()])
    first_person = person_dirs[0]
    images = sorted(list(first_person.glob("*.jpg")) + list(first_person.glob("*.png")))
    image_path = images[0]

    full_img = load_lfw_image(image_path, input_size=64)
    patch = full_img[:, :, :32, :32]

    print(f"Using image: {image_path}")
    print(f"Patch: shape={patch.shape}, range=[{patch.min():.4f}, {patch.max():.4f}]\n")

    # Load cleartext model
    print("Loading cleartext model...")
    clear_model = CryptoFaceNet4()
    load_checkpoint_for_config(clear_model, input_size=64, verbose=False)
    clear_backbone = clear_model.nets[0]
    clear_backbone.eval()
    clear_backbone.init_orion_params()

    # Load FHE model
    print("Loading FHE model...")
    fhe_model = CryptoFaceNet4()
    load_checkpoint_for_config(fhe_model, input_size=64, verbose=False)
    fhe_backbone = fhe_model.nets[0]
    fhe_backbone.eval()
    fhe_backbone.init_orion_params()

    # Create partial backbone - include all of layer2 so herpn2 gets compiled
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
            x = self.layer2(x)  # Run full layer2 to ensure herpn2 gets compiled
            return x

    backbone = PartialBackbone(fhe_backbone)
    backbone.eval()

    # Initialize FHE
    print("Initializing CKKS scheme...")
    orion.init_scheme("configs/pcnn-backbone.yml")

    # Fit and compile
    print("Fitting and compiling...")
    orion.fit(backbone, patch)
    input_level = orion.compile(backbone)
    print(f"✓ Compiled (input_level={input_level})\n")

    # ==================================================================
    # Get input to layer2_herpn2 (output of layer2_conv1)
    # ==================================================================
    print(f"{'='*80}")
    print(f"Getting input to layer2.herpn2")
    print(f"{'='*80}\n")

    # Cleartext path
    with torch.no_grad():
        clear_x = patch
        clear_x = clear_backbone.conv(clear_x)
        clear_x = clear_backbone.layer1(clear_x)
        clear_x = clear_backbone.layer2.herpn1(clear_x)
        clear_x = clear_backbone.layer2.conv1(clear_x)
        herpn2_input_clear = clear_x.clone()

    print(f"Cleartext input: shape={tuple(herpn2_input_clear.shape)}, range=[{herpn2_input_clear.min():.4f}, {herpn2_input_clear.max():.4f}]")

    # FHE path
    inp_ptxt = orion.encode(patch, input_level)
    inp_ctxt = orion.encrypt(inp_ptxt)
    backbone.he()

    with torch.no_grad():
        fhe_x = inp_ctxt
        fhe_x = backbone.conv(fhe_x)
        fhe_x = backbone.layer1(fhe_x)
        fhe_x = backbone.layer2.herpn1(fhe_x)
        fhe_x = backbone.layer2.conv1(fhe_x)
        herpn2_input_fhe = fhe_x
        herpn2_input_fhe_decrypt = herpn2_input_fhe.decrypt().decode()

    gap = calculate_gap(herpn2_input_clear.shape, herpn2_input_fhe_decrypt.shape)
    print(f"FHE input:       shape={tuple(herpn2_input_fhe_decrypt.shape)}, range=[{herpn2_input_fhe_decrypt.min():.4f}, {herpn2_input_fhe_decrypt.max():.4f}]")
    print(f"Gap:             {gap}\n")

    input_error = compare_tensors("HerPN2 INPUT", herpn2_input_clear, herpn2_input_fhe_decrypt, gap)

    # ==================================================================
    # Debug HerPN2 internal operations
    # ==================================================================
    print(f"\n{'='*80}")
    print(f"HerPN2 Internal Operations")
    print(f"{'='*80}\n")

    # Get HerPN2 layer
    herpn2_clear = clear_backbone.layer2.herpn2
    herpn2_fhe = backbone.layer2.herpn2

    # Print HerPN2 parameters
    print(f"HerPN2 parameters:")
    print(f"  w1_fhe exists: {hasattr(herpn2_fhe, 'w1_fhe')}")
    if hasattr(herpn2_fhe, 'w1_fhe'):
        print(f"  w1_fhe type: {type(herpn2_fhe.w1_fhe)}")
    print(f"  w2 shape: {herpn2_clear.w2.shape if hasattr(herpn2_clear, 'w2') else 'N/A'}")
    print(f"  w1 shape: {herpn2_clear.w1.shape if hasattr(herpn2_clear, 'w1') else 'N/A'}")
    print(f"  w0 shape: {herpn2_clear.w0.shape if hasattr(herpn2_clear, 'w0') else 'N/A'}")
    print(f"  Input level: {herpn2_input_fhe.level()}")
    print(f"  Input scale: {herpn2_input_fhe.scale():.2e}\n")

    # ==================================================================
    # Step 1: Compute x² (quadratic term)
    # ==================================================================
    print(f"Step 1: Compute x² (quadratic term)")
    print(f"-" * 80)

    # Cleartext x²
    x2_clear = herpn2_input_clear * herpn2_input_clear
    print(f"Cleartext x²: range=[{x2_clear.min():.6f}, {x2_clear.max():.6f}]")

    # FHE x²
    x2_fhe = herpn2_input_fhe * herpn2_input_fhe
    print(f"FHE x² level: {x2_fhe.level()}, scale: {x2_fhe.scale():.2e}")
    x2_fhe_decrypt = x2_fhe.decrypt().decode()
    print(f"FHE x²: range=[{x2_fhe_decrypt.min():.6f}, {x2_fhe_decrypt.max():.6f}]")

    x2_error = compare_tensors("x² term", x2_clear, x2_fhe_decrypt, gap, indent=1)

    # ==================================================================
    # Step 2: Linear term w1·x
    # ==================================================================
    print(f"\nStep 2: Compute w1·x (linear term)")
    print(f"-" * 80)

    # Cleartext w1·x
    w1_clear = herpn2_clear.w1
    w1x_clear = herpn2_input_clear * w1_clear
    print(f"Cleartext w1: range=[{w1_clear.min():.6f}, {w1_clear.max():.6f}]")
    print(f"Cleartext w1·x: range=[{w1x_clear.min():.6f}, {w1x_clear.max():.6f}]")

    # FHE w1·x
    if hasattr(herpn2_fhe, 'w1_fhe'):
        print(f"FHE w1_fhe type: {type(herpn2_fhe.w1_fhe)}")
        w1x_fhe = herpn2_input_fhe * herpn2_fhe.w1_fhe
        print(f"FHE w1·x level: {w1x_fhe.level()}, scale: {w1x_fhe.scale():.2e}")
        w1x_fhe_decrypt = w1x_fhe.decrypt().decode()
        print(f"FHE w1·x: range=[{w1x_fhe_decrypt.min():.6f}, {w1x_fhe_decrypt.max():.6f}]")

        w1x_error = compare_tensors("w1·x term", w1x_clear, w1x_fhe_decrypt, gap, indent=1)
    else:
        print(f"  ⚠️  w1_fhe not found, HerPN2 not compiled properly")

    # ==================================================================
    # Step 3: Constant term w0
    # ==================================================================
    print(f"\nStep 3: Constant term w0")
    print(f"-" * 80)

    w0_clear = herpn2_clear.w0
    print(f"Cleartext w0: range=[{w0_clear.min():.6f}, {w0_clear.max():.6f}]")

    # ==================================================================
    # Step 4: Combine all terms (w1·x + w0)
    # ==================================================================
    print(f"\nStep 4: Combine w1·x + w0")
    print(f"-" * 80)

    # Cleartext
    if 'w1x_clear' in locals():
        linear_clear = w1x_clear + w0_clear
        print(f"Cleartext (w1·x + w0): range=[{linear_clear.min():.6f}, {linear_clear.max():.6f}]")

        # FHE
        if hasattr(herpn2_fhe, 'w1_fhe') and 'w1x_fhe' in locals():
            linear_fhe = w1x_fhe + w0_clear  # w0 is plaintext
            print(f"FHE (w1·x + w0) level: {linear_fhe.level()}, scale: {linear_fhe.scale():.2e}")
            linear_fhe_decrypt = linear_fhe.decrypt().decode()
            print(f"FHE (w1·x + w0): range=[{linear_fhe_decrypt.min():.6f}, {linear_fhe_decrypt.max():.6f}]")

            linear_error = compare_tensors("w1·x + w0", linear_clear, linear_fhe_decrypt, gap, indent=1)

    # ==================================================================
    # Step 5: Full forward pass through HerPN2
    # ==================================================================
    print(f"\nStep 5: Full HerPN2 forward pass")
    print(f"-" * 80)

    # Cleartext
    with torch.no_grad():
        herpn2_output_clear = herpn2_clear(herpn2_input_clear)
    print(f"Cleartext output: range=[{herpn2_output_clear.min():.6f}, {herpn2_output_clear.max():.6f}]")

    # FHE
    herpn2_output_fhe = herpn2_fhe(herpn2_input_fhe)
    print(f"FHE output level: {herpn2_output_fhe.level()}, scale: {herpn2_output_fhe.scale():.2e}")
    herpn2_output_fhe_decrypt = herpn2_output_fhe.decrypt().decode()
    print(f"FHE output: range=[{herpn2_output_fhe_decrypt.min():.6f}, {herpn2_output_fhe_decrypt.max():.6f}]")

    output_error = compare_tensors("HerPN2 OUTPUT", herpn2_output_clear, herpn2_output_fhe_decrypt, gap, indent=1)

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'='*80}")
    print(f"Error Summary")
    print(f"{'='*80}\n")

    print(f"Input (x):         {input_error:.2f}% error")
    print(f"Quadratic (x²):    {x2_error:.2f}% error")
    if 'w1x_error' in locals():
        print(f"Linear (w1·x):     {w1x_error:.2f}% error")
    if 'linear_error' in locals():
        print(f"Combined (w1·x+w0): {linear_error:.2f}% error")
    print(f"Final output:      {output_error:.2f}% error")

    print(f"\n{'='*80}")

    # Identify where error jumps
    errors = [
        ("Input", input_error),
        ("x²", x2_error),
    ]
    if 'w1x_error' in locals():
        errors.append(("w1·x", w1x_error))
    if 'linear_error' in locals():
        errors.append(("w1·x+w0", linear_error))
    errors.append(("Output", output_error))

    print(f"Error progression:")
    for i in range(1, len(errors)):
        prev_name, prev_err = errors[i-1]
        curr_name, curr_err = errors[i]
        delta = curr_err - prev_err
        status = "📈" if delta > 5 else "→"
        print(f"  {status} {prev_name:15s} -> {curr_name:15s}: {prev_err:6.2f}% -> {curr_err:6.2f}% (Δ {delta:+6.2f}%)")

    print(f"{'='*80}\n")


if __name__ == "__main__":
    debug_herpn2_internals()
