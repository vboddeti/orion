"""
Debug any layer by decrypting input/output at each operation.

This script identifies where FHE diverges from cleartext within a specified layer.

Usage:
    python debug_layer3_detailed.py [layer_number]

Example:
    python debug_layer3_detailed.py 2  # Debug layer2
    python debug_layer3_detailed.py 3  # Debug layer3
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
    """
    Unpack a hybrid-packed FHE tensor back to original cleartext shape.

    Args:
        packed_tensor: Tensor in packed format (on_shape)
        gap: The packing gap used
        original_shape: The target cleartext shape

    Returns:
        Tensor in original cleartext shape
    """
    if gap == 1:
        # No packing, already in cleartext format
        return packed_tensor

    # Reverse pixel_shuffle: pixel_unshuffle
    unpacked = F.pixel_unshuffle(packed_tensor, gap)

    # Trim to original shape (removes padding)
    N, C_orig, H_orig, W_orig = original_shape
    unpacked = unpacked[:, :C_orig, :H_orig, :W_orig]

    return unpacked


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


def calculate_gap(cleartext_shape, packed_shape):
    """Calculate packing gap from cleartext and packed shapes."""
    clear_H = cleartext_shape[2]
    packed_H = packed_shape[2]
    return packed_H // clear_H if clear_H > 0 else 1


def compare_tensors(name, cleartext, fhe_decrypted_packed, gap, indent=0):
    """
    Compare cleartext vs decrypted FHE tensors.

    Args:
        name: Name of the operation
        cleartext: Cleartext tensor
        fhe_decrypted_packed: Decrypted FHE tensor (still in packed format)
        gap: Packing gap to unpack the FHE tensor
        indent: Indentation level for printing
    """
    prefix = "  " * indent

    # Unpack FHE tensor to match cleartext shape
    fhe_unpacked = unpack_fhe_tensor(fhe_decrypted_packed, gap, cleartext.shape)

    diff = (cleartext - fhe_unpacked).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    clear_max = cleartext.abs().max().item()
    rel_error = (max_diff / clear_max * 100) if clear_max > 0 else 0

    print(f"{prefix}{name}:")
    print(f"{prefix}  Cleartext:     shape={tuple(cleartext.shape)}, range=[{cleartext.min():.6f}, {cleartext.max():.6f}]")
    print(f"{prefix}  FHE (packed):  shape={tuple(fhe_decrypted_packed.shape)}, gap={gap}")
    print(f"{prefix}  FHE (unpack):  shape={tuple(fhe_unpacked.shape)}, range=[{fhe_unpacked.min():.6f}, {fhe_unpacked.max():.6f}]")
    print(f"{prefix}  Difference:    max={max_diff:.6f}, mean={mean_diff:.6f}")
    print(f"{prefix}  Relative err:  {rel_error:.2f}%")

    if rel_error > 5.0:
        print(f"{prefix}  ⚠️  WARNING: Large divergence!")
    elif rel_error < 0.1:
        print(f"{prefix}  ✓ Match")

    return rel_error


def debug_layer(layer_num):
    """
    Debug specified layer operation by operation.

    Args:
        layer_num: Layer number to debug (1, 2, 3, 4, etc.)
    """

    print(f"\n{'='*80}")
    print(f"Layer{layer_num} Detailed Debugging")
    print(f"{'='*80}\n")

    # Load real image
    lfw_dir = Path("/research/hal-datastage/datasets/original/LFW/lfw-mtcnn-aligned")
    person_dirs = sorted([d for d in lfw_dir.iterdir() if d.is_dir()])
    first_person = person_dirs[0]
    images = sorted(list(first_person.glob("*.jpg")) + list(first_person.glob("*.png")))
    image_path = images[0]

    full_img = load_lfw_image(image_path, input_size=64)
    patch = full_img[:, :, :32, :32]

    print(f"Using image: {image_path}")
    print(f"Patch: shape={patch.shape}, range=[{patch.min():.4f}, {patch.max():.4f}]\n")

    # Load model for cleartext inference
    print("Loading cleartext model...")
    from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4
    from face_recognition.models.weight_loader import load_checkpoint_for_config

    clear_model = CryptoFaceNet4()
    load_checkpoint_for_config(clear_model, input_size=64, verbose=False)
    clear_backbone = clear_model.nets[0]
    clear_backbone.eval()
    clear_backbone.init_orion_params()

    # Load model for FHE inference
    print("Loading FHE model...")
    fhe_model = CryptoFaceNet4()
    load_checkpoint_for_config(fhe_model, input_size=64, verbose=False)
    full_backbone = fhe_model.nets[0]
    full_backbone.eval()

    # Fuse BatchNorm
    print("Fusing BatchNorm into HerPN...")
    full_backbone.init_orion_params()

    # Create incremental model (only up to target layer to avoid unnecessary compilation)
    import orion.nn as on

    class PartialBackbone(on.Module):
        def __init__(self, backbone, max_layer):
            super().__init__()
            self.conv = backbone.conv
            # Dynamically add layers up to max_layer
            for i in range(1, max_layer + 1):
                setattr(self, f'layer{i}', getattr(backbone, f'layer{i}'))
            self.max_layer = max_layer

        def forward(self, x):
            x = self.conv(x)
            # Forward through all layers up to max_layer
            for i in range(1, self.max_layer + 1):
                layer = getattr(self, f'layer{i}')
                x = layer(x)
            return x

    backbone = PartialBackbone(full_backbone, max_layer=layer_num)
    backbone.eval()

    # Initialize FHE
    print("Initializing CKKS scheme...")
    orion.init_scheme("configs/pcnn-backbone.yml")

    # Fit and compile
    print(f"Fitting and compiling (up to layer{layer_num})...")
    orion.fit(backbone, patch)
    input_level = orion.compile(backbone)
    print(f"✓ Compiled (input_level={input_level})\n")

    # ==================================================================
    # Run all layers before target layer to get its input
    # ==================================================================
    print(f"{'='*80}")
    print(f"Getting input to layer{layer_num} (running all previous layers)")
    print(f"{'='*80}\n")

    # Cleartext - run through all layers before target
    with torch.no_grad():
        clear_x = patch
        clear_x = clear_backbone.conv(clear_x)

        # Run through layers 1 to (layer_num - 1)
        for i in range(1, layer_num):
            layer = getattr(clear_backbone, f'layer{i}')
            clear_x = layer(clear_x)

        layer_input_clear = clear_x.clone()

    print(f"Layer{layer_num} input (cleartext): shape={tuple(layer_input_clear.shape)}, range=[{layer_input_clear.min():.4f}, {layer_input_clear.max():.4f}]")

    # FHE - run through all layers before target
    inp_ptxt = orion.encode(patch, input_level)
    inp_ctxt = orion.encrypt(inp_ptxt)
    backbone.he()

    with torch.no_grad():
        fhe_x = inp_ctxt
        fhe_x = backbone.conv(fhe_x)

        # Run through layers 1 to (layer_num - 1)
        for i in range(1, layer_num):
            layer = getattr(backbone, f'layer{i}')
            fhe_x = layer(fhe_x)

        layer_input_fhe = fhe_x
        layer_input_fhe_decrypt = layer_input_fhe.decrypt().decode()

    # Calculate gap from shapes
    clear_H = layer_input_clear.shape[2]
    packed_H = layer_input_fhe_decrypt.shape[2]
    layer_input_gap = packed_H // clear_H

    print(f"Layer{layer_num} input (FHE):       shape={tuple(layer_input_fhe_decrypt.shape)}, range=[{layer_input_fhe_decrypt.min():.4f}, {layer_input_fhe_decrypt.max():.4f}]")
    print(f"Layer{layer_num} input gap:         {layer_input_gap}\n")

    compare_tensors(f"Layer{layer_num} INPUT", layer_input_clear, layer_input_fhe_decrypt, layer_input_gap)

    # ==================================================================
    # Debug target layer operations step by step
    # ==================================================================
    print(f"\n{'='*80}")
    print(f"Layer{layer_num} Internal Operations")
    print(f"{'='*80}\n")

    # Get target layer references
    target_layer_clear = getattr(clear_backbone, f'layer{layer_num}')  # Use cleartext backbone
    target_layer_fhe = getattr(backbone, f'layer{layer_num}')  # Use compiled backbone in FHE mode

    # -------------------- HerPN1 --------------------
    print(f"Operation 1: HerPN1")
    print(f"-" * 80)

    # Cleartext
    with torch.no_grad():
        clear_herpn1_out = target_layer_clear.herpn1(layer_input_clear)
    print(f"  Cleartext output: range=[{clear_herpn1_out.min():.6f}, {clear_herpn1_out.max():.6f}]")

    # FHE
    fhe_herpn1_out = target_layer_fhe.herpn1(layer_input_fhe)
    fhe_herpn1_decrypt = fhe_herpn1_out.decrypt().decode()
    print(f"  FHE output:       range=[{fhe_herpn1_decrypt.min():.6f}, {fhe_herpn1_decrypt.max():.6f}]")
    print(f"  FHE level:        {fhe_herpn1_out.level()}")
    print(f"  FHE scale:        {fhe_herpn1_out.scale():.2e}")

    gap1 = calculate_gap(clear_herpn1_out.shape, fhe_herpn1_decrypt.shape)
    herpn1_error = compare_tensors("  HerPN1 output", clear_herpn1_out, fhe_herpn1_decrypt, gap1, indent=1)

    # -------------------- Conv1 --------------------
    print(f"\nOperation 2: Conv1")
    print(f"-" * 80)

    # Cleartext
    clear_conv1_out = target_layer_clear.conv1(clear_herpn1_out)
    print(f"  Cleartext output: range=[{clear_conv1_out.min():.6f}, {clear_conv1_out.max():.6f}]")

    # FHE
    fhe_conv1_out = target_layer_fhe.conv1(fhe_herpn1_out)
    fhe_conv1_decrypt = fhe_conv1_out.decrypt().decode()
    print(f"  FHE output:       range=[{fhe_conv1_decrypt.min():.6f}, {fhe_conv1_decrypt.max():.6f}]")
    print(f"  FHE level:        {fhe_conv1_out.level()}")
    print(f"  FHE scale:        {fhe_conv1_out.scale():.2e}")

    gap2 = calculate_gap(clear_conv1_out.shape, fhe_conv1_decrypt.shape)
    conv1_error = compare_tensors("  Conv1 output", clear_conv1_out, fhe_conv1_decrypt, gap2, indent=1)

    # -------------------- HerPN2 --------------------
    print(f"\nOperation 3: HerPN2")
    print(f"-" * 80)

    # Cleartext
    clear_herpn2_out = target_layer_clear.herpn2(clear_conv1_out)
    print(f"  Cleartext output: range=[{clear_herpn2_out.min():.6f}, {clear_herpn2_out.max():.6f}]")

    # FHE
    fhe_herpn2_out = target_layer_fhe.herpn2(fhe_conv1_out)
    fhe_herpn2_decrypt = fhe_herpn2_out.decrypt().decode()
    print(f"  FHE output:       range=[{fhe_herpn2_decrypt.min():.6f}, {fhe_herpn2_decrypt.max():.6f}]")
    print(f"  FHE level:        {fhe_herpn2_out.level()}")
    print(f"  FHE scale:        {fhe_herpn2_out.scale():.2e}")

    gap3 = calculate_gap(clear_herpn2_out.shape, fhe_herpn2_decrypt.shape)
    herpn2_error = compare_tensors("  HerPN2 output", clear_herpn2_out, fhe_herpn2_decrypt, gap3, indent=1)

    # -------------------- Conv2 --------------------
    print(f"\nOperation 4: Conv2")
    print(f"-" * 80)

    # Cleartext
    clear_conv2_out = target_layer_clear.conv2(clear_herpn2_out)
    print(f"  Cleartext output: range=[{clear_conv2_out.min():.6f}, {clear_conv2_out.max():.6f}]")

    # FHE
    fhe_conv2_out = target_layer_fhe.conv2(fhe_herpn2_out)
    fhe_conv2_decrypt = fhe_conv2_out.decrypt().decode()
    print(f"  FHE output:       range=[{fhe_conv2_decrypt.min():.6f}, {fhe_conv2_decrypt.max():.6f}]")
    print(f"  FHE level:        {fhe_conv2_out.level()}")
    print(f"  FHE scale:        {fhe_conv2_out.scale():.2e}")

    gap4 = calculate_gap(clear_conv2_out.shape, fhe_conv2_decrypt.shape)
    conv2_error = compare_tensors("  Conv2 output", clear_conv2_out, fhe_conv2_decrypt, gap4, indent=1)

    # -------------------- Shortcut Path --------------------
    print(f"\nOperation 5: Shortcut Path")
    print(f"-" * 80)

    # Cleartext shortcut (scaled identity through herpn1)
    with torch.no_grad():
        clear_shortcut = target_layer_clear.herpn1(layer_input_clear)  # Get herpn1 output
        clear_shortcut = clear_shortcut * target_layer_clear.herpn1.scale_factor  # Apply scale
    print(f"  Cleartext shortcut: range=[{clear_shortcut.min():.6f}, {clear_shortcut.max():.6f}]")

    # FHE shortcut
    fhe_shortcut = target_layer_fhe.herpn1(layer_input_fhe)  # This is same as fhe_herpn1_out
    fhe_shortcut = target_layer_fhe.shortcut_scale(fhe_shortcut)  # Apply scale module
    fhe_shortcut_decrypt = fhe_shortcut.decrypt().decode()
    print(f"  FHE shortcut:       range=[{fhe_shortcut_decrypt.min():.6f}, {fhe_shortcut_decrypt.max():.6f}]")
    print(f"  FHE level:          {fhe_shortcut.level()}")
    print(f"  FHE scale:          {fhe_shortcut.scale():.2e}")

    gap5 = calculate_gap(clear_shortcut.shape, fhe_shortcut_decrypt.shape)
    shortcut_error = compare_tensors("  Shortcut", clear_shortcut, fhe_shortcut_decrypt, gap5, indent=1)

    # -------------------- Addition --------------------
    print(f"\nOperation 6: Addition (main + shortcut)")
    print(f"-" * 80)

    # Cleartext
    clear_final = clear_conv2_out + clear_shortcut
    print(f"  Cleartext final: range=[{clear_final.min():.6f}, {clear_final.max():.6f}]")

    # FHE
    fhe_final = fhe_conv2_out + fhe_shortcut
    fhe_final_decrypt = fhe_final.decrypt().decode()
    print(f"  FHE final:       range=[{fhe_final_decrypt.min():.6f}, {fhe_final_decrypt.max():.6f}]")
    print(f"  FHE level:       {fhe_final.level()}")
    print(f"  FHE scale:       {fhe_final.scale():.2e}")

    gap6 = calculate_gap(clear_final.shape, fhe_final_decrypt.shape)
    final_error = compare_tensors("  Final output", clear_final, fhe_final_decrypt, gap6, indent=1)

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'='*80}")
    print(f"Error Summary for Layer{layer_num}")
    print(f"{'='*80}\n")

    # Get the input error by comparing layer input
    input_error = compare_tensors(f"Layer{layer_num} input (recalc)", layer_input_clear, layer_input_fhe_decrypt, layer_input_gap)

    print(f"\nLayer{layer_num} input:     {input_error:.2f}% error (propagated from previous layers)")
    print(f"HerPN1 output:    {herpn1_error:.2f}% error")
    print(f"Conv1 output:     {conv1_error:.2f}% error")
    print(f"HerPN2 output:    {herpn2_error:.2f}% error")
    print(f"Conv2 output:     {conv2_error:.2f}% error")
    print(f"Shortcut:         {shortcut_error:.2f}% error")
    print(f"Final output:     {final_error:.2f}% error")

    # Identify the problematic operation
    errors = {
        'HerPN1': herpn1_error,
        'Conv1': conv1_error,
        'HerPN2': herpn2_error,
        'Conv2': conv2_error,
        'Shortcut': shortcut_error,
        'Addition': final_error
    }

    max_error_op = max(errors, key=errors.get)
    print(f"\n⚠️  Largest error source: {max_error_op} ({errors[max_error_op]:.2f}%)")

    # Calculate error deltas to identify where error increases most
    error_deltas = {
        'Input→HerPN1': herpn1_error - input_error,
        'HerPN1→Conv1': conv1_error - herpn1_error,
        'Conv1→HerPN2': herpn2_error - conv1_error,
        'HerPN2→Conv2': conv2_error - herpn2_error,
        'Conv2→Addition': final_error - conv2_error
    }

    print(f"\nError increases:")
    for op, delta in error_deltas.items():
        status = "📈" if delta > 5 else "→"
        print(f"  {status} {op:20s}: {delta:+6.2f}%")

    if herpn1_error > 10:
        print(f"\n🔍 Issue identified: HerPN1 operation has {herpn1_error:.1f}% error")
        print(f"   This is likely a bootstrap-related issue or scale mismatch.")
    elif conv1_error > 10:
        print(f"\n🔍 Issue identified: Conv1 operation has {conv1_error:.1f}% error")
        print(f"   Check convolution weights or matrix diagonal packing.")
    elif herpn2_error > 10:
        print(f"\n🔍 Issue identified: HerPN2 operation has {herpn2_error:.1f}% error")
        print(f"   Check HerPN weight encoding or quadratic computation.")


if __name__ == "__main__":
    import sys

    # Parse command-line argument for layer number (default: 3)
    if len(sys.argv) > 1:
        layer_num = int(sys.argv[1])
    else:
        layer_num = 3
        print(f"No layer number specified, defaulting to layer{layer_num}")
        print(f"Usage: python {sys.argv[0]} [layer_number]\n")

    debug_layer(layer_num)
