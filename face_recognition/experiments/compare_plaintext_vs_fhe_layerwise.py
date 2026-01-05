"""
Layer-by-layer comparison between plaintext and FHE inference for a single patch.

This script helps debug FHE inference by comparing intermediate outputs
at each layer to identify where errors accumulate.

Modified to test BackboneWithLinear architecture where the linear layer
is fused into the backbone, allowing BN→Linear fusion during Orion compilation.
"""
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

sys.path.append('/research/hal-vishnu/code/orion-fhe')

import orion
import orion.nn as on
from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4
from face_recognition.models.weight_loader import load_checkpoint_for_config
from face_recognition.utils.fhe_utils import unpack_fhe_tensor, calculate_gap


class BackboneWithLinear(on.Module):
    """
    Backbone extended with linear layer, skipping the global BN.

    Architecture: conv → layers → herpnpool → flatten → linear

    Note: The global BN is already fused into the linear weights during
    checkpoint loading (see weight_loader.py), so we skip it in FHE inference.
    """

    def __init__(self, original_backbone, linear_layer, stop_at='linear'):
        super().__init__()
        self.stop_at = stop_at

        # Copy all original backbone layers
        self.conv = original_backbone.conv
        self.layer1 = original_backbone.layer1
        self.layer2 = original_backbone.layer2
        self.layer3 = original_backbone.layer3
        self.layer4 = original_backbone.layer4
        self.layer5 = original_backbone.layer5
        self.herpnpool = original_backbone.herpnpool
        self.flatten = original_backbone.flatten
        # SKIP: self.bn - global BN already fused into linear weights!

        # Add the linear layer (already has global BN fused)
        self.linear = linear_layer

    def forward(self, x):
        # Skip 'bn' - it's already fused into linear!
        layer_order = ['conv', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5',
                       'herpnpool', 'flatten', 'linear']

        for layer_name in layer_order:
            if hasattr(self, layer_name):
                x = getattr(self, layer_name)(x)
            if layer_name == self.stop_at:
                break
        return x

    def init_orion_params(self):
        """Initialize HerPN parameters from trained BatchNorm statistics."""
        self.layer1.init_orion_params()
        self.layer2.init_orion_params()
        self.layer3.init_orion_params()
        self.layer4.init_orion_params()
        self.layer5.init_orion_params()
        self.herpnpool.init_orion_params()
        # Note: Global BN already fused into linear weights during checkpoint loading


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


def compare_tensors(name, plaintext_out, fhe_out, verbose=True):
    """Compare plaintext and FHE outputs, handling packed tensors."""
    # Unpack FHE output if needed
    if plaintext_out.shape != fhe_out.shape:
        gap = calculate_gap(plaintext_out.shape, fhe_out.shape)
        fhe_out_unpacked = unpack_fhe_tensor(fhe_out, gap, plaintext_out.shape)
    else:
        fhe_out_unpacked = fhe_out

    if verbose:
        print(f"\n{name}:")
        print(f"  Plaintext: shape={plaintext_out.shape}, "
              f"range=[{plaintext_out.min():.4f}, {plaintext_out.max():.4f}]")
        if fhe_out.shape != plaintext_out.shape:
            print(f"  FHE (packed):   shape={fhe_out.shape}, "
                  f"range=[{fhe_out.min():.4f}, {fhe_out.max():.4f}]")
        print(f"  FHE (unpacked): shape={fhe_out_unpacked.shape}, "
              f"range=[{fhe_out_unpacked.min():.4f}, {fhe_out_unpacked.max():.4f}]")

        diff = (plaintext_out - fhe_out_unpacked).abs()
        rel_diff = diff / (plaintext_out.abs() + 1e-8)

        mae = diff.mean().item()
        max_abs = diff.max().item()
        mean_rel = rel_diff.mean().item()

        print(f"  Error: MAE={mae:.6f}, max_abs={max_abs:.6f}, mean_rel={mean_rel:.6f}")

        # Success threshold: MAE < 0.1 for high precision FHE
        if mae < 0.1:
            print(f"  ✓ Layer output acceptable (MAE < 0.1)")
            return True
        else:
            print(f"  ⚠ Layer error accumulating (MAE >= 0.1)")
            return False

    return True


def compare_single_patch_layerwise(patch_idx=0, stop_at='linear'):
    """Compare plaintext vs FHE inference layer-by-layer for a single patch.

    Args:
        patch_idx: Which patch to test (0-3)
        stop_at: Layer to stop at ('conv', 'layer1', ..., 'herpnpool', 'flatten', 'bn', 'linear')
    """

    print(f"\n{'='*80}")
    print(f"LAYER-BY-LAYER PLAINTEXT VS FHE COMPARISON (Patch {patch_idx}, stop_at={stop_at})")
    print(f"{'='*80}")
    print(f"Testing BackboneWithLinear: BN→Linear fusion for FHE")

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

    # Get full backbone for this patch
    original_backbone = model.nets[patch_idx]

    # Get the corresponding linear layer
    linear_layer = model.linear[patch_idx]

    print(f"\nCreating BackboneWithLinear:")
    print(f"  - Original backbone output dim: {model.backbone_dim}")
    print(f"  - Linear layer: {model.backbone_dim} → {model.embedding_dim}")
    print(f"  - This allows BN→Linear fusion during compilation")

    # Create backbone with linear layer
    backbone = BackboneWithLinear(original_backbone, linear_layer, stop_at=stop_at)

    # Call init_orion_params on the extended backbone
    backbone.init_orion_params()
    print(f"✓ BackboneWithLinear created")

    # Run plaintext inference layer-by-layer
    print(f"\n{'='*80}")
    print(f"Plaintext Inference (Layer-by-Layer, stopping at {stop_at})")
    print(f"{'='*80}")

    # Define layer order (skip 'bn' - already fused into linear!)
    layer_order = ['conv', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5',
                   'herpnpool', 'flatten', 'linear']
    stop_idx = layer_order.index(stop_at) if stop_at in layer_order else len(layer_order) - 1

    plaintext_outputs = {}
    with torch.no_grad():
        x = patch
        plaintext_outputs['input'] = x

        if stop_idx >= 0:
            x = backbone.conv(x)
            plaintext_outputs['conv'] = x
            if stop_at == 'conv': plaintext_outputs['final'] = x

        if stop_idx >= 1:
            x = backbone.layer1(x)
            plaintext_outputs['layer1'] = x
            if stop_at == 'layer1': plaintext_outputs['final'] = x

        if stop_idx >= 2:
            x = backbone.layer2(x)
            plaintext_outputs['layer2'] = x
            if stop_at == 'layer2': plaintext_outputs['final'] = x

        if stop_idx >= 3:
            x = backbone.layer3(x)
            plaintext_outputs['layer3'] = x
            if stop_at == 'layer3': plaintext_outputs['final'] = x

        if stop_idx >= 4:
            x = backbone.layer4(x)
            plaintext_outputs['layer4'] = x
            if stop_at == 'layer4': plaintext_outputs['final'] = x

        if stop_idx >= 5:
            x = backbone.layer5(x)
            plaintext_outputs['layer5'] = x
            if stop_at == 'layer5': plaintext_outputs['final'] = x

        if stop_idx >= 6:
            x = backbone.herpnpool(x)
            plaintext_outputs['herpnpool'] = x
            if stop_at == 'herpnpool': plaintext_outputs['final'] = x

        if stop_idx >= 7:
            x = backbone.flatten(x)
            plaintext_outputs['flatten'] = x
            if stop_at == 'flatten': plaintext_outputs['final'] = x

        # SKIP bn - already fused into linear!

        if stop_idx >= 8:
            x = backbone.linear(x)
            plaintext_outputs['linear'] = x
            if stop_at == 'linear': plaintext_outputs['final'] = x

    print(f"✓ Plaintext inference completed")

    # Fit and compile for FHE (just the backbone, not the full model)
    print(f"\n{'='*80}")
    print(f"Fitting and Compiling for FHE")
    print(f"{'='*80}")

    # Fit just the single backbone with the patch input
    orion.fit(backbone, patch)
    input_level = orion.compile(backbone)
    print(f"✓ Backbone compiled, input_level = {input_level}")

    # Encode and encrypt patch
    print(f"\n{'='*80}")
    print(f"Encrypting Patch {patch_idx}")
    print(f"{'='*80}")

    vec_ptxt = orion.encode(patch, input_level)
    vec_ctxt = orion.encrypt(vec_ptxt)
    print(f"✓ Patch encrypted")

    # Run FHE inference layer-by-layer
    print(f"\n{'='*80}")
    print(f"FHE Inference (Layer-by-Layer, stopping at {stop_at})")
    print(f"{'='*80}")

    backbone.he()  # Switch to FHE mode
    fhe_outputs = {}

    x_ctxt = vec_ctxt
    fhe_outputs['input'] = patch  # Keep plaintext for comparison

    if stop_idx >= 0:
        print(f"\nRunning conv...")
        x_ctxt = backbone.conv(x_ctxt)
        fhe_outputs['conv'] = x_ctxt.decrypt().decode()
        if stop_at == 'conv': fhe_outputs['final'] = fhe_outputs['conv']

    if stop_idx >= 1:
        print(f"Running layer1...")
        x_ctxt = backbone.layer1(x_ctxt)
        fhe_outputs['layer1'] = x_ctxt.decrypt().decode()
        if stop_at == 'layer1': fhe_outputs['final'] = fhe_outputs['layer1']

    if stop_idx >= 2:
        print(f"Running layer2...")
        x_ctxt = backbone.layer2(x_ctxt)
        fhe_outputs['layer2'] = x_ctxt.decrypt().decode()
        if stop_at == 'layer2': fhe_outputs['final'] = fhe_outputs['layer2']

    if stop_idx >= 3:
        print(f"Running layer3...")
        x_ctxt = backbone.layer3(x_ctxt)
        fhe_outputs['layer3'] = x_ctxt.decrypt().decode()
        if stop_at == 'layer3': fhe_outputs['final'] = fhe_outputs['layer3']

    if stop_idx >= 4:
        print(f"Running layer4...")
        x_ctxt = backbone.layer4(x_ctxt)
        fhe_outputs['layer4'] = x_ctxt.decrypt().decode()
        if stop_at == 'layer4': fhe_outputs['final'] = fhe_outputs['layer4']

    if stop_idx >= 5:
        print(f"Running layer5...")
        x_ctxt = backbone.layer5(x_ctxt)
        fhe_outputs['layer5'] = x_ctxt.decrypt().decode()
        if stop_at == 'layer5': fhe_outputs['final'] = fhe_outputs['layer5']

    if stop_idx >= 6:
        print(f"Running herpnpool...")
        x_ctxt = backbone.herpnpool(x_ctxt)
        fhe_outputs['herpnpool'] = x_ctxt.decrypt().decode()
        if stop_at == 'herpnpool': fhe_outputs['final'] = fhe_outputs['herpnpool']

    if stop_idx >= 7:
        print(f"Running flatten...")
        x_ctxt = backbone.flatten(x_ctxt)
        fhe_outputs['flatten'] = x_ctxt.decrypt().decode()
        if stop_at == 'flatten': fhe_outputs['final'] = fhe_outputs['flatten']

    # SKIP bn - already fused into linear!

    if stop_idx >= 8:
        print(f"Running linear...")
        x_ctxt = backbone.linear(x_ctxt)
        fhe_outputs['linear'] = x_ctxt.decrypt().decode()
        if stop_at == 'linear': fhe_outputs['final'] = fhe_outputs['linear']

    print(f"\n✓ FHE inference completed (stopped at {stop_at})")

    # Compare layer-by-layer
    print(f"\n{'='*80}")
    print(f"Layer-by-Layer Comparison (up to {stop_at})")
    print(f"{'='*80}")

    # Only compare layers up to stop_at (skip 'bn' - fused into linear!)
    all_layers = ['input', 'conv', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5',
                  'herpnpool', 'flatten', 'linear']
    layers = all_layers[:stop_idx + 2]  # +2 because we include 'input' and the stop layer

    cumulative_ok = True
    for layer_name in layers:
        if layer_name == 'input':
            continue  # Skip input comparison
        ok = compare_tensors(
            f"After {layer_name}",
            plaintext_outputs[layer_name],
            fhe_outputs[layer_name],
            verbose=True
        )
        if not ok:
            cumulative_ok = False

    # Final summary
    print(f"\n{'='*80}")
    if cumulative_ok:
        print(f"✓ ALL LAYERS ACCEPTABLE")
    else:
        print(f"⚠ SOME LAYERS HAVE HIGH ERROR")
    print(f"{'='*80}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Test BackboneWithLinear: layer-by-layer comparison (global BN skipped - fused into linear)"
    )
    parser.add_argument("--patch", type=int, default=0, help="Patch index (0-3)")
    parser.add_argument("--stop-at", type=str, default='linear',
                       help="Layer to stop at: conv, layer1, layer2, layer3, layer4, layer5, herpnpool, flatten, linear")
    args = parser.parse_args()

    compare_single_patch_layerwise(patch_idx=args.patch, stop_at=args.stop_at)
