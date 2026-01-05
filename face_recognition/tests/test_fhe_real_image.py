"""
Test FHE inference on CryptoFaceNet4 with a real LFW image.
"""
import sys
import glob
import random
import torch
from PIL import Image
from torchvision import transforms

sys.path.append('/research/hal-vishnu/code/orion-fhe')

import orion
import orion.nn as on
from orion.core import scheme
from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4, Backbone
from face_recognition.models.weight_loader import load_checkpoint_for_config


def load_lfw_image_patch(lfw_dir, image_path=None, patch_size=32):
    """Load a single image from LFW MTCNN-aligned folder and extract a patch.

    Args:
        lfw_dir: Path to lfw-mtcnn-aligned directory
        image_path: Optional specific image path. If None, picks random image.
        patch_size: Size of patch to extract (default: 32)

    Returns:
        Preprocessed patch tensor [1, 3, 32, 32] normalized to [-1, 1]
    """
    if image_path is None:
        # Find all JPG images
        all_images = glob.glob(f"{lfw_dir}/**/*.jpg", recursive=True)
        if not all_images:
            raise ValueError(f"No images found in {lfw_dir}")
        # Pick a random image
        image_path = random.choice(all_images)

    print(f"Loading image: {image_path}")

    # Load image
    img = Image.open(image_path).convert('RGB')
    print(f"Original image size: {img.size}")

    # Resize to 64×64 (CryptoFace input size for Net4)
    img = img.resize((64, 64), Image.Resampling.BILINEAR)
    print(f"Resized to: {img.size}")

    # Transform to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0, 255] uint8 -> [0, 1] float32
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # -> [-1, 1]
    ])

    img_tensor = transform(img).unsqueeze(0)  # [1, 3, 64, 64]
    print(f"Full image tensor: {img_tensor.shape}, range: [{img_tensor.min():.4f}, {img_tensor.max():.4f}]")

    # Extract top-left patch (matching CryptoFace patch extraction)
    # CryptoFaceNet4 creates 2×2 grid of patches from 64×64 image
    patch = img_tensor[:, :, 0:patch_size, 0:patch_size]  # Top-left 32×32 patch
    print(f"Extracted patch: {patch.shape}, range: [{patch.min():.4f}, {patch.max():.4f}]")

    return patch


def test_fhe_real_image():
    """Test FHE inference on a real LFW image."""
    print(f"\n{'='*80}")
    print("FHE Inference Test: CryptoFaceNet4 Backbone with Real LFW Image")
    print(f"{'='*80}\n")

    # Load a random real image from LFW MTCNN-aligned
    lfw_dir = "/research/hal-datastage/datasets/original/LFW/lfw-mtcnn-aligned"
    inp = load_lfw_image_patch(lfw_dir)
    print()

    # Initialize CKKS scheme
    print("Initializing CKKS scheme...")
    orion.init_scheme("configs/pcnn-backbone.yml")
    print("✓ Scheme initialized\n")

    # Create model and load checkpoint
    print("Loading model...")
    full_model = CryptoFaceNet4()
    load_checkpoint_for_config(full_model, input_size=64, verbose=False)

    # Extract just the second backbone for testing (nets[1] instead of nets[0])
    model = full_model.nets[1]
    model.eval()
    print("✓ Model loaded (using backbone nets[1])\n")

    # Get cleartext reference
    print("Running cleartext inference...")
    with torch.no_grad():
        out_cleartext = model(inp)
    print(f"Cleartext output shape: {out_cleartext.shape}")
    print(f"Cleartext output: [{out_cleartext.min():.4f}, {out_cleartext.max():.4f}], mean={out_cleartext.mean():.4f}\n")

    # STEP 1: Fuse BatchNorm BEFORE tracing
    print("Fusing BatchNorm into HerPN...")
    model.init_orion_params()
    print("✓ BatchNorm fused\n")

    # STEP 2: Trace fused graph
    print("Tracing model...")
    orion.fit(model, inp)
    print("✓ Model traced\n")

    # STEP 3: Compile and assign levels
    print("Compiling model and assigning levels...")
    input_level = orion.compile(model)
    print(f"✓ Model compiled, input_level={input_level}\n")

    # STEP 4: Copy FHE weights from traced modules to original modules
    # CRITICAL: orion.compile() creates FHE weights on the traced model (a copy).
    # We must copy them to the original model before FHE inference.
    print("Copying FHE weights from traced to original modules...")

    traced_model = scheme.trace

    def copy_herpn_weights(original_herpn, traced_path):
        """Copy FHE weights from traced HerPN/ChannelSquare to original."""
        try:
            traced_herpn = traced_model.get_submodule(traced_path)
            if hasattr(traced_herpn, 'w1_fhe'):
                print(f"  Copying w1_fhe and w0_fhe from {traced_path}")
                original_herpn.w1_fhe = traced_herpn.w1_fhe
                original_herpn.w0_fhe = traced_herpn.w0_fhe
            else:
                print(f"  Warning: {traced_path} has no w1_fhe")
        except Exception as e:
            print(f"  Warning: Could not copy weights for {traced_path}: {e}")

    # Copy weights for all HerPNConv layers in backbone
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4', 'layer5']:
        if hasattr(model, layer_name):
            layer = getattr(model, layer_name)
            # Copy HerPN weights
            if hasattr(layer, 'herpn1') and layer.herpn1 is not None:
                copy_herpn_weights(layer.herpn1, f'{layer_name}.herpn1')
            if hasattr(layer, 'herpn2') and layer.herpn2 is not None:
                copy_herpn_weights(layer.herpn2, f'{layer_name}.herpn2')

            # Copy ScaleModule weights (shortcut_scale)
            if hasattr(layer, 'shortcut_scale') and layer.shortcut_scale is not None:
                try:
                    traced_scale = traced_model.get_submodule(f'{layer_name}.shortcut_scale')
                    if hasattr(traced_scale, 'scale_fhe'):
                        print(f"  Copying scale_fhe from {layer_name}.shortcut_scale")
                        layer.shortcut_scale.scale_fhe = traced_scale.scale_fhe
                    else:
                        print(f"  Warning: {layer_name}.shortcut_scale has no scale_fhe")
                except Exception as e:
                    print(f"  Warning: Could not copy scale for {layer_name}.shortcut_scale: {e}")

    # Copy weights for HerPNPool
    if hasattr(model, 'herpnpool') and hasattr(model.herpnpool, 'herpn') and model.herpnpool.herpn is not None:
        copy_herpn_weights(model.herpnpool.herpn, 'herpnpool.herpn')

    # Copy pool_scale for HerPNPool
    if hasattr(model, 'herpnpool') and hasattr(model.herpnpool, 'pool_scale') and model.herpnpool.pool_scale is not None:
        try:
            traced_pool_scale = traced_model.get_submodule('herpnpool.pool_scale')
            if hasattr(traced_pool_scale, 'scale_fhe'):
                print(f"  Copying scale_fhe from herpnpool.pool_scale")
                model.herpnpool.pool_scale.scale_fhe = traced_pool_scale.scale_fhe
            else:
                print(f"  Warning: herpnpool.pool_scale has no scale_fhe")
        except Exception as e:
            print(f"  Warning: Could not copy scale for herpnpool.pool_scale: {e}")

    print("✓ FHE weights copied\n")

    # STEP 5: Encode and encrypt
    print("Encoding and encrypting input...")
    vec_ptxt = orion.encode(inp, input_level)
    vec_ctxt = orion.encrypt(vec_ptxt)
    print("✓ Input encrypted\n")

    # STEP 6: Run FHE inference
    print("Running FHE inference...")
    model.he()
    with torch.no_grad():
        out_ctxt = model(vec_ctxt)
    print("✓ FHE inference complete\n")

    # STEP 7: Decrypt and decode
    print("Decrypting output...")
    out_fhe = out_ctxt.decrypt().decode()
    print(f"FHE output shape: {out_fhe.shape}")
    print(f"FHE output: [{out_fhe.min():.4f}, {out_fhe.max():.4f}], mean={out_fhe.mean():.4f}\n")

    # Compare
    print("="*80)
    print("COMPARISON")
    print("="*80 + "\n")

    print(f"Cleartext shape: {out_cleartext.shape}")
    print(f"FHE shape: {out_fhe.shape}\n")

    # Handle shape mismatch due to FHE packing
    if out_cleartext.shape != out_fhe.shape:
        print(f"⚠️  Shape mismatch due to FHE packing (expected)")
        print(f"   Comparing first {out_cleartext.numel()} elements...\n")
        out_fhe_flat = out_fhe.flatten()[:out_cleartext.numel()].reshape(out_cleartext.shape)
        diff = (out_cleartext - out_fhe_flat).abs()
    else:
        diff = (out_cleartext - out_fhe).abs()

    mae = diff.mean().item()
    max_err = diff.max().item()

    print(f"MAE: {mae:.6f}")
    print(f"Max error: {max_err:.6f}")
    print(f"Relative error: {mae / out_cleartext.abs().mean().item():.6f}\n")

    if mae < 1.0:
        print("✓ FHE inference SUCCESSFUL!")
        return True
    else:
        print("✗ FHE inference has large errors")
        return False


if __name__ == "__main__":
    success = test_fhe_real_image()
    sys.exit(0 if success else 1)
