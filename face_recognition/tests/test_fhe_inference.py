"""
Test FHE inference on CryptoFaceNet4 with fused BatchNorm.
"""
import sys
import torch

sys.path.append('/research/hal-vishnu/code/orion-fhe')

import orion
import orion.nn as on
from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4, Backbone
from face_recognition.models.weight_loader import load_checkpoint_for_config

from orion.core import scheme

class PartialBackbone(on.Module):
    """Wrapper to test just conv + layer1 from Backbone."""
    def __init__(self, layers):
        super().__init__()
        # Use ModuleList so layers are properly registered as submodules
        self.layer_list = torch.nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layer_list:
            x = layer(x)
        return x

    def init_orion_params(self):
        """Initialize HerPN parameters for layers."""
        for layer in self.layer_list:
            if hasattr(layer, 'init_orion_params'):
                layer.init_orion_params()


def test_fhe_backbone():
    """Test FHE inference on a single backbone."""
    print(f"\n{'='*80}")
    print("FHE Inference Test: CryptoFaceNet4 Backbone")
    print(f"{'='*80}\n")

    # Initialize CKKS scheme
    print("Initializing CKKS scheme...")
    orion.init_scheme("configs/pcnn-backbone.yml")
    print("✓ Scheme initialized\n")

    # Create model and load checkpoint
    print("Loading model...")
    # Use the full CryptoFaceNet4 to leverage the checkpoint loader
    full_model = CryptoFaceNet4()
    load_checkpoint_for_config(full_model, input_size=64, verbose=False)

    # Extract just the first backbone for testing
    backbone = full_model.nets[0]
    backbone.eval()
    print("✓ Model loaded\n")

    # Test the full backbone
    model = backbone  # Use the full backbone directly

    # STEP 1: Collect BatchNorm statistics (already done by checkpoint)
    # The checkpoint contains pre-trained running_mean and running_var
    print("BatchNorm statistics loaded from checkpoint\n")

    # STEP 2: Fuse BatchNorm BEFORE tracing
    print("Fusing BatchNorm into HerPN...")
    model.init_orion_params()
    print("✓ BatchNorm fused\n")

    # Create test input
    print("Creating test input...")
    torch.manual_seed(42)
    inp = torch.randn(1, 3, 32, 32)
    inp = (inp - inp.min()) / (inp.max() - inp.min()) * 2 - 1
    print(f"Input shape: {inp.shape}, range: [{inp.min():.4f}, {inp.max():.4f}]\n")

    # Get cleartext reference
    print("Running cleartext inference...")
    with torch.no_grad():
        out_cleartext = model(inp)
    print(f"Cleartext output: [{out_cleartext.min():.4f}, {out_cleartext.max():.4f}], mean={out_cleartext.mean():.4f}\n")

    # STEP 3: Trace fused graph
    print("Tracing model...")
    orion.fit(model, inp)
    print("✓ Model traced\n")

    # STEP 4: Compile and assign levels
    print("Compiling model and assigning levels...")
    input_level = orion.compile(model)
    print(f"✓ Model compiled, input_level={input_level}\n")

    # STEP 5: Copy FHE weights from traced modules to original modules
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

    # # DEBUG: Check HerPN coefficient values
    # print("\n=== DEBUG: HerPN Coefficients ===")
    # if hasattr(model.layer1, 'herpn1') and model.layer1.herpn1 is not None:
    #     h1 = model.layer1.herpn1
    #     print(f"herpn1.weight1_raw range: {h1.weight1_raw.min():.6f} to {h1.weight1_raw.max():.6f}")
    #     print(f"herpn1.weight0_raw range: {h1.weight0_raw.min():.6f} to {h1.weight0_raw.max():.6f}")
    #     if hasattr(h1, 'w1_fhe'):
    #         # Decode w1_fhe to check its actual values
    #         w1_decoded = orion.decode(h1.w1_fhe)
    #         w0_decoded = orion.decode(h1.w0_fhe)
    #         print(f"herpn1.w1_fhe (decoded) range: {w1_decoded.min():.6f} to {w1_decoded.max():.6f}")
    #         print(f"herpn1.w0_fhe (decoded) range: {w0_decoded.min():.6f} to {w0_decoded.max():.6f}")

    #         # Check if they match
    #         w1_diff = (h1.weight1_raw - w1_decoded).abs().max()
    #         w0_diff = (h1.weight0_raw - w0_decoded).abs().max()
    #         print(f"w1 difference (raw vs FHE): {w1_diff:.6f}")
    #         print(f"w0 difference (raw vs FHE): {w0_diff:.6f}")
    # print("=================================\n")

    # STEP 6: Encode and encrypt
    print("Encoding and encrypting input...")
    vec_ptxt = orion.encode(inp, input_level)
    vec_ctxt = orion.encrypt(vec_ptxt)
    print("✓ Input encrypted\n")

    # STEP 7: Run FHE inference
    print("Running FHE inference...")
    model.he()
    with torch.no_grad():
        out_ctxt = model(vec_ctxt)
    print("✓ FHE inference complete\n")

    # STEP 8: Decrypt and decode
    print("Decrypting output...")
    out_fhe = out_ctxt.decrypt().decode()
    print(f"FHE output: [{out_fhe.min():.4f}, {out_fhe.max():.4f}], mean={out_fhe.mean():.4f}\n")

    # Compare
    print("="*80)
    print("COMPARISON")
    print("="*80 + "\n")

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
    success = test_fhe_backbone()
    sys.exit(0 if success else 1)
