"""
Test FHE inference with detailed path logging to debug layer3/layer4 errors.
"""
import sys
import torch

sys.path.append('/research/hal-vishnu/code/orion-fhe')

import orion
import orion.nn as on
from orion.core import scheme
from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4
from face_recognition.models.weight_loader import load_checkpoint_for_config


def test_fhe_with_path_logging():
    """Test FHE inference with detailed logging of residual paths."""
    print(f"\n{'='*80}")
    print("FHE Path Debug Test: CryptoFaceNet4 Backbone")
    print(f"{'='*80}\n")

    # Use a fixed input for reproducibility
    inp = torch.randn(1, 3, 32, 32)
    print(f"Input shape: {inp.shape}")
    print(f"Input range: [{inp.min():.4f}, {inp.max():.4f}]\n")

    # Initialize CKKS scheme
    print("Initializing CKKS scheme...")
    orion.init_scheme("configs/pcnn-backbone.yml")
    print("✓ Scheme initialized\n")

    # Create model and load checkpoint
    print("Loading model...")
    full_model = CryptoFaceNet4()
    load_checkpoint_for_config(full_model, input_size=64, verbose=False)

    # Extract backbone (using nets[1])
    model = full_model.nets[1]
    model.eval()
    print("✓ Model loaded (using backbone nets[1])\n")

    # Get cleartext reference
    print("Running cleartext inference...")
    with torch.no_grad():
        out_cleartext = model(inp)
    print(f"Cleartext output: [{out_cleartext.min():.4f}, {out_cleartext.max():.4f}]\n")

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
    print("Copying FHE weights from traced to original modules...")
    traced_model = scheme.trace

    def copy_herpn_weights(original_herpn, traced_path):
        """Copy FHE weights from traced HerPN to original."""
        try:
            traced_herpn = traced_model.get_submodule(traced_path)
            if hasattr(traced_herpn, 'w1_fhe'):
                original_herpn.w1_fhe = traced_herpn.w1_fhe
                original_herpn.w0_fhe = traced_herpn.w0_fhe
        except Exception as e:
            print(f"  Warning: Could not copy weights for {traced_path}: {e}")

    # Copy weights for all layers
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4', 'layer5']:
        if hasattr(model, layer_name):
            layer = getattr(model, layer_name)
            if hasattr(layer, 'herpn1') and layer.herpn1 is not None:
                copy_herpn_weights(layer.herpn1, f'{layer_name}.herpn1')
            if hasattr(layer, 'herpn2') and layer.herpn2 is not None:
                copy_herpn_weights(layer.herpn2, f'{layer_name}.herpn2')

            # Copy ScaleModule weights
            if hasattr(layer, 'shortcut_scale') and layer.shortcut_scale is not None:
                try:
                    traced_scale = traced_model.get_submodule(f'{layer_name}.shortcut_scale')
                    if hasattr(traced_scale, 'scale_fhe'):
                        layer.shortcut_scale.scale_fhe = traced_scale.scale_fhe
                except Exception as e:
                    pass

    # Copy HerPNPool weights
    if hasattr(model, 'herpnpool') and hasattr(model.herpnpool, 'herpn'):
        copy_herpn_weights(model.herpnpool.herpn, 'herpnpool.herpn')
    if hasattr(model, 'herpnpool') and hasattr(model.herpnpool, 'pool_scale'):
        try:
            traced_pool_scale = traced_model.get_submodule('herpnpool.pool_scale')
            if hasattr(traced_pool_scale, 'scale_fhe'):
                model.herpnpool.pool_scale.scale_fhe = traced_pool_scale.scale_fhe
        except:
            pass

    print("✓ FHE weights copied\n")

    # STEP 5: Encode and encrypt
    print("Encoding and encrypting input...")
    vec_ptxt = orion.encode(inp, input_level)
    vec_ctxt = orion.encrypt(vec_ptxt)
    print("✓ Input encrypted\n")

    # STEP 6: Run cleartext again to get path values
    print("="*80)
    print("CLEARTEXT PATH ANALYSIS")
    print("="*80 + "\n")

    # Model is already in cleartext mode after compilation
    with torch.no_grad():
        for layer_name in ['layer3', 'layer4']:
            if hasattr(model, layer_name):
                layer = getattr(model, layer_name)

                # Get input to this layer
                if layer_name == 'layer3':
                    # Run up to layer2
                    x = inp
                    x = model.conv(x)
                    x = model.layer1(x)
                    x = model.layer2(x)
                    layer_input = x
                elif layer_name == 'layer4':
                    # Run up to layer3
                    x = inp
                    x = model.conv(x)
                    x = model.layer1(x)
                    x = model.layer2(x)
                    x = model.layer3(x)
                    layer_input = x

                # Run the layer
                layer_output = layer(layer_input)

                # Extract path values
                main_min = layer._debug_paths.get('main_path', torch.tensor([0])).min().item() if hasattr(layer, '_debug_paths') else 0
                main_max = layer._debug_paths.get('main_path', torch.tensor([0])).max().item() if hasattr(layer, '_debug_paths') else 0
                short_min = layer._debug_paths.get('shortcut_path', torch.tensor([0])).min().item() if hasattr(layer, '_debug_paths') else 0
                short_max = layer._debug_paths.get('shortcut_path', torch.tensor([0])).max().item() if hasattr(layer, '_debug_paths') else 0
                add_min = layer_output.min().item()
                add_max = layer_output.max().item()

                print(f"{layer_name} (cleartext):")
                print(f"  Main path (conv2):     [{main_min:12.4f}, {main_max:12.4f}]")
                print(f"  Shortcut path:         [{short_min:12.4f}, {short_max:12.4f}]")
                print(f"  After add:             [{add_min:12.4f}, {add_max:12.4f}]")
                print()

    # STEP 7: Run FHE inference
    print("="*80)
    print("FHE PATH ANALYSIS")
    print("="*80 + "\n")

    model.he()
    with torch.no_grad():
        # Run layer by layer and log paths
        x_ctxt = vec_ctxt
        x_ctxt = model.conv(x_ctxt)
        x_ctxt = model.layer1(x_ctxt)
        x_ctxt = model.layer2(x_ctxt)

        # Layer3 with path logging
        layer3_out = model.layer3(x_ctxt)
        if hasattr(model.layer3, '_debug_paths'):
            main_ctxt = model.layer3._debug_paths.get('main_path')
            short_ctxt = model.layer3._debug_paths.get('shortcut_path')
            add_ctxt = model.layer3._debug_paths.get('after_add')

            main_fhe = main_ctxt.decrypt().decode()
            short_fhe = short_ctxt.decrypt().decode()
            add_fhe = add_ctxt.decrypt().decode()

            print(f"layer3 (FHE):")
            print(f"  Main path (conv2):     [{main_fhe.min():12.4f}, {main_fhe.max():12.4f}]")
            print(f"  Shortcut path:         [{short_fhe.min():12.4f}, {short_fhe.max():12.4f}]")
            print(f"  After add:             [{add_fhe.min():12.4f}, {add_fhe.max():12.4f}]")
            print()

        # Layer4 with path logging
        layer4_out = model.layer4(layer3_out)
        if hasattr(model.layer4, '_debug_paths'):
            main_ctxt = model.layer4._debug_paths.get('main_path')
            short_ctxt = model.layer4._debug_paths.get('shortcut_path')
            add_ctxt = model.layer4._debug_paths.get('after_add')

            main_fhe = main_ctxt.decrypt().decode()
            short_fhe = short_ctxt.decrypt().decode()
            add_fhe = add_ctxt.decrypt().decode()

            print(f"layer4 (FHE):")
            print(f"  Main path (conv2):     [{main_fhe.min():12.4f}, {main_fhe.max():12.4f}]")
            print(f"  Shortcut path:         [{short_fhe.min():12.4f}, {short_fhe.max():12.4f}]")
            print(f"  After add:             [{add_fhe.min():12.4f}, {add_fhe.max():12.4f}]")
            print()

    print("="*80)
    print("Test complete - check path values above!")
    print("="*80)


if __name__ == "__main__":
    test_fhe_with_path_logging()
