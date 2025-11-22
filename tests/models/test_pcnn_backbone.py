"""
PCNN FHE Test - Full Backbone Inference

Tests the complete PCNN Backbone model with FHE encryption using the custom fit workflow:
    1. Collect BatchNorm statistics: model.eval(); for _ in range(20): model(data)
    2. Fuse HerPN: model.init_orion_params()  [BEFORE orion.fit()]
    3. Trace fused graph: orion.fit(model, inp)
    4. Compile: orion.compile(model)
"""
import torch
import orion
from models import Backbone
from pathlib import Path
import numpy as np
import time


def get_config_path(yml_str):
    orion_path = Path(__file__).parent.parent.parent
    return str(orion_path / "configs" / f"{yml_str}")


def test_pcnn_single_backbone_fhe():
    """
    Test the full PCNN Backbone network in FHE mode.

    Uses the custom fit workflow:
    1. Collect BatchNorm statistics
    2. Fuse HerPN before orion.fit()
    3. Orion traces the fused path (correct DAG structure)
    """
    torch.manual_seed(42)
    orion.init_scheme(get_config_path("pcnn-backbone-fast.yml"))

    # Create a single backbone for 32x32 patches (matching CryptoFace)
    # Use (4,4) output_size to if (2,2) causes issues with rotations in pooling layer
    output_size = (4, 4)
    input_size = 32
    model = Backbone(output_size, input_size=input_size)

    # STEP 1: Collect BatchNorm statistics with custom fit
    print("\nStep 1: Collecting BatchNorm statistics for Backbone...")
    model.eval()
    with torch.no_grad():
        for i in range(20):  # More samples for stable statistics
            _ = model(torch.randn(4, 3, input_size, input_size))
    print("  ✓ BatchNorm statistics collected")

    # STEP 2: Fuse HerPN BEFORE orion.fit()
    print("Step 2: Fusing HerPN activations...")
    model.init_orion_params()
    print("  ✓ HerPN modules fused for all 5 layers + pool")

    # Test input (single 32x32 patch)
    inp = torch.randn(1, 3, input_size, input_size)

    # Cleartext forward with fused HerPN
    print("Step 3: Testing cleartext inference with fused HerPN...")
    with torch.no_grad():
        out_clear = model(inp)
    print(f"  Cleartext output shape: {out_clear.shape}")
    print(f"  Cleartext output range: [{out_clear.min():.4f}, {out_clear.max():.4f}]")

    # STEP 3: Call orion.fit() - traces the fused HerPN path
    print("Step 4: Running orion.fit() (traces fused path)...")
    orion.fit(model, inp)
    print("  ✓ Orion fit completed with correct DAG structure")

    print("Step 5: Compiling network...")
    input_level = orion.compile(model)
    print(f"  ✓ Compilation successful! Input level: {input_level}")

    # IMPORTANT: Copy FHE weights from traced modules to original modules
    print("Step 6: Copying FHE weights from traced to original modules...")
    from orion.core import scheme
    traced_model = scheme.trace

    # Helper function to copy FHE weights from a traced HerPN module to original
    def copy_herpn_weights(original_herpn, traced_path):
        try:
            traced_herpn = traced_model.get_submodule(traced_path)
            if hasattr(traced_herpn, 'w2_fhe'):
                original_herpn.w2_fhe = traced_herpn.w2_fhe
                original_herpn.w1_fhe = traced_herpn.w1_fhe
                original_herpn.w0_fhe = traced_herpn.w0_fhe
            elif hasattr(traced_herpn, 'w1_fhe'):  # HerPN without quadratic term
                original_herpn.w1_fhe = traced_herpn.w1_fhe
                original_herpn.w0_fhe = traced_herpn.w0_fhe
        except Exception as e:
            print(f"Warning: Could not copy weights for {traced_path}: {e}")

    # Copy weights for all HerPNConv layers
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4', 'layer5']:
        layer = getattr(model, layer_name)
        if hasattr(layer, 'herpn1') and layer.herpn1 is not None:
            copy_herpn_weights(layer.herpn1, f'{layer_name}.herpn1')
        if hasattr(layer, 'herpn2') and layer.herpn2 is not None:
            copy_herpn_weights(layer.herpn2, f'{layer_name}.herpn2')
        if hasattr(layer, 'shortcut_herpn') and layer.shortcut_herpn is not None:
            copy_herpn_weights(layer.shortcut_herpn, f'{layer_name}.shortcut_herpn')

    # Copy weights for HerPNPool
    if hasattr(model.herpnpool, 'herpn') and model.herpnpool.herpn is not None:
        copy_herpn_weights(model.herpnpool.herpn, 'herpnpool.herpn')

    print(f"Input level: {input_level}")

    # Run cleartext inference one more time to get the reference output
    model.eval()
    with torch.no_grad():
        out_clear_final = model(inp)

    print(f"\nCleartext output (final) shape: {out_clear_final.shape}")
    print(f"Cleartext output (final) range: [{out_clear_final.min():.4f}, {out_clear_final.max():.4f}]")

    # Encode and encrypt
    inp_ptxt = orion.encode(inp, input_level)
    inp_ctxt = orion.encrypt(inp_ptxt)

    # FHE mode
    model.he()

    # Measure encrypted inference time
    print("\nRunning encrypted inference...")
    start_time = time.time()
    out_ctxt = model(inp_ctxt)
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Encrypted inference time: {inference_time:.2f} seconds")

    out_fhe = out_ctxt.decrypt().decode()

    print(f"Backbone FHE output shape: {out_fhe.shape}")
    print(f"Backbone FHE output range: [{out_fhe.min():.4f}, {out_fhe.max():.4f}]")

    # Flatten both outputs for comparison
    out_clear_flat = out_clear_final.view(-1)
    out_fhe_flat = out_fhe.view(-1)

    # Compare with the final cleartext output
    dist = np.max(np.abs(out_clear_flat.numpy() - out_fhe_flat.numpy()))
    print(f"Backbone max absolute error: {dist:.6f}")

    assert dist < 1.0, f"Backbone error {dist:.6f} exceeds tolerance 1.0"

    print("✓ Single Backbone FHE test passed")

if __name__ == "__main__":
    test_pcnn_single_backbone_fhe()