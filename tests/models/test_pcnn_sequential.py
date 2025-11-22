"""
PCNN FHE Test - Sequential Execution (No Parallelization)

Tests the complete PCNN model with FHE encryption using sequential processing.
This provides a baseline for comparison with parallel implementations.

Uses the custom fit workflow:
    1. Collect BatchNorm statistics: model.eval(); for _ in range(20): model(data)
    2. Fuse HerPN: model.init_orion_params()  [BEFORE orion.fit()]
    3. Trace fused graph: orion.fit(model, inp)
    4. Compile: orion.compile(model)
    5. Process patches SEQUENTIALLY through backbones
"""
import torch
import orion
from models import PatchCNN
from pathlib import Path
import numpy as np
import time


def get_config_path(yml_str):
    orion_path = Path(__file__).parent.parent.parent
    return str(orion_path / "configs" / f"{yml_str}")


def test_pcnn_sequential():
    """
    Test the full PCNN network in FHE mode with SEQUENTIAL processing.

    Uses the custom fit workflow:
    1. Collect BatchNorm statistics
    2. Fuse HerPN before orion.fit()
    3. Orion traces the fused path (correct DAG structure)
    4. Process patches sequentially (no multiprocessing)
    """
    print("\n" + "="*70)
    print("PCNN FHE TEST - SEQUENTIAL EXECUTION (BASELINE)")
    print("="*70)

    torch.manual_seed(42)
    orion.init_scheme(get_config_path("pcnn.yml"))

    # Create a PatchCNN for 32x32 patches (matching CryptoFace)
    output_size = (4, 4)
    input_size = 64
    model = PatchCNN(input_size=input_size, patch_size=32, sqrt_weights=(1.0,1.0), output_size=output_size)

    # STEP 1: Collect BatchNorm statistics with custom fit
    print("\nStep 1: Collecting BatchNorm statistics...")
    model.eval()
    with torch.no_grad():
        for i in range(20):  # More samples for stable statistics
            _ = model(torch.randn(4, 3, input_size, input_size))
    print("  ✓ BatchNorm statistics collected")

    # STEP 2: Fuse HerPN BEFORE orion.fit()
    print("Step 2: Fusing HerPN activations...")
    model.init_orion_params()
    print("  ✓ HerPN modules fused")

    # Test input (single 64x64 image)
    inp = torch.randn(1, 3, input_size, input_size)

    # Cleartext forward with fused HerPN
    print("Step 3: Testing cleartext inference...")
    model.he_mode = True
    with torch.no_grad():
        out_clear = model(inp)
        out_clear = out_clear[0]
    print(f"  Cleartext output shape: {out_clear.shape}")
    print(f"  Cleartext output range: [{out_clear.min():.4f}, {out_clear.max():.4f}]")

    # STEP 3: Call orion.fit() - traces the fused HerPN path
    print("\nStep 4: Running orion.fit() (traces fused path)...")
    orion.fit(model, inp)
    print("  ✓ Orion fit completed")

    print("Step 5: Compiling network...")
    start_compile = time.time()
    input_level = orion.compile(model)
    compile_time = time.time() - start_compile
    print(f"  ✓ Compilation successful! Input level: {input_level}")
    print(f"  Compilation time: {compile_time:.2f} seconds")

    # Copy FHE weights from traced modules to original modules
    print("\nStep 6: Copying FHE weights...")
    from orion.core import scheme
    traced_model = scheme.trace

    def copy_herpn_weights(original_herpn, traced_path):
        try:
            traced_herpn = traced_model.get_submodule(traced_path)
            if hasattr(traced_herpn, 'w2_fhe'):
                original_herpn.w2_fhe = traced_herpn.w2_fhe
                original_herpn.w1_fhe = traced_herpn.w1_fhe
                original_herpn.w0_fhe = traced_herpn.w0_fhe
            elif hasattr(traced_herpn, 'w1_fhe'):
                original_herpn.w1_fhe = traced_herpn.w1_fhe
                original_herpn.w0_fhe = traced_herpn.w0_fhe
        except Exception as e:
            pass

    # Copy weights for all Backbone networks in PatchCNN
    for backbone_idx, backbone in enumerate(model.nets):
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4', 'layer5']:
            layer = getattr(backbone, layer_name)
            if hasattr(layer, 'herpn1') and layer.herpn1 is not None:
                copy_herpn_weights(layer.herpn1, f'nets.{backbone_idx}.{layer_name}.herpn1')
            if hasattr(layer, 'herpn2') and layer.herpn2 is not None:
                copy_herpn_weights(layer.herpn2, f'nets.{backbone_idx}.{layer_name}.herpn2')
            if hasattr(layer, 'shortcut_herpn') and layer.shortcut_herpn is not None:
                copy_herpn_weights(layer.shortcut_herpn, f'nets.{backbone_idx}.{layer_name}.shortcut_herpn')

        if hasattr(backbone.herpnpool, 'herpn') and backbone.herpnpool.herpn is not None:
            copy_herpn_weights(backbone.herpnpool.herpn, f'nets.{backbone_idx}.herpnpool.herpn')

    # Copy weights for normalization
    if hasattr(model, 'normalization') and hasattr(model.normalization, 'w2_fhe'):
        traced_norm = traced_model.get_submodule('normalization')
        model.normalization.w2_fhe = traced_norm.w2_fhe
        model.normalization.w1_fhe = traced_norm.w1_fhe
        model.normalization.w0_fhe = traced_norm.w0_fhe

    print("  ✓ FHE weights copied")

    # Switch to HE mode
    print("\nStep 7: Switching to HE mode...")
    model.he()
    print("  ✓ Model in HE mode")

    # Extract patches
    print("\nStep 8: Extracting patches...")
    H, W = model.H, model.W
    N = model.N
    P = model.patch_size

    patches = []
    for h in range(H):
        for w in range(W):
            patch = inp[:, :, h * P : (h + 1) * P, w * P : (w + 1) * P]
            patches.append(patch)
    print(f"  Extracted {N} patches")

    # Encrypt patches
    print("\nStep 9: Encrypting patches...")
    encrypted_patches = []
    for i, patch in enumerate(patches):
        patch_ptxt = orion.encode(patch, input_level)
        patch_ctxt = orion.encrypt(patch_ptxt)
        encrypted_patches.append(patch_ctxt)
    print(f"  Encrypted {N} patches")

    # Process patches SEQUENTIALLY through backbones
    print(f"\nStep 10: Processing {N} patches SEQUENTIALLY through backbones...")
    start_time = time.time()

    backbone_outputs_ctxt = []
    for i in range(N):
        print(f"  Processing patch {i}...")
        y_i = model.nets[i](encrypted_patches[i])
        backbone_outputs_ctxt.append(y_i)

    sequential_time = time.time() - start_time
    print(f"\n✓ Sequential backbone processing completed in {sequential_time:.2f} seconds")

    # Process through linear layers (ENCRYPTED!)
    print("\nStep 11: Processing through linear layers (encrypted)...")
    y_outputs = []
    for i in range(N):
        y_i = model.linear[i](backbone_outputs_ctxt[i])
        y_outputs.append(y_i)

    # Aggregate using tree reduction (ENCRYPTED!)
    print("Step 12: Aggregating results (encrypted)...")
    y = model._tree_reduce_add(y_outputs)

    # Apply normalization (ENCRYPTED!)
    print("Step 13: Applying normalization (encrypted)...")
    out_ctxt = model.normalization(y)

    # Decrypt only at the end!
    print("\nStep 14: Decrypting final result...")
    out_fhe = out_ctxt.decrypt().decode()

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\n{'='*70}")
    print("TIMING RESULTS")
    print(f"{'='*70}")
    print(f"Compilation time (once):          {compile_time:.2f} seconds")
    print(f"Sequential backbone inference:    {sequential_time:.2f} seconds")
    print(f"Linear + aggregation + normalize: {total_time - sequential_time:.2f} seconds")
    print(f"Total FHE inference time:         {total_time:.2f} seconds")
    print(f"{'='*70}")

    # Verify results
    print("\nStep 15: Verifying results...")
    print(f"FHE output shape: {out_fhe.shape}")
    print(f"FHE output range: [{out_fhe.min():.4f}, {out_fhe.max():.4f}]")

    out_clear_flat = out_clear.view(-1)
    out_fhe_flat = out_fhe.view(-1)

    dist = np.max(np.abs(out_clear_flat.numpy() - out_fhe_flat.numpy()))
    print(f"Max absolute error: {dist:.6f}")

    assert dist < 1.0, f"Error {dist:.6f} exceeds tolerance 1.0"

    print("\n" + "="*70)
    print("✓ PCNN FHE TEST PASSED (SEQUENTIAL)")
    print("="*70)
    print(f"End-to-end encryption maintained!")
    print(f"Total time: {total_time:.2f} seconds")


if __name__ == "__main__":
    test_pcnn_sequential()
