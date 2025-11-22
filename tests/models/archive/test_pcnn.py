"""
PCNN FHE Test - Full BackbonePCNN Inference

Tests the complete PCNN model with FHE encryption using the custom fit workflow:
    1. Collect BatchNorm statistics: model.eval(); for _ in range(20): model(data)
    2. Fuse HerPN: model.init_orion_params()  [BEFORE orion.fit()]
    3. Trace fused graph: orion.fit(model, inp)
    4. Compile: orion.compile(model)
"""
import torch
import orion
from models import PatchCNN, Backbone
from pathlib import Path
import numpy as np
import time
import multiprocessing


def get_config_path(yml_str):
    orion_path = Path(__file__).parent.parent.parent
    return str(orion_path / "configs" / f"{yml_str}")


def worker_process_patch(args):
    """
    Worker function to process a single patch through a backbone network.

    Each worker:
    1. Initializes FHE scheme
    2. Creates and compiles backbone model
    3. Processes encrypted patch
    4. Returns the result
    """
    patch_idx, config_path, encrypted_patch_data, input_level, output_size, input_size = args

    print(f"  Worker {patch_idx}: Starting (PID={multiprocessing.current_process().pid})")

    # Initialize scheme in this worker process
    orion.init_scheme(config_path)

    # Create backbone model (same as main process)
    torch.manual_seed(42)  # Same seed for reproducible weights
    backbone = Backbone(output_size, input_size=input_size)

    # Collect BatchNorm statistics
    backbone.eval()
    with torch.no_grad():
        for i in range(20):
            _ = backbone(torch.randn(4, 3, input_size, input_size))

    # Fuse HerPN
    backbone.init_orion_params()

    # Fit and compile
    sample_patch = torch.randn(1, 3, input_size, input_size)
    orion.fit(backbone, sample_patch)
    orion.compile(backbone)

    # Copy FHE weights from traced model
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
            pass  # Silent fail

    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4', 'layer5']:
        layer = getattr(backbone, layer_name)
        if hasattr(layer, 'herpn1') and layer.herpn1 is not None:
            copy_herpn_weights(layer.herpn1, f'{layer_name}.herpn1')
        if hasattr(layer, 'herpn2') and layer.herpn2 is not None:
            copy_herpn_weights(layer.herpn2, f'{layer_name}.herpn2')
        if hasattr(layer, 'shortcut_herpn') and layer.shortcut_herpn is not None:
            copy_herpn_weights(layer.shortcut_herpn, f'{layer_name}.shortcut_herpn')

    if hasattr(backbone.herpnpool, 'herpn') and backbone.herpnpool.herpn is not None:
        copy_herpn_weights(backbone.herpnpool.herpn, 'herpnpool.herpn')

    # Reconstruct the encrypted patch
    # encrypted_patch_data contains the ciphertext IDs
    # We need to recreate the CipherTensor from serialized data
    # For now, we'll need to re-encrypt the patch in this worker
    # This is a limitation - we can't easily pass encrypted data between processes

    # Since we can't serialize ciphertexts easily, we'll pass the plaintext patch
    # and encrypt it in the worker
    plaintext_patch = encrypted_patch_data
    patch_ptxt = orion.encode(plaintext_patch, input_level)
    patch_ctxt = orion.encrypt(patch_ptxt)

    # Switch to HE mode
    backbone.he()

    # Process patch
    y_i = backbone(patch_ctxt)

    # Decrypt and return as numpy
    # y_i_clear = y_i.decrypt().decode()

    print(f"  Worker {patch_idx}: Completed")

    # return patch_idx, y_i_clear.numpy()
    return patch_idx, y_i


def test_pcnn():
    """
    Test the full PCNN network in FHE mode.

    Uses the custom fit workflow:
    1. Collect BatchNorm statistics
    2. Fuse HerPN before orion.fit()
    3. Orion traces the fused path (correct DAG structure)
    """
    torch.manual_seed(42)
    orion.init_scheme(get_config_path("pcnn.yml"))

    # Create a PatchCNN for 32x32 patches (matching CryptoFace)
    # Use (4,4) output_size to if (2,2) causes issues with rotations in pooling layer
    output_size = (4, 4)
    input_size = 64
    model = PatchCNN(input_size=input_size, patch_size=32, sqrt_weights=(1.0,1.0), output_size=output_size)

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

    # Test input (single 64x64 image)
    inp = torch.randn(1, 3, input_size, input_size)

    # Cleartext forward with fused HerPN (BEFORE removing normalization)
    print("Step 3: Testing cleartext inference with fused HerPN...")
    model.he_mode = True
    with torch.no_grad():
        out_clear = model(inp)
        out_clear = out_clear[0]
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

    # Copy weights for all Backbone networks in PatchCNN
    # PatchCNN has model.nets (ModuleList of N Backbones)
    for backbone_idx, backbone in enumerate(model.nets):
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4', 'layer5']:
            layer = getattr(backbone, layer_name)
            if hasattr(layer, 'herpn1') and layer.herpn1 is not None:
                copy_herpn_weights(layer.herpn1, f'nets.{backbone_idx}.{layer_name}.herpn1')
            if hasattr(layer, 'herpn2') and layer.herpn2 is not None:
                copy_herpn_weights(layer.herpn2, f'nets.{backbone_idx}.{layer_name}.herpn2')
            if hasattr(layer, 'shortcut_herpn') and layer.shortcut_herpn is not None:
                copy_herpn_weights(layer.shortcut_herpn, f'nets.{backbone_idx}.{layer_name}.shortcut_herpn')

        # Copy weights for HerPNPool in each backbone
        if hasattr(backbone.herpnpool, 'herpn') and backbone.herpnpool.herpn is not None:
            copy_herpn_weights(backbone.herpnpool.herpn, f'nets.{backbone_idx}.herpnpool.herpn')

    # Copy weights for the normalization layer (ChannelSquare)
    if hasattr(model, 'normalization') and hasattr(model.normalization, 'w2_fhe'):
        traced_norm = traced_model.get_submodule('normalization')
        model.normalization.w2_fhe = traced_norm.w2_fhe
        model.normalization.w1_fhe = traced_norm.w1_fhe
        model.normalization.w0_fhe = traced_norm.w0_fhe

    print(f"Input level: {input_level}")

    # Run cleartext inference one more time to get the reference output
    model.eval()
    with torch.no_grad():
        out_clear_final = model(inp)

    print(f"\nCleartext output (final) shape: {out_clear_final.shape}")
    print(f"Cleartext output (final) range: [{out_clear_final.min():.4f}, {out_clear_final.max():.4f}]")

    # Extract patches in cleartext
    print("\nExtracting patches in cleartext...")
    B = inp.shape[0]
    H, W = model.H, model.W
    N = model.N
    P = model.patch_size

    patches = []
    for h in range(H):
        for w in range(W):
            patch = inp[:, :, h * P : (h + 1) * P, w * P : (w + 1) * P]
            patches.append(patch)
    print(f"  Extracted {len(patches)} patches of size {patches[0].shape}")

    # Prepare worker arguments
    # We pass plaintext patches - workers will encrypt them
    config_path = get_config_path("pcnn.yml")
    worker_args = [
        (i, config_path, patches[i], input_level, output_size, 32)  # patch_size is 32
        for i in range(N)
    ]

    # Process patches in parallel using multiprocessing
    print(f"\nProcessing {N} patches IN PARALLEL using multiprocessing...")
    start_time = time.time()

    with multiprocessing.Pool(processes=N) as pool:
        results = pool.map(worker_process_patch, worker_args)

    # Sort results by patch index
    results.sort(key=lambda x: x[0])
    backbone_outputs = [r[1] for r in results]

    print(f"\n✓ Parallel backbone processing completed in {time.time() - start_time:.2f} seconds")

    # Now process through linear layers and aggregate in main process
    print("\nProcessing through linear layers and aggregating...")

    # Switch model to HE mode for linear layer processing
    model.he()

    # Re-encrypt backbone outputs for linear layer processing
    y_outputs = []
    for i in range(N):
        # # Convert numpy back to tensor
        # y_i_tensor = torch.from_numpy(backbone_outputs[i]).float()

        # # Encode and encrypt
        # y_i_ptxt = orion.encode(y_i_tensor, input_level)  # Use same level
        # y_i_ctxt = orion.encrypt(y_i_ptxt)

        # Process through linear layer
        # y_i = model.linear[i](y_i_ctxt)
        y_i = model.linear[i](backbone_outputs[i])  # Directly use CipherTensor from worker
        y_outputs.append(y_i)

    # Aggregate using tree reduction
    y = model._tree_reduce_add(y_outputs)

    # Apply normalization
    out_ctxt = model.normalization(y)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal encrypted inference time: {total_time:.2f} seconds")
    print(f"  (includes parallel backbone processing + linear layers + aggregation)")

    out_fhe = out_ctxt.decrypt().decode()

    print(f"PatchCNN FHE output shape: {out_fhe.shape}")
    print(f"PatchCNN FHE output range: [{out_fhe.min():.4f}, {out_fhe.max():.4f}]")

    # Flatten both outputs for comparison
    out_clear_flat = out_clear_final.view(-1)
    out_fhe_flat = out_fhe.view(-1)

    # Compare with the final cleartext output
    dist = np.max(np.abs(out_clear_flat.numpy() - out_fhe_flat.numpy()))
    print(f"PatchCNN max absolute error: {dist:.6f}")

    assert dist < 1.0, f"PatchCNN error {dist:.6f} exceeds tolerance 1.0"

    print("✓ PatchCNN FHE test passed")

if __name__ == "__main__":
    # Required for multiprocessing on some platforms
    multiprocessing.set_start_method('spawn', force=True)
    test_pcnn()