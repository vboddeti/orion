"""
PCNN FHE Test - Option D (Fork + Serialization)

Uses fork to inherit compiled model and serialization for encrypted data transfer.
Maintains end-to-end encryption without intermediate decrypt/re-encrypt.
"""
import torch
import orion
from models import PatchCNN
from pathlib import Path
import numpy as np
import time
import multiprocessing


def get_config_path(yml_str):
    orion_path = Path(__file__).parent.parent.parent
    return str(orion_path / "configs" / f"{yml_str}")


# Global model for fork to inherit
_global_model = None


def worker_process_patch_fork(patch_idx, serialized_patch):
    """
    Worker function using fork + serialization.

    Inherits compiled model via fork, receives serialized encrypted patch,
    processes it, and returns serialized encrypted result.
    """
    global _global_model
    print(f"  Worker {patch_idx}: Starting (using inherited model via fork)...")

    # Deserialize input patch (scheme is inherited via fork)
    from orion.core import scheme
    from orion.backend.python.tensors import CipherTensor
    patch_ctxt = CipherTensor.deserialize(scheme, serialized_patch)

    # Process through backbone (encrypted! using inherited compiled model!)
    y_i = _global_model.nets[patch_idx](patch_ctxt)

    # Serialize encrypted result
    serialized_result = y_i.serialize()

    print(f"  Worker {patch_idx}: Completed")

    return patch_idx, serialized_result


def test_pcnn_fork():
    """
    Test PCNN with Option D: Fork + Serialization.

    Compiles model once, forks workers that inherit it, passes encrypted
    data via serialization, maintains end-to-end encryption.
    """
    global _global_model

    print("\n" + "="*70)
    print("PCNN FHE TEST - Option D (Fork + Serialization)")
    print("="*70)

    torch.manual_seed(42)
    orion.init_scheme(get_config_path("pcnn.yml"))

    # Create PCNN model
    output_size = (4, 4)
    input_size = 64
    model = PatchCNN(input_size=input_size, patch_size=32,
                     sqrt_weights=(1.0, 1.0), output_size=output_size)
    _global_model = model  # Set global for fork to inherit

    # STEP 1: Collect BatchNorm statistics
    print("\nStep 1: Collecting BatchNorm statistics...")
    model.eval()
    with torch.no_grad():
        for i in range(20):
            _ = model(torch.randn(4, 3, input_size, input_size))
    print("  ✓ BatchNorm statistics collected")

    # STEP 2: Fuse HerPN
    print("Step 2: Fusing HerPN activations...")
    model.init_orion_params()
    print("  ✓ HerPN modules fused")

    # Test input
    inp = torch.randn(1, 3, input_size, input_size)

    # Cleartext forward
    print("Step 3: Testing cleartext inference...")
    model.he_mode = True
    with torch.no_grad():
        out_clear = model(inp)[0]
    print(f"  Cleartext output shape: {out_clear.shape}")
    print(f"  Cleartext output range: [{out_clear.min():.4f}, {out_clear.max():.4f}]")

    # STEP 3: Compile (ONCE!)
    print("\nStep 4: Compiling network (ONCE, before forking)...")
    start_compile = time.time()
    orion.fit(model, inp)
    input_level = orion.compile(model)
    compile_time = time.time() - start_compile
    print(f"  ✓ Compilation successful! Input level: {input_level}")
    print(f"  Compilation time: {compile_time:.2f} seconds")

    # Copy FHE weights (same as before)
    print("Step 5: Copying FHE weights...")
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

    if hasattr(model, 'normalization') and hasattr(model.normalization, 'w2_fhe'):
        traced_norm = traced_model.get_submodule('normalization')
        model.normalization.w2_fhe = traced_norm.w2_fhe
        model.normalization.w1_fhe = traced_norm.w1_fhe
        model.normalization.w0_fhe = traced_norm.w0_fhe

    print("  ✓ FHE weights copied")

    # Switch model to HE mode (CRITICAL!)
    print("Step 5b: Switching model to HE mode...")
    model.he()
    print("  ✓ Model switched to HE mode")

    # Extract patches and encrypt
    print("\nStep 6: Extracting and encrypting patches...")
    H, W = model.H, model.W
    N = model.N
    P = model.patch_size

    patches = []
    for h in range(H):
        for w in range(W):
            patch = inp[:, :, h * P : (h + 1) * P, w * P : (w + 1) * P]
            patches.append(patch)

    # Encrypt patches
    encrypted_patches = []
    for i, patch in enumerate(patches):
        patch_ptxt = orion.encode(patch, input_level)
        patch_ctxt = orion.encrypt(patch_ptxt)
        encrypted_patches.append(patch_ctxt)

    print(f"  Encrypted {N} patches")

    # Serialize encrypted patches
    print("Step 7: Serializing encrypted patches...")
    serialized_patches = [patch_ctxt.serialize() for patch_ctxt in encrypted_patches]
    total_size = sum(sum(len(d) for d in sp['data']) for sp in serialized_patches)
    print(f"  Total serialized size: {total_size / 1024 / 1024:.2f} MB")

    # Process patches in parallel using fork
    print(f"\nStep 8: Processing {N} patches IN PARALLEL (fork + serialization)...")
    start_time = time.time()

    with multiprocessing.Pool(processes=N) as pool:
        results = pool.starmap(
            worker_process_patch_fork,
            [(i, serialized_patches[i]) for i in range(N)]
        )

    parallel_time = time.time() - start_time
    print(f"\n✓ Parallel backbone processing completed in {parallel_time:.2f} seconds")

    # Deserialize results
    print("\nStep 9: Deserializing backbone outputs...")
    from orion.backend.python.tensors import CipherTensor
    results.sort(key=lambda x: x[0])
    backbone_outputs_ctxt = []
    for idx, serialized_result in results:
        y_i_ctxt = CipherTensor.deserialize(scheme, serialized_result)
        backbone_outputs_ctxt.append(y_i_ctxt)

    print(f"  Deserialized {N} encrypted backbone outputs")

    # Process through linear layers (ENCRYPTED!)
    print("\nStep 10: Processing through linear layers (encrypted)...")
    model.he()
    y_outputs = []
    for i in range(N):
        y_i = model.linear[i](backbone_outputs_ctxt[i])  # Still encrypted!
        y_outputs.append(y_i)

    # Aggregate (ENCRYPTED!)
    print("Step 11: Aggregating results (encrypted)...")
    y = model._tree_reduce_add(y_outputs)

    # Normalize (ENCRYPTED!)
    print("Step 12: Applying normalization (encrypted)...")
    out_ctxt = model.normalization(y)

    # Decrypt only at the end!
    print("\nStep 13: Decrypting final result...")
    out_fhe = out_ctxt.decrypt().decode()

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\n{'='*70}")
    print("TIMING RESULTS")
    print(f"{'='*70}")
    print(f"Compilation time (once):          {compile_time:.2f} seconds")
    print(f"Parallel backbone inference:      {parallel_time:.2f} seconds")
    print(f"Linear + aggregation + normalize: {total_time - parallel_time:.2f} seconds")
    print(f"Total FHE inference time:         {total_time:.2f} seconds")
    print(f"{'='*70}")

    # Verify results
    print("\nStep 14: Verifying results...")
    print(f"FHE output shape: {out_fhe.shape}")
    print(f"FHE output range: [{out_fhe.min():.4f}, {out_fhe.max():.4f}]")

    out_clear_flat = out_clear.view(-1)
    out_fhe_flat = out_fhe.view(-1)

    dist = np.max(np.abs(out_clear_flat.numpy() - out_fhe_flat.numpy()))
    print(f"Max absolute error: {dist:.6f}")

    assert dist < 1.0, f"Error {dist:.6f} exceeds tolerance 1.0"

    print("\n" + "="*70)
    print("✓ PCNN FHE TEST PASSED (Option D)")
    print("="*70)
    print(f"End-to-end encryption maintained!")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Speedup vs sequential: ~{560/total_time:.2f}x (estimated)")


if __name__ == "__main__":
    multiprocessing.set_start_method('fork', force=True)
    test_pcnn_fork()
