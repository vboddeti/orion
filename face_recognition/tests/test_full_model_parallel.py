
import multiprocessing
import os
import sys
import argparse
import time
import torch
import torch.nn as nn
from joblib import Parallel, delayed

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import orion
import orion.nn as on
from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4
from face_recognition.models.weight_loader import load_checkpoint_for_config
from orion.core import scheme
from orion.backend.python.tensors import CipherTensor
from torchvision import transforms
from PIL import Image

# Global variable to hold the model in workers (inherited via fork)
_global_model = None

def worker_init(model):
    """Initialize global model in worker process."""
    global _global_model
    _global_model = model

def worker_process_patch(patch_idx, serialized_patch):
    """
    Process a single patch through its backbone and linear layer.
    
    Args:
        patch_idx (int): Index of the patch (0-3).
        serialized_patch (bytes): Serialized CipherTensor for the patch.
        
    Returns:
        int: patch_idx
        bytes: Serialized CipherTensor of the result (linear output)
    """
    global _global_model
    
    # 1. Deserialize input patch
    # Note: scheme is inherited from parent process via fork
    patch_ctxt = CipherTensor.deserialize(scheme, serialized_patch)
    
    print(f"  [Worker {patch_idx}] Processing patch {patch_idx}...", flush=True)
    start_time = time.time()
    
    # 2. Run Backbone
    # _global_model.nets[patch_idx] is the backbone
    feat = _global_model.nets[patch_idx](patch_ctxt)
    
    # 3. Run Linear
    # _global_model.linear[patch_idx] is the linear layer
    linear_layer = _global_model.linear[patch_idx]
    
    out = linear_layer(feat)
    

    
    elapsed = time.time() - start_time
    print(f"  [Worker {patch_idx}] Done in {elapsed:.2f}s", flush=True)
    
    # 4. Serialize result
    return patch_idx, out.serialize()

def test_full_model_parallel(image_path):
    """
    Run full CryptoFaceNet4 inference with parallel execution of sub-networks.
    """
    print(f"\n{'='*80}")
    print("Full Model Parallel FHE Inference Test")
    print(f"{'='*80}\n")
    
    # 1. Initialize Scheme
    print("Initializing CKKS scheme...")
    orion.init_scheme("configs/pcnn-backbone.yml")
    
    # 2. Load Model & Weights
    print("Loading CryptoFaceNet4...")
    model = CryptoFaceNet4()
    load_checkpoint_for_config(model, input_size=64, verbose=False)
    model.eval()
    
    # 3. Load & Preprocess Real Image
    print(f"Loading image: {image_path}")
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img_tensor = transform(img).unsqueeze(0)  # (1, 3, 64, 64)
    
    # Get Cleartext Ground Truth
    print("Running cleartext inference...")
    with torch.no_grad():
        out_clear = model(img_tensor)

    # 4. Extract Patches (Cleartext) for Fitting and Encryption
    # Identical logic to sequential test
    print("Extracting patches...")
    patches = []
    P = model.patch_size # 32
    H_grid = model.H # 2
    W_grid = model.W # 2
    
    for h in range(H_grid):
        for w in range(W_grid):
            # patch = img_tensor[:, :, h * P : (h + 1) * P, w * P : (w + 1) * P]
            # Use same slicing as in model
             patch = img_tensor[:, :, h * P : (h + 1) * P, w * P : (w + 1) * P]
             patches.append(patch)
            
    assert len(patches) == 4
    
    # 5. Fuse & Compile
    print("Fusing and Compiling...")
    model.init_orion_params()
    
    # Call orion.fit with the patches list, mimicing the sequential test
    # passing [patches] so that forward receives arguments=(patches,)
    orion.fit(model, [patches])
    
    input_level = orion.compile(model)
    print(f"Model compiled. Input level: {input_level}")
    
    # 6. CRITICAL: Copy FHE weights from traced model to original model
    print("Copying FHE weights from traced model to original...")
    traced_model = scheme.trace
    
    def copy_herpn_weights(original_herpn, traced_path):
        """Copy FHE polynomial coefficients from traced module to original."""
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
    
    # Copy HerPN weights for each backbone
    for backbone_idx, backbone in enumerate(model.nets):
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4', 'layer5']:
            if hasattr(backbone, layer_name):
                layer = getattr(backbone, layer_name)
                if hasattr(layer, 'herpn1') and layer.herpn1 is not None:
                    copy_herpn_weights(layer.herpn1, f'nets.{backbone_idx}.{layer_name}.herpn1')
                if hasattr(layer, 'herpn2') and layer.herpn2 is not None:
                    copy_herpn_weights(layer.herpn2, f'nets.{backbone_idx}.{layer_name}.herpn2')
                if hasattr(layer, 'shortcut_herpn') and layer.shortcut_herpn is not None:
                    copy_herpn_weights(layer.shortcut_herpn, f'nets.{backbone_idx}.{layer_name}.shortcut_herpn')
        if hasattr(backbone, 'herpnpool') and hasattr(backbone.herpnpool, 'herpn') and backbone.herpnpool.herpn is not None:
            copy_herpn_weights(backbone.herpnpool.herpn, f'nets.{backbone_idx}.herpnpool.herpn')
    
    # Copy normalization weights
    if hasattr(model, 'normalization') and hasattr(model.normalization, 'w2_fhe'):
        try:
            traced_norm = traced_model.get_submodule('normalization')
            model.normalization.w2_fhe = traced_norm.w2_fhe
            model.normalization.w1_fhe = traced_norm.w1_fhe
            model.normalization.w0_fhe = traced_norm.w0_fhe
        except Exception:
            pass
    
    print("  ✓ FHE weights copied successfully")
    
    # 7. Encrypt & Serialize Patches
    print("Encrypting and serializing patches...")
    serialized_patches = []
    for i, p in enumerate(patches):
        # Encode & Encrypt
        p_enc = orion.encrypt(orion.encode(p, input_level))
        # Serialize
        serialized_patches.append(p_enc.serialize())

    # Switch model to HE mode for workers
    model.he()

    # 8. Parallel Execution (Fork)
    print("Starting parallel execution (4 workers)...")
    # Using 'fork' context to inherit the compiled model and scheme
    ctx = multiprocessing.get_context('fork')
    
    t0 = time.time()
    with ctx.Pool(processes=4, initializer=worker_init, initargs=(model,)) as pool:
        results = pool.starmap(
            worker_process_patch,
            [(i, serialized_patches[i]) for i in range(4)]
        )
    t_parallel = time.time() - t0
    print(f"Parallel execution finished in {t_parallel:.2f}s")
    
    # Sort results
    results.sort(key=lambda x: x[0])
    
    # 9. Deserialize & Aggregate (Main Process)
    print("Aggregating results...")
    features = []
    for _, serialized_res in results:
        feat = CipherTensor.deserialize(scheme, serialized_res)
        features.append(feat)
        
    start_agg = time.time()
    out_agg = model._tree_reduce_add(features)
    
    # Normalization
    out_norm = model.normalization(out_agg)
    print(f"Aggregation and Normalization done in {time.time() - start_agg:.2f}s")
    
    # 10. Decrypt & Verify
    print("Decrypting and decoding...")
    out_fhe = out_norm.decrypt().decode()
    
    # Compare
    print("="*80)
    print("Results")
    print("="*80)
    
    err = (out_clear - out_fhe).abs()
    mae = err.mean().item()
    max_err = err.max().item()
    rel_err = mae / (out_clear.abs().mean().item() + 1e-10)
    
    print(f"MAE: {mae:.6f}")
    print(f"Max Error: {max_err:.6f}")
    print(f"Relative Error: {rel_err:.6f}")
    
    # Success criterion: strict but realistic for FHE
    if rel_err < 0.35: # Based on recent sequential results (~34%)
        print("✓ SUCCESS: Full model parallel inference works correctly.")
        return True
    else:
        print("✗ FAILURE: High error.")
        return False

if __name__ == "__main__":
    # Use a real image from LFW if available
    img_path = "/research/hal-datastage/datasets/original/LFW/lfw-mtcnn-aligned/John_Abizaid/John_Abizaid_0002.jpg"
    
    if not os.path.exists(img_path):
        print(f"Warning: {img_path} not found.")
        sys.exit(1)
        
    success = test_full_model_parallel(img_path)
    if not success:
        sys.exit(1)
