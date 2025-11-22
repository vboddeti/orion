# Parallel FHE Inference Implementation for PCNN

## Overview

This document describes the implementation of parallel FHE inference for PatchCNN using fork-based multiprocessing with ciphertext serialization.

## Problem Statement

PCNN (Patch-based CNN) divides an input image into 4 patches and processes each through a separate backbone network. Sequential processing is slow:
- **Sequential baseline**: 786.42 seconds (~13.1 minutes)
- Goal: Parallelize the 4 independent backbone computations to reduce latency

## Challenges

1. **Model Compilation**: Orion's FHE compilation is expensive (~40-50 seconds per backbone)
2. **Encrypted Data Transfer**: Need to pass encrypted data between processes
3. **End-to-end Encryption**: Must maintain encryption throughout (no intermediate decrypt/re-encrypt)
4. **Process Memory**: Avoid duplicating large compiled models

## Solution: Option D (Fork + Serialization)

### Key Idea

**Compile once, fork to inherit, serialize ciphertexts for data transfer**

1. **Main Process**:
   - Compile the model ONCE (~50 seconds)
   - Call `model.he()` to switch to HE mode
   - Extract and encrypt patches
   - Serialize encrypted patches (CipherTensor → bytes)

2. **Fork Workers** (using `multiprocessing.set_start_method('fork')`):
   - Workers inherit the compiled model from parent memory (copy-on-write)
   - No recompilation needed!
   - Receive serialized ciphertext via function arguments
   - Deserialize to CipherTensor
   - Process through backbone (FHE operations via Lattigo)
   - Serialize output ciphertext
   - Return serialized result

3. **Main Process (aggregation)**:
   - Deserialize encrypted backbone outputs
   - Process through linear layers (encrypted)
   - Aggregate results with tree reduction (encrypted)
   - Apply normalization (encrypted)
   - Decrypt ONLY the final result

### Implementation Pattern

```python
import multiprocessing

# Global variable for fork to inherit
_global_model = None

def worker_process_patch_fork(patch_idx, serialized_patch):
    global _global_model

    # Deserialize encrypted patch (scheme inherited via fork)
    from orion.core import scheme
    from orion.backend.python.tensors import CipherTensor
    patch_ctxt = CipherTensor.deserialize(scheme, serialized_patch)

    # Process through backbone (encrypted, using inherited compiled model)
    y_i = _global_model.nets[patch_idx](patch_ctxt)

    # Serialize and return encrypted result
    return patch_idx, y_i.serialize()

def test_pcnn_fork():
    global _global_model

    # 1. Setup and compilation (ONCE)
    model = PatchCNN(...)
    model.eval()
    model.init_orion_params()  # Fuse HerPN
    orion.fit(model, inp)
    input_level = orion.compile(model)

    # 2. Switch to HE mode (CRITICAL!)
    model.he()
    _global_model = model

    # 3. Encrypt and serialize patches
    encrypted_patches = [orion.encrypt(orion.encode(p, input_level)) for p in patches]
    serialized_patches = [p.serialize() for p in encrypted_patches]

    # 4. Fork workers (inherit compiled model)
    multiprocessing.set_start_method('fork', force=True)
    with multiprocessing.Pool(processes=4) as pool:
        results = pool.starmap(worker_process_patch_fork,
                              [(i, serialized_patches[i]) for i in range(4)])

    # 5. Deserialize encrypted results
    encrypted_outputs = [CipherTensor.deserialize(scheme, r[1]) for r in results]

    # 6. Continue with encrypted processing (linear, aggregate, normalize)
    # 7. Decrypt only at the end
```

## Critical Bug Fix: @timer Decorator on Flatten

### The Problem

Initial implementation crashed with:
```
RuntimeError: Cannot re-initialize CUDA in forked subprocess
```

### Root Cause

The `Flatten` module in `orion/nn/reshape.py` was missing the `@timer` decorator. This decorator includes `@torch.compiler.disable`, which prevents PyTorch from trying to compile the module.

Without this decorator:
- PyTorch's compiler tried to optimize Flatten.forward()
- During compilation tracing, it checked CUDA availability
- CUDA cannot re-initialize in forked subprocesses → crash

### The Fix

Added `@timer` decorator to Flatten.forward():

```python
# orion/nn/reshape.py
from .module import Module, timer  # Import timer

class Flatten(Module):
    @timer  # <-- Added this decorator
    def forward(self, x):
        if self.he_mode:
            return x  # No-op in HE mode
        return torch.flatten(x, start_dim=1)
```

### Why This Matters

- Orion uses **Lattigo (Go backend, CPU-only)** for FHE operations
- PyTorch should NOT be involved in FHE inference
- The `@timer` decorator (which includes `@torch.compiler.disable`) ensures PyTorch's compiler doesn't interfere
- All other Orion modules (Conv2d, HerPN, etc.) already had this decorator
- Flatten was the only one missing it

## Performance Results

| Implementation | Time (seconds) | Speedup vs Sequential |
|---|---|---|
| **Sequential baseline** | 786.42 | 1.0x |
| **Fork + Serialization (Option D)** | 291.36 | **2.7x** ✅ |
| **Old parallel (recompile in workers)** | 772.12 | 1.02x (worse!) ❌ |

### Key Insights

1. **Option D achieves 2.7x speedup** - Significant improvement
2. **Old approach was slower than sequential** - Redundant compilation in each worker wasted ~160-200s
3. **Compilation overhead is critical** - Compile once and share via fork is essential
4. **End-to-end encryption maintained** - No security compromise

## File Organization

### Active Tests (tests/models/)
- `test_pcnn_fork_serialization.py` - Main parallel implementation (Option D)
- `test_pcnn_sequential.py` - Sequential baseline for comparison
- `test_pcnn_backbone.py` - Single backbone test
- `test_parallel_aggregation.py` - Simple 4-branch parallel + aggregation validation
- `test_serialization.py` - Ciphertext serialization validation
- `test_simple_compile.py` - Basic compilation sanity check

### Archived Tests (tests/models/archive/)
- `test_pcnn.py` - Old inefficient parallel approach (recompiled in each worker)
- Other experimental tests from development

### Log Files (logs/)
- `test_pcnn_fork_serialization.log` - Option D results (291.36s)
- `test_pcnn_sequential.log` - Sequential baseline (786.42s)
- `test_pcnn_backbone.log` - Single backbone (165.82s)
- `test_parallel_aggregation.log` - Validation test
- `test_serialization_validation.log` - Serialization test

## Key Takeaways

1. **Fork is essential** for avoiding recompilation overhead
2. **Serialization works perfectly** for encrypted data transfer
3. **@timer decorator is critical** for all Orion modules to prevent PyTorch compilation interference
4. **FHE operations are CPU-only** via Lattigo backend - no CUDA/GPU involvement
5. **End-to-end encryption is maintained** - decrypt only at the final step
6. **2.7x speedup achieved** with clean, maintainable code

## Future Work

- Extend to more patches (8x8 grid instead of 2x2)
- Optimize serialization overhead
- Explore GPU-accelerated FHE backends (when available)
- Apply same pattern to other models with parallel components
