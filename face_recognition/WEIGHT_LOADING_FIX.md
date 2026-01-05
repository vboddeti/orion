# CryptoFace Weight Loading Fix

## Summary

Fixed the weight loading implementation in `face_recognition/models/weight_loader.py` to correctly fuse BatchNorm statistics into linear layer weights **before** chunking, matching the CryptoFace reference implementation.

## The Problem

The original implementation was:
1. ❌ Chunk linear weights first
2. ❌ Divide bias by N
3. ❌ Load BatchNorm separately (not fused)

This resulted in incorrect weights being loaded into the per-patch linear layers.

## The Solution

The corrected implementation now follows CryptoFace's `fuse()` method (`CryptoFace/models/pcnn.py:73-86`):

1. ✅ **Fuse BatchNorm into Linear layer FIRST**
   - Normalize weight: `weight_fused = weight / sqrt(var + eps)`
   - Normalize bias: `bias_fused = (bias - mean) / sqrt(var + eps)`

2. ✅ **Then divide bias by N** (number of patches)
   - `bias_fused = bias_fused / N`

3. ✅ **Then chunk the fused weights**
   - `chunked_weights = torch.chunk(weight_fused, N, dim=1)`

4. ✅ **Set final BatchNorm to identity**
   - `running_mean = 0`, `running_var = 1`
   - Since BatchNorm is already fused into linear layers

## Mathematical Verification

The fusion is mathematically equivalent to:
```
Unfused:  output = BatchNorm(Linear(x))
          output = (Wx + b - mean) / sqrt(var + eps)

Fused:    output = Linear_fused(x)
          output = (W / sqrt(var + eps)) @ x + (b - mean) / sqrt(var + eps)
```

These produce identical outputs (verified with max diff < 1e-5).

## Code Changes

### File: `face_recognition/models/weight_loader.py`

**Key changes in `load_cryptoface_checkpoint()` function:**

```python
# Step 2: Fuse BatchNorm into Linear layer (lines 80-125)
# Get all parameters
full_weight = state_dict['linear.weight']  # [256, N*256]
full_bias = state_dict['linear.bias']      # [256]
bn_mean = state_dict['bn.running_mean']    # [256]
bn_var = state_dict['bn.running_var']      # [256]
bn_eps = model.normalization.eps

# Fuse BatchNorm (following CryptoFace pcnn.py:78-83)
weight_fused = torch.divide(full_weight.T, torch.sqrt(bn_var + bn_eps))
weight_fused = weight_fused.T
bias_fused = torch.divide(full_bias - bn_mean, torch.sqrt(bn_var + bn_eps))

# Step 3: Chunk and divide bias (lines 127-154)
bias_fused = bias_fused / N
chunked_weights = torch.chunk(weight_fused, N, dim=1)

for i in range(N):
    model.linear[i].weight.data = chunked_weights[i].clone()
    model.linear[i].bias.data = bias_fused.clone()

# Step 4: Set BatchNorm to identity (lines 156-170)
model.normalization.running_mean.data = torch.zeros_like(bn_mean)
model.normalization.running_var.data = torch.ones_like(bn_var)
```

## Test Results

All verification tests pass:

### 1. Weight Loading Test (`test_weight_loading_detailed.py`)
- ✓ Backbone BatchNorm statistics loaded correctly
- ✓ Final BatchNorm set to identity

### 2. Fusion Correctness Test (`test_fusion_correctness.py`)
- ✓ Final BatchNorm is identity (mean=0, var=1)
- ✓ Fused bias consistent across all patches
- ✓ Forward pass produces valid outputs

### 3. Determinism Test (`test_determinism.py`)
- ✓ Model is deterministic (same input → same output)
- ✓ Model is sensitive to different inputs
- ✓ Backbone BatchNorm statistics loaded

### 4. Mathematical Equivalence Test (`test_fusion_math.py`)
- ✓ Fused weights match manual computation
- ✓ Forward pass equivalence (max diff < 1e-5)
- ✓ **BatchNorm fusion is MATHEMATICALLY CORRECT**

## Reference

The implementation follows the CryptoFace reference code:
- `CryptoFace/models/pcnn.py` - `PatchCNN.fuse()` method (lines 73-86)
- `CryptoFace/helper.py` - `model2txt()` function for weight export

## Notes

- The special weight transformation in `helper.py:82-84` (view/transpose/flatten) is for CKKS packing optimization in the SEAL implementation, NOT for PyTorch weight loading
- All per-patch linear layers receive the same fused bias (`bias_fused / N`)
- When summing outputs from N patches, we get: `N × (bias_fused / N) = bias_fused` (correct total)
