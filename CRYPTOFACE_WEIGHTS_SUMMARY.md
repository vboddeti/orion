# CryptoFace Model Weights Summary

## Checkpoint Structure

**File**: `face_recognition/checkpoints/backbone-64x64.ckpt`

### Top-Level Keys (Final Aggregation Layer)

```python
# Linear layer (aggregates 4 patches: 1024 → 256)
linear.weight: [256, 1024]
linear.bias:   [256]

# Final BatchNorm (applied after linear)
bn.running_mean: [256]  # mean=0.26, std=4.25
bn.running_var:  [256]  # mean=400.9, std=17.7
bn.num_batches_tracked: []

# Note: NO bn.weight or bn.bias (affine=False)
# BatchNorm only does: (x - mean) / sqrt(var + eps)
```

### Per-Patch Backbone Weights

For each patch `i` (0, 1, 2, 3):

```python
nets.{i}.conv.weight: [16, 3, 3, 3]
nets.{i}.layers.{j}.conv1.weight
# ... (standard backbone weights)
nets.{i}.bn.running_mean: [256]
nets.{i}.bn.running_var:  [256]
```

## L2 Normalization Coefficients (Updated)

We use custom coefficients fitted for the **aggregated feature range** ($y \in [100, 3000]$), rather than the original checkpoint coefficients which assumed normalized inputs.

**Original (Checkpoint) Coefficients for Normalized Range [0, 2]**:
```
a = 2.409946e-07
b = -2.440841e-04
c = 1.092854e-01
```

**New Coefficients for Aggregated Range [100, 3000]**:
```
a = 1.057367e-08
b = -4.753845e-05
c = 7.176193e-02
```
*These are hardcoded in `create_cryptoface_pcnn` / `CryptoFaceNet4` initialization.*

## How Weights Are Loaded

### 1. Backbone Weights (`_load_backbone_weights`)

**Location**: `face_recognition/models/weight_loader.py:182-310`

- Loads weights for each of the 4 patch backbones
- Remaps CryptoFace naming to Orion naming.

### 2. Linear + BatchNorm Fusion (`_load_linear_and_bn_weights`)

**Location**: `face_recognition/models/weight_loader.py:57-180`

**Critical**: BatchNorm is fused INTO the linear layers before inference!

```python
# Fused weight = weight / sqrt(var + eps)
# Fused bias = (bias - mean) / sqrt(var + eps)
```

**Why Fusion?**
- Matches SEAL C++ implementation behavior
- Reduces one level of computation in FHE
- BatchNorm becomes a no-op (identity transform) after fusion

### 3. L2 Normalization Coefficients

**Note**: We do **NOT** use `load_l2_coeffs` for the full model anymore. We use the updated coefficients hardcoded in the model definition to ensuring scaling correctness.
