# CryptoFace Checkpoint Loading Guide

This guide explains how to properly load CryptoFace pre-trained checkpoints into Orion for both cleartext and FHE inference.

## Table of Contents

1. [Overview](#overview)
2. [The Two Key Issues](#the-two-key-issues)
3. [Solution Architecture](#solution-architecture)
4. [Step-by-Step Workflow](#step-by-step-workflow)
5. [Implementation Details](#implementation-details)
6. [Testing and Verification](#testing-and-verification)
7. [Troubleshooting](#troubleshooting)

---

## Overview

CryptoFace checkpoints contain PyTorch models trained with BatchNorm layers and dynamic operations that need special handling for FHE inference:

- **BatchNorm layers** must be fused into preceding layers (convs, linear) because FHE doesn't support dynamic statistics
- **Dynamic tensor operations** (reshape, expand, broadcast) must be resolved at compile time, not runtime

This guide shows how the Orion framework handles both issues automatically.

---

## The Two Key Issues

### Issue 1: BatchNorm Fusion

**Problem**: CryptoFace checkpoints contain BatchNorm layers with running statistics (mean, var) that need to be fused into weights.

**Where it occurs**:
1. Final `BatchNorm1d` after linear layers
2. HerPN layers with 3 BatchNorms each (bn0, bn1, bn2)
3. Shortcut BatchNorms in residual connections

**Why it's needed for FHE**:
- FHE cannot compute running statistics dynamically
- BatchNorm operations must be "baked into" the weights before encryption

**Solution**:
- **Final BatchNorm**: Fused into linear layer weights during checkpoint loading
- **HerPN BatchNorms**: Fused into quadratic activation during `init_orion_params()`
- **Shortcut BatchNorms**: Loaded as regular BatchNorms (will be fused by Orion's auto-fuser)

### Issue 2: Dynamic Tensor Operations

**Problem**: PyTorch models use dynamic broadcasting/reshaping (e.g., `[C] * [B,C,H,W]`), but FHE requires all shapes to be known at compile time.

**Where it occurs**:
1. HerPN coefficients: `a0`, `a1` are `[C,1,1]` but need to broadcast to `[B,C,H,W]`
2. Scale factors: `w2` is `[C,1,1]` but needs to multiply `[B,C,H,W]` tensors

**Why it's needed for FHE**:
- CKKS operates on fixed-size polynomial slots
- All tensor shapes must be known before encryption
- Broadcasting must be explicit, not implicit

**Solution**:
- Use `ScaleModule` and `ChannelSquare` that record actual input shapes during warmup
- During compilation, expand all weights to match recorded shapes
- Store expanded weights as encoded plaintexts for FHE operations

---

## Solution Architecture

The solution has three main components:

### 1. Weight Loader (`face_recognition/models/weight_loader.py`)

**Responsibilities**:
- Load CryptoFace checkpoint `.ckpt` files
- Remap CryptoFace naming → Orion naming
- Fuse final BatchNorm into linear layers
- Chunk linear weights per-patch
- Store HerPN weight/bias as attributes for later use

**Key Function**: `load_cryptoface_checkpoint(model, checkpoint_path)`

### 2. HerPN Fusion (`models/pcnn.py` - `init_orion_params()`)

**Responsibilities**:
- Fuse 3 BatchNorms per HerPN into quadratic coefficients
- Factor coefficients for CryptoFace-style optimization: `a1 = w1/w2`, `a0 = w0/w2`
- Absorb `w2` scale factor into conv weights
- Create `ScaleModule` for shortcut scaling

**Key Method**: `HerPNConv.init_orion_params()`, `HerPNPool.init_orion_params()`

### 3. Compile-Time Expansion (`models/pcnn.py` - `compile()`)

**Responsibilities**:
- Expand `[C,1,1]` coefficients to `[B,C,H,W]` using recorded shapes
- Encode expanded tensors as FHE plaintexts at correct levels
- Prepare all FHE parameters before encryption

**Key Method**: `ChannelSquare.compile()`, `ScaleModule.compile()`

---

## Step-by-Step Workflow

### Step 1: Load Model and Checkpoint

```python
from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4
from face_recognition.models.weight_loader import load_checkpoint_for_config

# Create model
model = CryptoFaceNet4()  # 64×64 input, 4 patches

# Load checkpoint
load_checkpoint_for_config(model, input_size=64, verbose=True)
```

**What happens**:
- Loads `backbone-64x64.ckpt`
- Remaps keys: `nets.0.layers.0.herpn1.bn0` → `nets.0.layer1.bn0_1`
- Fuses final BatchNorm: `weight_fused = weight / sqrt(var + eps)`, `bias_fused = (bias - mean) / sqrt(var + eps)`
- Chunks weights: `[256, 1024] → 4×[256, 256]`
- Stores HerPN weights: `layer.herpn1_weight`, `layer.herpn1_bias`

**After this step**:
- ✓ All weights loaded
- ✓ Final BatchNorm fused
- ✗ HerPN BatchNorms NOT yet fused (need statistics first)

### Step 2: Collect BatchNorm Statistics (Warmup)

```python
model.eval()
with torch.no_grad():
    for _ in range(20):
        _ = model(torch.randn(4, 3, 64, 64))
```

**What happens**:
- Runs forward passes through unfused model
- BatchNorm layers update `running_mean` and `running_var`
- Statistics converge to representative values

**Why needed**:
- Loaded checkpoints may not have recent statistics
- Ensures fusion uses correct normalization parameters

**After this step**:
- ✓ BatchNorm statistics collected
- ✗ Still using unfused forward pass

### Step 3: Fuse HerPN Operations

```python
model.init_orion_params()
```

**What happens**:
- For each HerPNConv layer:
  - Computes HerPN coefficients from BatchNorm statistics and loaded weights
  - Factors: `a2 = w2`, `a1 = w1/w2`, `a0 = w0/w2`
  - Absorbs `w2` into conv weights: `conv.weight *= w2`
  - Creates `ScaleModule` for shortcut: `shortcut_scale = ScaleModule(w2)`
- For HerPNPool:
  - Same HerPN fusion process
  - Creates `pool_scale = ScaleModule(w2)` for post-pooling scaling

**After this step**:
- ✓ All BatchNorms fused into HerPN quadratic activations
- ✓ Conv weights scaled by `w2`
- ✓ Model now uses fused forward pass
- ✗ Weights still in `[C,1,1]` format (not expanded for FHE yet)

**CRITICAL**: This must happen BEFORE `orion.fit()` for correct level assignment!

### Step 4: Verify Fusion Correctness

```python
from face_recognition.utils.checkpoint_verification import CheckpointVerifier

verifier = CheckpointVerifier(model, verbose=True)
all_passed = verifier.verify_all()
```

**What happens**:
- Checks final BatchNorm is identity (mean=0, var=1)
- Verifies HerPN weights loaded correctly
- Confirms linear layer fusion
- Validates weight shapes
- Ensures no dynamic operations remain

**After this step**:
- ✓ Verified model is correctly fused and ready for FHE

### Step 5: Cleartext Inference

```python
model.eval()
with torch.no_grad():
    cleartext_output = model(test_input)
```

**What happens**:
- Runs through fused model in cleartext
- Records actual tensor shapes in `_actual_fhe_input_shape`
- These shapes will be used during compilation

**After this step**:
- ✓ Cleartext baseline established
- ✓ All shapes recorded for compilation

### Step 6: FHE Compilation

```python
import orion

orion.init_scheme("configs/cryptoface_net4.yml")
orion.fit(model, test_input)
input_level = orion.compile(model)
```

**What happens during `orion.fit()`**:
- Traces model to build computation graph
- Identifies all operations and dependencies
- Assigns levels based on multiplicative depth

**What happens during `orion.compile()`**:
- Places bootstraps optimally
- Assigns final input/output levels
- Calls `module.compile()` on all modules

**What happens in `ChannelSquare.compile()` and `ScaleModule.compile()`**:
- Retrieves recorded shape: `_actual_fhe_input_shape`
- Expands `[C,1,1]` weights to full shape `[B,C,H,W]`
- Encodes expanded tensors as plaintexts at correct levels
- Stores as `w0_fhe`, `w1_fhe`, `scale_fhe`, etc.

**After this step**:
- ✓ Model compiled for FHE
- ✓ All weights expanded and encoded
- ✓ Ready for encryption

### Step 7: FHE Inference

```python
vec_ptxt = orion.encode(test_input, input_level)
vec_ctxt = orion.encrypt(vec_ptxt)

model.he()  # Switch to FHE mode
out_ctxt = model(vec_ctxt)

fhe_output = out_ctxt.decrypt().decode()
```

**What happens**:
- Input encrypted
- Forward pass uses pre-encoded plaintexts (no dynamic operations!)
- All operations are homomorphic (Add, Mul, Conv, Bootstrap)
- Output decrypted and decoded

**After this step**:
- ✓ FHE inference complete
- ✓ Can compare with cleartext output

### Step 8: Validation

```python
mae = (cleartext_output - fhe_output).abs().mean()
assert mae < 1.0, f"FHE inference failed: MAE = {mae}"
```

**Success criteria**:
- MAE (Mean Absolute Error) < 1.0
- No NaN or Inf values
- Output shape matches cleartext

---

## Implementation Details

### HerPN Fusion Mathematics

HerPN (Hermite Polynomial Network) activation computes:
```
output = w2·x² + w1·x + w0
```

Where `w0`, `w1`, `w2` are computed from 3 BatchNorm layers:

```python
# From BatchNorm statistics
m0, v0 = bn0.running_mean, bn0.running_var
m1, v1 = bn1.running_mean, bn1.running_var
m2, v2 = bn2.running_mean, bn2.running_var
g, b = herpn.weight, herpn.bias
eps = bn0.eps

# Compute full coefficients
w2 = g / sqrt(8π(v2 + eps))
w1 = g / (2√(v1 + eps))
w0 = b + g(...)  # Complex formula

# CryptoFace optimization: Factor out w2
a2 = w2
a1 = w1 / w2
a0 = w0 / w2

# Now compute: x² + a1·x + a0 (factored form)
# Then scale output by a2 (absorbed into next conv)
```

**Why factor?**:
- Reduces FHE multiplications by 1 per HerPN
- Output of `x² + a1·x + a0` is multiplied by next conv weight
- So we pre-multiply: `conv_weight_new = conv_weight_old * a2`
- This makes `conv(a2·(x² + a1·x + a0)) = conv_scaled(x² + a1·x + a0)`

### Shortcut Scaling

In residual connections, the shortcut must also be scaled by `a2`:

```python
# Main path
x_herpn = x² + a1·x + a0  # HerPN factored
out = conv_scaled(x_herpn)

# Shortcut path (even for identity shortcuts!)
shortcut = shortcut_conv(a2 · x_herpn)

# Addition
out = out + shortcut
```

This is why we use `ScaleModule(a2)` for shortcuts.

### Dynamic Operation Resolution

**Problem**: FHE needs fixed shapes, but PyTorch broadcasts dynamically.

**Example**:
```python
# PyTorch (cleartext)
a1 = torch.randn(16, 1, 1)  # [16, 1, 1]
x = torch.randn(2, 16, 8, 8)  # [2, 16, 8, 8]
out = a1 * x  # Broadcasting: [16,1,1] → [2,16,8,8]
```

**FHE Problem**:
- CKKS packs `[2,16,8,8]` into polynomial slots
- `a1` is `[16,1,1]` - different shape!
- Cannot broadcast at runtime

**Solution**:
```python
# Step 1: Record shape during warmup (cleartext)
def forward(self, x):
    if not self.he_mode and not hasattr(self, '_actual_fhe_input_shape'):
        self._actual_fhe_input_shape = x.shape  # Record [2,16,8,8]

    # Normal forward...

# Step 2: Expand during compilation
def compile(self):
    target_shape = self._actual_fhe_input_shape  # [2,16,8,8]
    a1_expanded = self.a1.expand(target_shape)  # [16,1,1] → [2,16,8,8]
    self.a1_fhe = self.scheme.encoder.encode(a1_expanded, self.level)

# Step 3: Use pre-expanded in FHE mode
def forward(self, x):
    if self.he_mode:
        out = self.a1_fhe * x  # Both [2,16,8,8] - no broadcasting!
    else:
        out = self.a1 * x  # [16,1,1] * [2,16,8,8] - broadcasts
```

---

## Testing and Verification

### Quick Test (Single Backbone)

```bash
cd /research/hal-vishnu/code/orion-fhe
uv run python face_recognition/examples/end_to_end_cryptoface_inference.py \
    --model net4 \
    --config configs/cryptoface_net4.yml \
    --single-backbone
```

**Time**: ~2-5 minutes
**Tests**: Single patch backbone (32×32 input)

### Full Test (All Patches)

```bash
uv run python face_recognition/examples/end_to_end_cryptoface_inference.py \
    --model net4 \
    --config configs/cryptoface_net4.yml
```

**Time**: ~15-30 minutes (4 patches in parallel)
**Tests**: Full CryptoFaceNet4 model

### Verification Only

```python
from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4
from face_recognition.models.weight_loader import load_checkpoint_for_config
from face_recognition.utils.checkpoint_verification import CheckpointVerifier

model = CryptoFaceNet4()
load_checkpoint_for_config(model, input_size=64)

# Warmup
model.eval()
for _ in range(20):
    _ = model(torch.randn(4, 3, 64, 64))

# Fuse
model.init_orion_params()

# Verify
verifier = CheckpointVerifier(model, verbose=True)
verifier.verify_all()
```

**Time**: <1 minute
**Tests**: All fusion and weight loading (no FHE)

---

## Troubleshooting

### Issue: "Final BatchNorm is not identity"

**Symptom**:
```
✗ FAILED: Final BatchNorm Identity
  running_mean max abs: 2.3451
  running_var deviation from 1.0: 0.8923
```

**Cause**: Checkpoint loading didn't set BatchNorm to identity.

**Fix**: Check that `weight_loader.py` sets:
```python
model.normalization.running_mean.data = torch.zeros_like(bn_mean)
model.normalization.running_var.data = torch.ones_like(bn_var)
```

### Issue: "HerPN weights not loaded"

**Symptom**:
```
✗ FAILED: HerPN Weight Loading
  layer1: ✗ HerPN weights not loaded
```

**Cause**: Weight loader didn't extract and store HerPN weights.

**Fix**: Verify checkpoint contains:
```python
ckpt['backbone']['nets.0.layers.0.herpn1.weight']
ckpt['backbone']['nets.0.layers.0.herpn1.bias']
```

And weight loader sets:
```python
setattr(layer_module, 'herpn1_weight', value)
setattr(layer_module, 'herpn1_bias', value)
```

### Issue: "MAE > 1.0 in FHE inference"

**Symptom**:
```
✗ FHE INFERENCE FAILED
  MAE = 15.234
```

**Possible causes**:
1. **Incorrect fusion**: `init_orion_params()` not called before `orion.fit()`
   - Fix: Always call `init_orion_params()` after warmup, before fitting

2. **Wrong CKKS parameters**: Insufficient precision or depth
   - Fix: Increase `LogScale` or add more `LogQ` entries

3. **Bootstrap placement**: Bootstraps at levels with low Q/Scale ratio
   - Fix: Increase `bootstrap_placement_margin` in config

4. **Shape mismatch**: Dynamic operations not properly resolved
   - Fix: Check that all modules record `_actual_fhe_input_shape`

### Issue: "Shape mismatch during compilation"

**Symptom**:
```
RuntimeError: shape '[16, 1, 1]' is invalid for input of size 2048
```

**Cause**: Module trying to expand weights without recorded shape.

**Fix**: Ensure warmup forward pass happens before compilation:
```python
model.eval()
with torch.no_grad():
    _ = model(test_input)  # Records shapes

# Then compile
orion.fit(model, test_input)
```

### Issue: "Level assignment errors"

**Symptom**:
```
panic: level cannot be larger than max level
```

**Cause**: Called `orion.fit()` before `init_orion_params()`, so tracer captured unfused graph with extra depth.

**Fix**: Always use this order:
```python
# 1. Warmup
for _ in range(20):
    _ = model(data)

# 2. Fuse (BEFORE fit!)
model.init_orion_params()

# 3. Fit
orion.fit(model, data)

# 4. Compile
orion.compile(model)
```

---

## Summary

**Key Takeaways**:

1. **BatchNorm fusion happens in two places**:
   - Final BatchNorm: During checkpoint loading
   - HerPN BatchNorms: During `init_orion_params()`

2. **Dynamic operations are resolved at compile time**:
   - Shapes recorded during warmup
   - Weights expanded during compilation
   - FHE forward uses pre-expanded plaintexts

3. **Correct order is critical**:
   ```
   Load checkpoint → Warmup → Fuse → Verify → Fit → Compile → Encrypt → Infer
   ```

4. **Always verify before FHE**:
   ```python
   verifier = CheckpointVerifier(model)
   assert verifier.verify_all(), "Fusion failed!"
   ```

5. **MAE < 1.0 is success criterion**:
   - Typical MAE: 0.01 - 0.5
   - If MAE > 1.0, check fusion and CKKS parameters

For more details, see:
- `docs/CUSTOM_FIT_WORKFLOW.md` - Why init_orion_params() before fit()
- `BOOTSTRAP_PLACEMENT_MARGIN_EXPLANATION.md` - Bootstrap placement tuning
- `face_recognition/models/weight_loader.py` - Checkpoint loading implementation
- `models/pcnn.py` - HerPN fusion implementation
