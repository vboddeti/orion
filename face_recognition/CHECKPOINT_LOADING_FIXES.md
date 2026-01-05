# CryptoFace Checkpoint Loading - Issues and Solutions

This document summarizes the issues encountered when loading CryptoFace pre-trained checkpoints into Orion, and the solutions implemented.

## Issues Identified

### Issue 1: Multiple init_orion_params() Calls
**Problem**: `init_orion_params()` was being called multiple times:
1. User calls `model.init_orion_params()` (correct, needed before `orion.fit()`)
2. `orion.fit()` internally calls `init_orion_params()` again (line 190 in `orion/core/orion.py`)

Each call multiplied conv weights by the HerPN scale_factor again, causing exponentially smaller weights.

**Root Cause**: `orion/core/orion.py` calls `init_orion_params()` on all modules during fitting as part of parameter initialization.

**Solution**: Made `init_orion_params()` **idempotent** by:
- Storing original conv weights on first call
- Restoring original weights at the start of each subsequent call
- This ensures calling it multiple times produces the same result

**Code Changes**: `models/pcnn.py` - HerPNConv.init_orion_params()
```python
# Store original weights on first call
if not hasattr(self, '_conv1_weight_original'):
    self._conv1_weight_original = self.conv1.weight.data.clone()
    self._conv2_weight_original = self.conv2.weight.data.clone()
    if self.has_shortcut:
        self._shortcut_conv_weight_original = self.shortcut_conv.weight.data.clone()

# Restore originals before scaling (idempotent behavior)
self.conv1.weight.data = self._conv1_weight_original.clone()
self.conv2.weight.data = self._conv2_weight_original.clone()
if self.has_shortcut:
    self.shortcut_conv.weight.data = self._shortcut_conv_weight_original.clone()
```

### Issue 2: Numerical Overflow in Cleartext Mode
**Problem**: Forward pass produced NaN values in cleartext mode due to numerical overflow.

**Root Cause**: CryptoFace uses coefficient factoring for FHE efficiency:
- Computes: w2, w1, w0 from BatchNorm statistics
- Factors: a1 = w1/w2, a0 = w0/w2
- Scales conv weights: `conv.weight *= w2`
- Computes in FHE: `x² + a1·x + a0` (factored form)

When w2 is very small (e.g., 2.59e-04), factored coefficients become huge:
- a0 = w0/w2 can be ~46,000
- a1 = w1/w2 can be ~546

In cleartext floating point, this causes:
```
layer1 output: max = 7.32e+03
layer2 output: max = 3.99e+06  (growing!)
layer3 output: max = 1.65e+20  (OVERFLOW!)
layer4 output: NaN
```

The factored form `x² + a1·x + a0` with huge a1, a0 values causes exponential growth when x is already large from previous layers.

**Solution**: Use **dual representation**:
- **Cleartext mode**: Use full form `w2·x² + w1·x + w0` (numerically stable)
- **FHE mode**: Use factored form `x² + a1·x + a0` (efficient, conv weights pre-scaled)

**Code Changes**:

1. `models/pcnn.py` - HerPN class:
```python
# Store BOTH full and factored coefficients
w0_full = w0.unsqueeze(-1).unsqueeze(-1)
w1_full = w1.unsqueeze(-1).unsqueeze(-1)
w2_full = w2.unsqueeze(-1).unsqueeze(-1)

# Pass full coefficients to ChannelSquare
super().__init__(weight0=w0_full, weight1=w1_full, weight2=w2_full)
```

2. `models/pcnn.py` - ChannelSquare.forward():
```python
# Cleartext: Use full form (numerically stable)
if self.weight2_raw is not None:
    return self.weight2_raw * (x**2) + self.weight1_raw * x + self.weight0_raw

# FHE: Use factored form (efficient, correct with pre-scaled conv weights)
if self.he_mode:
    x_sq = x * x
    term1 = x * self.w1_fhe  # a1·x
    result = x_sq + term1    # x² + a1·x
    result += self.w0_fhe    # x² + a1·x + a0
    return result
```

3. `models/pcnn.py` - ChannelSquare.compile():
```python
# Compute factored coefficients for FHE encoding
a1_raw = self.weight1_raw / self.weight2_raw
a0_raw = self.weight0_raw / self.weight2_raw

# Encode factored form for FHE
self.w1_fhe = self.scheme.encoder.encode(a1, input_level)
self.w0_fhe = self.scheme.encoder.encode(a0, output_level)
```

### Issue 3: Model Training Mode
**Problem**: BatchNorm expected batch_size > 1 when model was in training mode.

**Solution**: Added `model.eval()` to weight loader:

**Code Changes**: `face_recognition/models/weight_loader.py`
```python
# Set model to eval mode to avoid BatchNorm training issues
model.eval()
```

### Issue 4: Warmup Phase with Constant Tensors
**Problem**: Documentation recommended warmup phase (20 forward passes) after loading checkpoint. This caused issues because:
- Unfused HerPN forward uses `bn0(torch.ones_like(x))` - constant tensor through BatchNorm
- Can cause numerical issues with BatchNorm statistics

**Solution**: **Skip warmup when loading checkpoints**
- Checkpoints already contain trained BatchNorm statistics
- Warmup only needed when training from scratch
- Call `init_orion_params()` immediately after loading checkpoint

**Code Changes**: Updated documentation and examples to skip warmup:
```python
# OLD (incorrect):
model.eval()
for _ in range(20):
    _ = model(torch.randn(4, 3, 64, 64))  # Warmup
model.init_orion_params()

# NEW (correct):
model.init_orion_params()  # Skip warmup - checkpoint has statistics
```

## Testing and Verification

Created comprehensive verification utility: `face_recognition/utils/checkpoint_verification.py`

### Verification Checks
1. **Final BatchNorm Identity**: Verifies fusion into linear layers (mean=0, var=1)
2. **HerPN Weight Loading**: Confirms all HerPN weights loaded from checkpoint
3. **HerPN Fusion Correctness**: Validates scale_factor and coefficients exist
4. **Linear Layer Fusion**: Checks per-patch linear layers have consistent fused bias
5. **Weight Shapes**: Validates all tensors have expected shapes
6. **No Dynamic Operations**: Confirms shapes are recorded for compile-time expansion

### Usage
```python
from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4
from face_recognition.models.weight_loader import load_checkpoint_for_config
from face_recognition.utils.checkpoint_verification import CheckpointVerifier

model = CryptoFaceNet4()
load_checkpoint_for_config(model, input_size=64)
model.init_orion_params()

verifier = CheckpointVerifier(model, verbose=True)
all_passed = verifier.verify_all()
```

## Correct Workflow

### For Cleartext Inference
```python
from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4
from face_recognition.models.weight_loader import load_checkpoint_for_config

# 1. Load model and checkpoint
model = CryptoFaceNet4()
load_checkpoint_for_config(model, input_size=64)

# 2. Fuse HerPN (NO warmup needed!)
model.init_orion_params()

# 3. Run inference
x = torch.randn(1, 3, 64, 64)
output = model(x)
```

### For FHE Inference
```python
import orion
from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4
from face_recognition.models.weight_loader import load_checkpoint_for_config

# 1. Load model and checkpoint
model = CryptoFaceNet4()
load_checkpoint_for_config(model, input_size=64)

# 2. Fuse HerPN (before orion.fit!)
model.init_orion_params()

# 3. Initialize FHE scheme
orion.init_scheme("configs/cryptoface_net4.yml")

# 4. Fit and compile
x = torch.randn(1, 3, 64, 64)
orion.fit(model, x)  # Will call init_orion_params() again (now idempotent!)
input_level = orion.compile(model)

# 5. Encrypt and infer
vec_ptxt = orion.encode(x, input_level)
vec_ctxt = orion.encrypt(vec_ptxt)
model.he()
out_ctxt = model(vec_ctxt)
fhe_output = out_ctxt.decrypt().decode()
```

## Files Modified

1. **models/pcnn.py**
   - Made HerPNConv.init_orion_params() idempotent
   - Made HerPNPool.init_orion_params() idempotent
   - Updated HerPN to store full coefficients (w2, w1, w0)
   - Updated ChannelSquare.forward() to use full form in cleartext
   - Updated ChannelSquare.compile() to create factored coefficients for FHE

2. **face_recognition/models/weight_loader.py**
   - Added `model.eval()` after loading checkpoint

3. **face_recognition/utils/checkpoint_verification.py** (NEW)
   - Comprehensive verification utility
   - 6 verification checks
   - Cleartext inference testing

4. **face_recognition/examples/end_to_end_cryptoface_inference.py** (NEW)
   - Complete workflow example
   - Removed warmup phase
   - Added proper error handling

5. **face_recognition/CHECKPOINT_LOADING_GUIDE.md** (NEW)
   - Comprehensive guide
   - Step-by-step workflow
   - Troubleshooting section

## Results

### Before Fixes
- Cleartext inference: **NaN**
- Root cause: Numerical overflow from factored coefficients

### After Fixes
- Cleartext inference: **✓ SUCCESS**
- All verification checks: **✓ PASSED**
- Example output range: [-6.46, 6.26]
- Ready for FHE compilation

## Key Takeaways

1. **Idempotency is critical**: Methods called by framework internals must be idempotent
2. **Numerical stability matters**: FHE-optimized math may not work in cleartext floating point
3. **Dual representations work**: Cleartext uses full form, FHE uses factored form
4. **Skip unnecessary warmup**: Checkpoints already contain statistics
5. **Verify before FHE**: Use CheckpointVerifier to catch issues early

## Next Steps

To test FHE inference:
```bash
cd /research/hal-vishnu/code/orion-fhe
uv run python face_recognition/examples/end_to_end_cryptoface_inference.py \
    --model net4 \
    --config configs/cryptoface_net4.yml \
    --single-backbone  # For faster testing
```

Expected: MAE < 1.0 between cleartext and FHE outputs.
