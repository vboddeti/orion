# PCNN Level Consumption Issue and Fix

## Problem Summary

When compiling PCNN models with HerPN activations using Orion, we observed an unexpected level consumption pattern:

```
conv @ level=18
layer1_herpn1 @ level=7  ← 11-level gap! Should be ~16-17
```

The 11-level gap between consecutive layers indicated something was consuming far more multiplicative depth than expected.

## Root Cause

The issue was in **when** HerPN fusion occurs relative to Orion's tracing phase:

### HerPNConv Forward Path Has Two Branches

```python
class HerPNConv(on.Module):
    def forward(self, x):
        if self.herpn1 is not None:
            # FUSED PATH: Uses ChannelSquare operations
            x = self.conv0(x)
            x = self.herpn1(x)  # ChannelSquare: depth=2
            x = self.conv1(x)
            x = self.herpn2(x)  # ChannelSquare: depth=2
            x = self.conv2(x)
        else:
            # UNFUSED PATH: Uses separate BatchNorm layers
            x = self.conv0(x)
            x = self.bn0_1(x)   # BatchNorm2d
            x = torch.square(x)
            x = self.bn1_1(x)   # BatchNorm2d
            x = x * math.sqrt(2)
            x = self.bn2_1(x)   # BatchNorm2d
            x = self.conv1(x)
            # ... continues with more BatchNorms
```

### Original Compilation Flow (BROKEN)

```python
# Original approach (WRONG!)
model = PatchCNN()
orion.fit(model, inp)              # ← Traces UNFUSED path (herpn1 is None)
input_level = orion.compile(model) # ← Calls init_orion_params() here (TOO LATE!)
```

**What happens:**
1. `orion.fit()` calls `model.forward()` to trace the computation graph
2. At this point, `herpn1` and `herpn2` are `None` (not yet initialized)
3. The `if self.herpn1 is not None` check fails
4. Tracer sees the **unfused** path with 3 separate `BatchNorm2d` layers per HerPNConv block
5. These BatchNorm layers appear in the DAG and consume extra levels
6. Later, `compile()` calls `init_orion_params()` which creates fused HerPN, but DAG is already locked

**Result:** DAG contains unfused BatchNorm operations → incorrect depth calculation → 11-level gap

### Why 11 Levels?

Each HerPNConv block in the unfused path has:
- 3 BatchNorm layers in the first activation (bn0_1, bn1_1, bn2_1)
- 3 BatchNorm layers in the second activation (bn0_2, bn1_2, bn2_2)
- Each BatchNorm's computation adds to the total depth
- The unfused path operations don't fold neatly like the fused ChannelSquare

Total extra depth from unfused operations ≈ 11 levels per analysis

## Solution: Custom Fit Workflow

### Corrected Compilation Flow

```python
# Step 1: Collect BatchNorm running statistics
model.eval()
with torch.no_grad():
    for _ in range(20):  # Sufficient iterations for stable statistics
        dummy_input = torch.randn(batch_size, channels, height, width)
        _ = model(dummy_input)

# Step 2: Fuse HerPN BEFORE orion.fit()
model.init_orion_params()  # ← Creates herpn1, herpn2 from BN statistics

# Verify fusion succeeded
assert model.layer1.herpn1 is not None
assert model.layer1.herpn2 is not None

# Step 3: Now trace with Orion (sees FUSED path)
orion.fit(model, inp)  # ← Now herpn1 is not None → traces fused path

# Step 4: Compile
input_level = orion.compile(model)
```

**What happens:**
1. We manually collect BatchNorm statistics with 20 forward passes
2. We call `init_orion_params()` which creates fused `ChannelSquare` modules
3. When `orion.fit()` traces the model, `herpn1` and `herpn2` are NOT None
4. The `if self.herpn1 is not None` check succeeds
5. Tracer sees the **fused** path with `ChannelSquare` operations
6. DAG contains the correct operations with proper depth

**Result:** DAG contains fused ChannelSquare operations → correct depth calculation → NO 11-level gap

### Verification

After applying the fix:

```
conv @ level=18
layer1_herpn1 @ level=17  ← Only 1-level gap (correct!)
```

The gap is now 1-2 levels as expected for the actual HerPN depth, not the 11-level catastrophe.

## Why This Works

1. **BatchNorm Statistics Preserved**: The 20 forward passes collect running_mean and running_var needed for fusion
2. **No Interference from Orion's Fuser**: Orion's automatic Conv→BN fusion (in `orion/core/fuser.py`) doesn't interfere because the fused HerPN doesn't have BatchNorm layers in the DAG
3. **Correct DAG Structure**: The tracer sees `ChannelSquare(weight0, weight1, weight2)` instead of separate `BatchNorm2d` layers
4. **Proper Depth Calculation**: `ChannelSquare` has `depth=2` when `weight2` is provided (which HerPN always does)

## Implementation Checklist

For any model using HerPN or similar fused activations:

- [ ] Implement `init_orion_params()` method in your model
- [ ] Manually run 20+ forward passes to collect BatchNorm statistics
- [ ] Call `model.init_orion_params()` BEFORE `orion.fit()`
- [ ] Verify fusion succeeded (check that fused modules are not None)
- [ ] Then proceed with normal Orion compilation flow

## Files Modified

### Configuration Files
- `configs/pcnn_optionB.yml` - High-level solution (50 LogQ primes, minimal bootstraps)
- `configs/pcnn_optionC.yml` - Practical solution (30 LogQ primes, 2-3 strategic bootstraps)

### Test Files
- `tests/models/test_pcnn.py` - Updated 3 test functions with custom fit workflow:
  - `test_pcnn_herpnconv_fhe()`
  - `test_pcnn_single_backbone_fhe()`
  - `test_pcnn_backbone_level_analysis()`
  
- `tests/models/test_pcnn_custom_fit.py` - Standalone verification test demonstrating the fix

### Model Files
- `models/pcnn.py` - No changes required (already has both paths and `init_orion_params()`)

## Testing

Run the verification test:

```bash
python tests/models/test_pcnn_custom_fit.py
```

Expected output:
```
✓ BatchNorm statistics collected
✓ HerPN modules created and fused
✓ No unfused BatchNorms in traced graph
✓ Found fused HerPN modules
✓ Level gap is reasonable: 1 levels (HerPN depth=2)
✓ ALL CHECKS PASSED!
```

## Additional Notes

### Depth Analysis

Full PCNN Backbone depth calculation:

```
Component                 Depth
---------------------------------
Initial Conv0               1
HerPNConv Block 1           6  (HerPN:2 + Conv:1 + HerPN:2 + Conv:1)
HerPNConv Block 2           6
HerPNConv Block 3           6
HerPNConv Block 4           6
HerPNConv Block 5           6
Final operations            3  (AvgPool, Linear, etc.)
---------------------------------
TOTAL                      34 levels
```

This exceeds ResNet20's depth (20 levels), explaining why PCNN needs more CKKS levels.

### CryptoFace SEAL Implementation

The SEAL code burns 11 levels via `mod_switch_to_next()` before computation:

```cpp
// CryptoFace: Drops from level 32 to level 21
for (int i = 0; i < boot_level - 3; i++) {
    evaluator.mod_switch_to_next_inplace(ct);
}
```

This is intentional positioning for their bootstrap strategy at level 21, not related to our Orion issue.

## Conclusion

The 11-level gap was caused by Orion tracing the unfused BatchNorm path during `fit()`. The solution is to fuse HerPN **before** calling `orion.fit()` by:

1. Manually collecting BatchNorm statistics
2. Calling `model.init_orion_params()`
3. Then calling `orion.fit()`

This ensures the tracer sees the correct fused operations, producing an accurate DAG with proper depth calculations.
