# Custom Fit Workflow for Models with Post-Init Fusion

## Overview

This document describes the **custom fit workflow** required for neural network models that use **post-initialization fusion** of batch normalization layers or similar operations. The most common example is the HerPN activation used in PCNN (PatchCNN) models.

## The Problem

### Background

Orion's compilation pipeline works in two main stages:

1. **`orion.fit(model, input)`** - Traces the model to build a computation DAG and collects per-layer statistics
2. **`orion.compile(model)`** - Assigns levels to each operation and generates FHE parameters

Some models, like PCNN with HerPN activations, use a **two-phase forward pass structure**:

- **Unfused phase** (during training/BN collection): Uses separate BatchNorm layers for statistical tracking
- **Fused phase** (after fusion): Replaces multiple BatchNorms with a single fused operation (e.g., HerPN's ChannelSquare)

### The Issue

If `orion.fit()` is called **before** the fusion happens, the tracer captures the **unfused graph** with separate BatchNorm operations. This causes:

1. **Incorrect DAG structure**: Extra BatchNorm nodes in the computation graph
2. **Incorrect depth calculation**: The unfused operations consume more multiplicative levels than the fused version
3. **Level assignment errors**: The extra depth can cause bootstrap placement to fail or levels to be misaligned

**Example**: In PCNN's HerPNConv, the unfused path uses 3 BatchNorm2d layers (bn0_1, bn1_1, bn2_1) which add ~11 extra levels of depth. After fusion, these are replaced by a single 2-depth ChannelSquare operation.

## The Solution: Custom Fit Workflow

### Standard Workflow (Incorrect for Post-Init Fusion)

```python
# ❌ DON'T DO THIS for models with post-init fusion
model = PatchCNN_Backbone(...)
inp = torch.randn(1, 3, 8, 8)

orion.fit(model, inp)  # Traces BEFORE fusion - captures unfused graph!
orion.compile(model)
```

### Custom Workflow (Correct)

```python
# ✅ DO THIS for models with post-init fusion
model = PatchCNN_Backbone(...)

# STEP 1: Collect BatchNorm statistics with forward passes
model.eval()
with torch.no_grad():
    for _ in range(20):  # Use 10-20 iterations for stable statistics
        _ = model(torch.randn(4, 3, 8, 8))

# STEP 2: Fuse operations BEFORE orion.fit()
# This creates the fused modules (e.g., HerPN from BatchNorms)
model.init_orion_params()

# STEP 3: Now fit and compile with the fused graph
inp = torch.randn(1, 3, 8, 8)
orion.fit(model, inp)  # Traces the FUSED graph
orion.compile(model)
```

### Why This Works

1. **Step 1** populates BatchNorm `running_mean` and `running_var` statistics needed for fusion
2. **Step 2** creates fused modules (e.g., HerPN) using those statistics **before tracing**
3. **Step 3** traces the model when fusion is already complete, so the tracer sees the fused operations

The key insight: **Fusion must happen before tracing, not after.**

## When to Use This Workflow

Use the custom fit workflow when your model has:

- ✅ **Post-initialization fusion**: Operations that are fused after the model is created (e.g., `init_orion_params()`)
- ✅ **Conditional forward paths**: Different execution paths during training vs inference based on fusion state
- ✅ **BatchNorm-based fusion**: Fusing multiple BatchNorm layers into a single operation

Examples:
- **PCNN with HerPN**: Uses HerPNConv blocks with post-init BatchNorm→HerPN fusion
- **Custom activation functions**: Any activation that fuses BatchNorm statistics into parameters

## Implementation Details

### HerPNConv Example

The HerPNConv module has two forward pass modes:

**Unfused mode** (when `self.herpn1 is None`):
```python
# Uses 3 separate BatchNorm layers per activation
x0 = self.bn0_1(torch.ones_like(x))
x1 = self.bn1_1(x)
x2 = self.bn2_1((torch.square(x) - 1) / math.sqrt(2))
out = (x0 / sqrt(2π) + x1 / 2 + x2 / sqrt(4π))
```

**Fused mode** (after `init_orion_params()` creates `self.herpn1`):
```python
# Uses single fused ChannelSquare operation
out = self.herpn1(x)  # Depth=2, no separate BatchNorms
```

### Depth Comparison

**Unfused path**:
- 3 BatchNorm2d operations (bn0_1, bn1_1, bn2_1)
- Each contributes to the DAG
- Appears as ~11 extra levels in depth calculation

**Fused path**:
- 1 ChannelSquare (HerPN) operation  
- Depth = 2 (quadratic activation)
- Correct multiplicative depth for level assignment

## Verification

After implementing the custom fit workflow, verify:

### 1. Check traced modules

```python
from orion.core import scheme
traced = scheme.trace

# Should see fused operations, not unfused BatchNorms
for name, module in traced.named_modules():
    print(name, type(module))
    
# ✅ Should see: layer.herpn1 (HerPN or ChannelSquare)
# ❌ Should NOT see: layer.bn0_1, layer.bn1_1, layer.bn2_1
```

### 2. Check level assignments

For residual blocks with shortcuts:

```python
# Both paths should end at the same level
main_level = model.layer2.conv2.level
shortcut_level = model.layer2.shortcut_conv.level

assert main_level == shortcut_level, f"Misalignment: {main_level} vs {shortcut_level}"
```

### 3. Check for 11-level gaps

Compare the level assignments between consecutive layers. You should NOT see unexplained large gaps (e.g., level 18 → level 7) unless there's an intentional bootstrap.

## Example: Full PCNN Test

```python
import torch
import orion
from models import PatchCNN_Backbone

# Initialize Orion scheme
orion.init_scheme('configs/pcnn_optionC.yml')

# Create model
model = PatchCNN_Backbone(num_classes=10, num_stages=5)
model.eval()

# STEP 1: Collect BatchNorm statistics
print("Collecting BatchNorm statistics...")
with torch.no_grad():
    for i in range(20):
        _ = model(torch.randn(4, 3, 8, 8))
print("✓ Statistics collected")

# STEP 2: Fuse HerPN activations BEFORE orion.fit()
print("Fusing HerPN activations...")
model.init_orion_params()
print("✓ Fusion complete")

# STEP 3: Fit and compile with fused graph
print("Running orion.fit()...")
inp = torch.randn(1, 3, 8, 8)
orion.fit(model, inp)
print("✓ Fit complete")

print("Compiling...")
input_level = orion.compile(model)
print(f"✓ Compilation complete. Input level: {input_level}")

# STEP 4: Verify shortcuts align (for residual blocks)
for layer_name in ['layer2', 'layer4']:
    layer = getattr(model, layer_name)
    if hasattr(layer, 'has_shortcut') and layer.has_shortcut:
        main = layer.conv2.level
        shortcut = layer.shortcut_conv.level
        assert main == shortcut, f"{layer_name}: {main} != {shortcut}"
        print(f"✓ {layer_name}: paths align at level={main}")
```

## Common Pitfalls

### ❌ Pitfall 1: Calling fit() before init_orion_params()

```python
# WRONG - traces unfused graph
model.init_orion_params()  # After fit - too late!
orion.fit(model, inp)
```

### ❌ Pitfall 2: Not collecting enough BN statistics

```python
# WRONG - only 1 forward pass, statistics may be unstable
_ = model(torch.randn(1, 3, 8, 8))
model.init_orion_params()  # May use poor statistics
```

Use 10-20 forward passes with batch size 2-4 for stable statistics.

### ❌ Pitfall 3: Using orion.compile() twice

```python
# WRONG - compiles twice, causes issues
orion.compile(model)
input_level = orion.compile(model)  # Duplicate compilation
```

Only call `orion.compile()` once.

## Related Issues

- **Issue**: Shortcut levels misaligned at addition points
  - **Cause**: Unfused BatchNorms added extra depth
  - **Solution**: Custom fit workflow ensures fused graph is traced

- **Issue**: Bootstrap placement fails or too many bootstraps required  
  - **Cause**: Incorrect depth calculation from unfused operations
  - **Solution**: Fused operations have correct depth

- **Issue**: Level drops unexpectedly (e.g., level 18 → level 7)
  - **Cause**: Unfused BatchNorm nodes consuming levels
  - **Solution**: Fused graph eliminates extra nodes

## Debugging Tips

### Enable debug output

```yaml
# In your config YAML
orion:
  debug: true
```

This shows FHE input/output levels at each layer during inference.

### Add debug prints in auto_bootstrap.py

The LevelDAG shortest path shows all nodes and their assigned levels. Look for:

- Unexpected node names (e.g., `layer.bn0_1` instead of `layer.herpn1`)
- Large level gaps between consecutive operations
- Mismatch between expected depth and actual level consumption

### Check traced graph structure

```python
from orion.core import scheme
traced = scheme.trace

# Print all traced modules
for name, module in traced.named_modules():
    if name:
        level = getattr(module, 'level', 'N/A')
        depth = getattr(module, 'depth', 'N/A')
        print(f"{name:40s} level={level:>3}, depth={depth}")
```

## Summary

**Key Principle**: For models with post-initialization fusion, **always fuse before tracing**.

**Custom Fit Workflow**:
1. Collect statistics (forward passes)
2. Call `model.init_orion_params()` 
3. Call `orion.fit()` and `orion.compile()`

This ensures the tracer captures the fused graph structure, leading to correct depth calculations and level assignments.

## References

- PCNN model: `models/pcnn.py`
- Test examples: `tests/models/test_pcnn.py`
- LevelDAG implementation: `orion/core/level_dag.py`
- Bootstrap placement: `orion/core/auto_bootstrap.py`
