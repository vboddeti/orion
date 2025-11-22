# Bootstrap Placement Margin: Understanding the Q/Scale Constraint

## Summary

The `bootstrap_placement_margin` parameter prevents bootstrap operations from being placed at dangerously low levels where **scale accumulation** causes the Q/Scale ratio to fall below the threshold required by the CKKS bootstrap circuit.

## Background: Two Types of Parameters

### 1. `margin` (default: 2)
- **Purpose**: Controls the range expansion during bootstrap prescale/postscale operations
- **Effect**: With `margin=2`, input range `[A, B]` expands to `[A-range, B+range]` where `range = 2 × half_range`
- **Impact on operations**:
  - `margin=2` → `prescale=1/2`, `postscale=2`
  - `margin=5` → `prescale=1/5`, `postscale=5`
- **Why we use margin=2**: Smaller range expansion = better precision, simpler arithmetic, matches system defaults

### 2. `bootstrap_placement_margin` (default: 0)
- **Purpose**: Prevents operations from outputting below a minimum level threshold
- **Effect**: With `bootstrap_placement_margin=2`, operations must satisfy: `output_level = level - depth >= 2`
- **Critical role**: Ensures bootstraps happen at levels with sufficient Q/Scale ratio

## The Problem: Scale Accumulation in CKKS

### How Scale Grows in CKKS

In CKKS homomorphic encryption:

1. **Multiplication grows scale**: `scale_out = scale_a × scale_b`
2. **Rescaling consumes a level**: `level -= 1`, `scale /= prime`
3. **Over many operations**: Scale can accumulate to enormous values

### Example: PCNN Without `bootstrap_placement_margin`

Starting configuration:
- Input at level 30 with scale = 2^46
- Compute through 28 levels of operations (level 30 → level 2)
- Bootstrap scheduled for `layer5_conv1 @ level=2`

**What happens:**
```
After 28 levels of computation:
  • Ciphertext reaches level 2
  • Scale has accumulated to ≈ 2^1513 (started at 2^46!)
  • Q at level 2 = 2^1518 (sum of remaining LogQ primes)
  • Q/Scale = 2^1518 / 2^1513 = 2^5 = 32
```

**Bootstrap fails:**
```
Lattigo bootstrap requirement:
  Q/Scale >= 0.5 × Q[0] / MessageRatio
  Q/Scale >= 0.5 × 2^51 / 1
  Q/Scale >= 2^50

Actual Q/Scale = 2^5 = 32
32 << 2^50  ❌ BOOTSTRAP FAILS!

Error: "cannot BootstrapMany: initial Q/Scale = 31.993392 <
        0.5*Q[0]/MessageRatio = 256.000000"
```

### Solution: `bootstrap_placement_margin=2`

With this parameter, the LevelDAG optimization is constrained:
```python
# In level_dag.py estimate_layer_latency():
placement_margin = getattr(module.scheme, 'bootstrap_placement_margin', 0)
if level - module.depth < placement_margin:
    return float("inf")  # Prevent this assignment
```

**Effect:**
- Bootstrap must happen at level 4 (not level 2)
- At level 4:
  - Q = 2^1610 (2 more 46-bit primes)
  - Scale ≈ 2^1513 (similar accumulation)
  - **Q/Scale ≈ 2^97**
- 2^97 >> 2^50 ✓ **Bootstrap succeeds!**

## Why ResNet Doesn't Need This

ResNet uses a different strategy:
- **Total levels**: 11 (vs PCNN's 35)
- **Bootstrap count**: 38 bootstraps (vs PCNN's 1)
- **Bootstrap frequency**: Throughout the network, not just at the end

**Key difference**: Frequent bootstrapping **refreshes the scale** back to normal levels, preventing massive accumulation. ResNet bootstraps at levels 1, 2, and 5 repeatedly, so scale never grows as large as in PCNN.

## PCNN Configuration Comparison

### ResNet
```yaml
ckks_params:
  LogQ: [55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40]  # 11 levels
  LogScale: 40

orion:
  margin: 2
  bootstrap_placement_margin: 0  # Not needed due to frequent bootstraps
```

Result: 38 bootstraps, but scale stays manageable

### PCNN (Final Working Configuration)
```yaml
ckks_params:
  LogQ: [51, 46, 46, ..., 46]  # 35 levels total
  LogScale: 46

orion:
  margin: 2
  bootstrap_placement_margin: 2  # Critical for preventing low-level bootstrap
```

Result: Only 1 bootstrap, but must happen at level 4+ to avoid Q/Scale issues

## Technical Deep Dive: Bootstrap Operations

### Bootstrap is Different from Regular Operations

**Regular operations** (conv, batchnorm, etc.):
- Input level L → Output level L - depth
- Consume levels through multiplication + rescaling
- Can become "computation-dead" at level 0

**Bootstrap operations**:
- Input at level L_in → **Output at level L_max** (top level!)
- Refresh the ciphertext back to maximum level
- **Requirement**: Input must have Q/Scale >= threshold

### Why Bootstrap Can Work at Low Input Levels

The bootstrap operation itself doesn't care about the input level - it refreshes TO the maximum level. What matters is the **Q/Scale ratio at the input level**.

From `orion/backend/lattigo/bootstrapper.go`:
```go
func Bootstrap(ciphertextID, numSlots C.int) C.int {
    ctIn := RetrieveCiphertext(int(ciphertextID))
    bootstrapper := GetBootstrapper(int(numSlots))

    // Bootstrap refreshes to maximum level
    ctOut, err := bootstrapper.Bootstrap(ctBtp)
    if err != nil {
        panic(err)  // Panics if Q/Scale too small!
    }

    return C.int(idx)
}
```

The Lattigo library checks that `Q/Scale >= threshold` before performing the expensive bootstrap computation.

## Level Assignment Optimization

The LevelDAG uses Dijkstra's shortest path algorithm to find optimal level assignments that minimize total cost (layer latency + bootstrap latency).

**Without placement margin:**
- Optimization chooses level 2 for bootstrap (minimizes layer latency)
- But Q/Scale = 32 causes bootstrap to fail

**With placement_margin=2:**
- Optimization blocked from choosing levels where output < 2
- Chooses level 4 for bootstrap (slightly higher layer latency)
- But Q/Scale = 2^97 allows bootstrap to succeed

The constraint acts as a "guard rail" ensuring the optimizer doesn't choose levels that are theoretically optimal for latency but practically broken due to Q/Scale constraints.

## Calculation: Minimum Safe Bootstrap Level

For PCNN with 35 levels:

**Requirements:**
- Bootstrap needs: Q/Scale >= 2^50
- LogQ primes: [51, 46, 46, 46, ...] (35 total)
- Accumulated scale at bootstrap: ≈ 2^1513

**Analysis by level:**

| Level | Q (log) | Q/Scale (log) | Q/Scale | Safe? |
|-------|---------|---------------|---------|-------|
| 2 | 1518 | 5 | 32 | ❌ (32 << 2^50) |
| 3 | 1564 | 51 | 2^51 | ⚠️ (barely!) |
| 4 | 1610 | 97 | 2^97 | ✅ (safe margin) |
| 5 | 1656 | 143 | 2^143 | ✅ (very safe) |

**Conclusion:**
- `bootstrap_placement_margin=1` might work (allows level 3, Q/Scale ≈ 2^51)
- `bootstrap_placement_margin=2` is safer (ensures level 4+, Q/Scale ≈ 2^97+)

## Final Recommendations

1. **For PCNN-like networks** (deep computation, minimal bootstraps):
   - Use `bootstrap_placement_margin=2` for safety
   - Could potentially use `bootstrap_placement_margin=1` if scale behavior is well-understood

2. **For ResNet-like networks** (frequent bootstraps):
   - `bootstrap_placement_margin=0` is fine
   - Frequent bootstraps prevent scale accumulation

3. **General principle**:
   - Longer computation between bootstraps → Higher placement margin needed
   - More frequent bootstraps → Lower placement margin acceptable

## References

- PCNN config: `configs/pcnn_optionC.yml`
- ResNet config: `configs/resnet.yml`
- Level assignment: `orion/core/level_dag.py:185-191`
- Bootstrap operation: `orion/nn/operations.py:38-100`
- Lattigo backend: `orion/backend/lattigo/bootstrapper.go`
