# CryptoFace FHE Evaluation Results

**Date:** February 8, 2026
**Test:** Full Model FHE Inference (Sequential & Parallel)
**Model:** CryptoFaceNet4 (4 patches + aggregation)
**Image:** Donald Rumsfeld (LFW dataset)

## Summary

**Status:** ✅ PASSED
**MAE:** 0.019087 (threshold: < 0.1)
**Max Error:** 0.0768
**Relative Error:** 34.8% (down from 164%)
**Inference Time:** 
- Sequential: ~532s
- Parallel: ~308s (**1.7x Speedup**)

## Test Configuration

- **Config File:** `configs/cryptoface_net4.yml`
- **Input:** Real LFW image (64×64), normalized to [-1, 1]
- **Patches:** 4 patches (32×32 each)
- **Input Level:** 15
- **Bootstrap Count:** 4 (managed by auto-bootstrap)

## Detailed Results

### ✅ Successful Components

#### 1. Full Model Architecture
- ✓ **4 Backbones**: Each processed independently (parallelized).
- ✓ **Aggregation**: Correctly sums encrypted 256-dim features.
- ✓ **Linear Layers**: Weights correctly handled with `gap=16` packing.
- ✓ **BatchNorm**: Fused into Linear layers (identity operation in forwarding).
- ✓ **L2 Normalization**: Applied after aggregation with correct $[100, 3000]$ range coefficients.

#### 2. Accuracy Metrics (Parallel vs Sequential)

| Metric | Sequential | Parallel | Match |
|-------|------------|----------|-------|
| MAE | 0.019087 | 0.019087 | ✓ |
| Relative Error | 34.8% | 34.8% | ✓ |
| Determinism | Exact | Exact | ✓ |

#### 3. Key Improvements
- **Level Offset Fix**: Correctly handles residual block joins, preventing early level exhaustion.
- **BN-Linear Fusion**: Reduced depth by fusing BatchNorm into Linear weights.
- **L2NormPoly**: Fixed coefficients to handle aggregated feature magnitude ($\sum x^2 \approx 1600$).

### ❌ Previous Failures (Fixed)

**Old Issue:** Relative Error ~164%
**Root Cause:** L2NormPoly coefficients were for normalized range ($y \approx 1$).
**Fix:** Recalculated for unnormalized range ($y \approx 1600$).

**Old Issue:** Modulus Chain Empty
**Root Cause:** Auto-bootstrap assigned max level to join nodes instead of min.
**Fix:** Updated `auto_bootstrap.py` for correct DAG traversal.

## Execution Logs

- **Sequential**: `logs/full_model_sequential_pre_patch_v6.log`
- **Parallel**: `logs/test_full_model_parallel.log`

## Next Steps

1. **Cleanup**: Remove temporary debug prints.
2. **Optimization**: Explore aggressive bootstrap placement for >2x speedup.
3. **Scaling**: Test with larger models (CryptoFaceNet9/16).
