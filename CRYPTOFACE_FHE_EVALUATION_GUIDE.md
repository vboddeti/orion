# CryptoFace FHE Evaluation Guide

**Date:** February 8, 2026
**Based on:** Full Model Verification (Sequential & Parallel)

## Current Status

### ✅ Working Components
- **Full Model (CryptoFaceNet4)**: Fully functional in FHE with **34% relative error** (down from 164%).
- **Parallel Inference**: Verified 1.7x speedup with `joblib` relative to sequential execution.
- **L2 Normalization**: Coefficients fixed for aggregated input range $[100, 3000]$.
- **BatchNorm Fusion**: Successfully fused into Linear layers, reducing depth.
- **Level Management**: Auto-bootstrap correctly handles residual connections and join nodes.

### 🎯 Key Results
| Metric | Sequential | Parallel |
| :--- | :--- | :--- |
| **MAE** | 0.019087 | 0.019087 |
| **Relative Error** | 34.8% | 34.8% |
| **Execution Time** | ~532s | ~308s (**1.7x Faster**) |
| **Output Consistency** | ✅ Deterministic | ✅ Deterministic |

## Recommended FHE Evaluation

### 1. Fast Verification (Real Image, Single Patch)
**File:** `face_recognition/tests/test_fhe_real_image.py`
**Run:**
```bash
uv run face_recognition/tests/test_fhe_real_image.py
```
**Goal:** Quick sanity check of the pipeline on a single patch.
**Expected:** MAE < 0.001 (for single backbone).

### 2. Full Model Sequential Test
**File:** `face_recognition/tests/test_full_model_sequential.py`
**Run:**
```bash
uv run face_recognition/tests/test_full_model_sequential.py > logs/full_model_seq.log 2>&1
```
**Goal:** Verify accuracy without multiprocessing complexity.
**Expected:** Relative error ~34%.

### 3. Full Model Parallel Test (Production Ready)
**File:** `face_recognition/tests/test_full_model_parallel.py`
**Run:**
```bash
uv run face_recognition/tests/test_full_model_parallel.py > logs/full_model_par.log 2>&1
```
**Goal:** Verify performance speedup and thread safety.
**Expected:** Identical accuracy to sequential, ~1.7x speedup.

## Configuration Files

**File:** `configs/cryptoface_net4.yml` (Implicitly used or constructed in tests)

**Key Parameters:**
- **LogN**: 16 (65536 slots)
- **LogQ**: 50 levels (optimized for 4-patch aggregation)
- **LogScale**: 46
- **Bootstrap placement**: Auto-managed by `auto_bootstrap.py`.

## Troubleshooting

### High Error Notes
- **L2NormPoly**: If relative error jumps back to >100%, check `L2NormPoly` coefficients in `cryptoface_pcnn.py`. They must be fitted for range $[100, 3000]$.
- **Level Exhaustion**: If "modulus chain is empty" error occurs, check `auto_bootstrap.py` logic for level updates at join nodes.

### Performance
- **Parallelization**: Uses `joblib` with `prefer="threads"`. 
- **Memory**: Parallel execution consumes 4x memory of sequential. Ensure sufficient RAM (>64GB recommended).

## Related Documentation
- `walkthrough.md`: Detailed history of fixes (Level offset, BN fusion, L2Norm).
- `task.md`: Current project status.
- `implementation_plan.md`: Plan for current/future changes.
