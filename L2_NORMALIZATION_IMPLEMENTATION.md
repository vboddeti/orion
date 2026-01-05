# L2 Normalization Implementation for CryptoFace

This document describes the implementation of L2 normalization with polynomial approximation for the CryptoFace PCNN model in Orion FHE.

## Overview

CryptoFace uses L2 normalization to normalize embeddings after feature aggregation. Since direct L2 normalization (which involves square root) is not FHE-friendly, we use a polynomial approximation of `1/√x`.

### Mathematical Background

The L2 normalization process:
1. Compute sum of squares: `y = sum(x²)` across all features
2. Apply polynomial approximation: `norm_inv = a*y² + b*y + c ≈ 1/√y`
3. Normalize: `x_norm = x * norm_inv`

The coefficients `(a, b, c)` are fitted to approximate `1/√x` using a three-point polynomial interpolation method.

## Implementation

### New Module: `L2NormPoly`

**Location**: `models/pcnn.py`

The `L2NormPoly` module implements FHE-friendly L2 normalization:

```python
from models.pcnn import L2NormPoly

# Create L2 normalization layer
l2norm = L2NormPoly(a=2.41e-07, b=-2.44e-04, c=1.09e-01, num_features=256)

# Apply normalization
x_normalized = l2norm(x)  # x shape: (B, num_features)
```

**Level Consumption**:
- `x²` (per feature): 1 level
- Sum reduction: 0 levels (additions only)
- `y²` (sum squared): 1 level
- Polynomial computation (`a*y² + b*y + c`): 0 levels (after rescale)
- Final multiplication (`x * norm_inv`): 1 level
- **Total depth: 3 levels**

### Updated Model: `CryptoFacePCNN`

**Location**: `face_recognition/models/cryptoface_pcnn.py`

The `CryptoFacePCNN` model now uses `L2NormPoly` instead of `BatchNorm1d` for final normalization:

```python
from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4

# Create model with L2 normalization coefficients
a, b, c = 2.41e-07, -2.44e-04, 1.09e-01
model = CryptoFaceNet4(l2_norm_coeffs=(a, b, c))
```

## Coefficient Estimation

### Utility Script: `estimate_l2_norm.py`

**Location**: `face_recognition/utils/estimate_l2_norm.py`

This script estimates optimal polynomial coefficients from validation datasets:

```bash
uv run python face_recognition/utils/estimate_l2_norm.py \
    --checkpoint face_recognition/checkpoints/cryptoface_net4.pth \
    --data_dir /path/to/faces_emore \
    --device cuda
```

The script:
1. Loads the pretrained CryptoFace model
2. Computes embeddings for all validation datasets (LFW, CFP_FP, CPLFW, AGEDB_30, CALFW)
3. Estimates polynomial coefficients using 10-fold cross-validation
4. Saves coefficients to threshold files

**Output Files**:
- `face_recognition/checkpoints/threshold_lfw.txt`
- `face_recognition/checkpoints/threshold_cfp_fp.txt`
- `face_recognition/checkpoints/threshold_cplfw.txt`
- `face_recognition/checkpoints/threshold_agedb_30.txt`
- `face_recognition/checkpoints/threshold_calfw.txt`

### Coefficient Values

Current estimated coefficients (averaged across 10 folds):

| Dataset   | a (×10⁻⁷) | b (×10⁻⁴) | c (×10⁻¹) |
|-----------|-----------|-----------|-----------|
| LFW       | 2.410     | -2.441    | 1.093     |
| CFP_FP    | 6.019     | -4.416    | 1.355     |
| CPLFW     | 6.796     | -4.684    | 1.374     |
| AGEDB_30  | 3.512     | -3.063    | 1.179     |
| CALFW     | 2.503     | -2.501    | 1.102     |

## Loading Coefficients

### Utility Module: `l2_coeffs.py`

**Location**: `face_recognition/utils/l2_coeffs.py`

Convenience functions for loading coefficients:

```python
from face_recognition.utils.l2_coeffs import load_l2_coeffs, get_default_coeffs

# Load coefficients for a specific dataset
a, b, c = load_l2_coeffs('lfw')

# Get default coefficients (LFW)
a, b, c = get_default_coeffs()

# Print all available coefficients
from face_recognition.utils.l2_coeffs import print_all_coeffs
print_all_coeffs()
```

## Usage Examples

### Example 1: Create model with specific coefficients

```python
from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4
from face_recognition.utils.l2_coeffs import load_l2_coeffs

# Load coefficients for LFW dataset
a, b, c = load_l2_coeffs('lfw')

# Create model
model = CryptoFaceNet4(l2_norm_coeffs=(a, b, c))

# Forward pass
import torch
x = torch.randn(1, 3, 64, 64)
embedding = model(x)  # Shape: (1, 256)
```

### Example 2: Create model with default coefficients

```python
from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4

# Model will use default LFW coefficients
model = CryptoFaceNet4()

# Or explicitly provide coefficients
model = CryptoFaceNet4(l2_norm_coeffs=(2.41e-07, -2.44e-04, 1.09e-01))
```

### Example 3: FHE inference with L2 normalization

```python
import orion
from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4
from face_recognition.utils.l2_coeffs import load_l2_coeffs

# Load coefficients
a, b, c = load_l2_coeffs('lfw')

# Create model
model = CryptoFaceNet4(l2_norm_coeffs=(a, b, c))

# Initialize FHE scheme
orion.init_scheme("configs/cryptoface_net4.yml")

# Prepare model (custom fit workflow)
model.eval()
with torch.no_grad():
    for _ in range(20):
        _ = model(torch.randn(1, 3, 64, 64))

# Fuse operations
model.init_orion_params()

# Fit and compile
inp = torch.randn(1, 3, 64, 64)
orion.fit(model, inp)
input_level = orion.compile(model)

# Encrypt and run FHE inference
vec_ptxt = orion.encode(inp, input_level)
vec_ctxt = orion.encrypt(vec_ptxt)
model.he()
out_ctxt = model(vec_ctxt)
out_fhe = out_ctxt.decrypt().decode()
```

## Testing

### Test Scripts

1. **`test_l2norm_poly.py`**: Tests L2NormPoly module against reference implementation
   ```bash
   uv run python test_l2norm_poly.py
   ```

2. **`test_cryptoface_pcnn_l2norm.py`**: Tests full CryptoFacePCNN integration
   ```bash
   uv run python test_cryptoface_pcnn_l2norm.py
   ```

### Expected Results

Both tests should pass with:
- L2NormPoly matches reference implementation (max diff < 1e-6)
- Normalized embeddings have L2 norm ≈ 1.0

## Architecture Changes

### Before (using BatchNorm1d)
```
Input → Backbones → Linear layers → Sum → BatchNorm1d → Output
```

### After (using L2NormPoly)
```
Input → Backbones → Linear layers → Sum → L2NormPoly → Output
                                             ↓
                                    (a*y² + b*y + c) × x
```

## Comparison with SEAL Implementation

This implementation matches the SEAL reference in `CryptoFace/cnn_ckks/cpu-ckks/single-key/cnn/cnn_seal.cpp`:

```cpp
void l2norm_seal(Ciphertext &ct, ...) {
    // Step 1: Compute sum of squares
    evaluator.multiply_inplace_reduced_error(temp, temp, relin_keys);
    evaluator.rescale_to_next_inplace(temp);

    // Step 2: Sum reduction across 256 elements
    // ... rotation and addition logic ...

    // Step 3: Apply polynomial a*sum² + b*sum + c
    // ... polynomial computation ...

    // Step 4: Multiply by normalization factor
    evaluator.multiply_inplace_reduced_error(ct, temp, relin_keys);
}
```

## File Summary

### New Files
- `models/pcnn.py` (modified): Added `L2NormPoly` class
- `face_recognition/models/cryptoface_pcnn.py` (modified): Updated to use `L2NormPoly`
- `face_recognition/utils/estimate_l2_norm.py`: Coefficient estimation script
- `face_recognition/utils/l2_coeffs.py`: Coefficient loading utilities
- `test_l2norm_poly.py`: Unit tests for L2NormPoly
- `test_cryptoface_pcnn_l2norm.py`: Integration tests

### Generated Files
- `face_recognition/checkpoints/threshold_*.txt`: Estimated coefficients for each dataset

## Next Steps

1. **Test with FHE**: Run full FHE inference to verify level assignment and accuracy
2. **Optimize sum reduction**: Implement efficient rotation-based sum for FHE mode
3. **Benchmark**: Compare inference time and accuracy with SEAL implementation
4. **Update configs**: Ensure CKKS parameters account for 3 additional levels from L2 normalization

## References

- **CryptoFace Paper**: CVPR 2025
- **SEAL Implementation**: `CryptoFace/cnn_ckks/cpu-ckks/single-key/cnn/cnn_seal.cpp`
- **Helper Functions**: `CryptoFace/helper.py` (l2norm function)
- **Coefficient Estimation**: `CryptoFace/ckks.py` (3-point polynomial fitting)
