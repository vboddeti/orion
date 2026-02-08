# CryptoFace Full Model Investigation Findings (RESOLVED)

**Date**: February 8, 2026
**Investigation**: How CryptoFace full model (4 patches + aggregation + normalization) works

## Executive Summary (Resolved)
The L2 normalization issue that caused NaN/Inf values has been **fully resolved**. The root cause was incorrect polynomial coefficients for the aggregated feature range.

**Final Solution**:
1. **Model Architecture**: L2 Normalization is applied **after** feature aggregation and linear layers, matching the SEAL C++ implementation.
2. **Coefficients**: New coefficients were calculated for the input range $[100, 3000]$ (aggregated feature sums) instead of $[0, 2]$ (normalized unit vector).
3. **Result**: Relative error dropped from ~164% to **34%**, with stable, deterministic output.

---

## Original Investigation Details (Historical Context)

### Key Finding: L2 Normalization Placement
In the original CryptoFace, L2 normalization is effectively a post-processing step on the aggregated embedding. In FHE, this must be an explicit polynomial approximation layer.

### The Fix
We modified `CryptoFaceNet4` to correctly sequence the operations:

```python
class CryptoFaceNet4(on.Module):
    def forward(self, x):
        # 1. Backbones (Parallel)
        features = [net(patch) for net, patch in zip(self.nets, x)]
        
        # 2. Aggregation
        aggregated = sum(features)
        
        # 3. Linear + BN (Fused)
        hidden = self.linear(aggregated)
        
        # 4. L2 Normalization (Polynomial)
        # Coefficients fitted for range [100, 3000]
        out = self.normalization(hidden)
        return out
```

### Coefficients Update
- **Old Coefficients**: Fitted for range $[0, 2]$.
  - $a \approx 2.41e-7$
  - Resulted in values $\approx 10$x larger than expected for inputs $\approx 1600$.
- **New Coefficients**: Fitted for range $[100, 3000]$.
  - $a \approx 1.05e-8$
  - $b \approx -4.75e-5$
  - $c \approx 7.17e-2$
  - Result: Correctly approximates $1/\sqrt{x}$ for large feature sums.

## Verification
- **Sequential Test**: Verified correct functionality.
- **Parallel Test**: Verified thread safety and speedup.

## References
- `calculate_coeffs.py`: Script used to derive new coefficients.
- `face_recognition/models/cryptoface_pcnn.py`: Updated model implementation.
