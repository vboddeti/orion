# CryptoFace CKKS Implementation Analysis

## Summary

Analysis of how CryptoFace's SEAL/CKKS C++ implementation actually processes the final linear layer in encrypted inference, compared to the PyTorch training model.

**Key Finding**: The final linear layer is **split into per-patch transformations** during inference, not used as a single monolithic matrix!

---

## PyTorch Training Model vs CKKS Inference Model

### Training Model (Unfused)

```python
# CryptoFace/models/pcnn.py (lines 40-42)
class PatchCNN(nn.Module):
    def __init__(self, input_size, patch_size):
        N = (input_size // patch_size) ** 2
        dim = 64 * 2 * 2  # 256

        self.nets = nn.ModuleList([Backbone(...) for _ in range(N)])
        self.linear = nn.Linear(N * dim, 256)  # Single linear: (N*256) → 256
        self.bn = nn.BatchNorm1d(256, affine=False)

    def forward(self, x):
        # Process patches through backbones
        out = self._forward(x)  # Shape: (B, N, 256)

        # Concatenate all patch features
        out_global = rearrange(out_global, 'b n c -> b (n c)')  # (B, N*256)

        # Single linear transformation
        out_global = self.linear(out_global)  # (B, N*256) @ (N*256, 256) = (B, 256)

        # Normalization
        out_global = self.bn(out_global)  # (B, 256)

        return out_global
```

**For CryptoFaceNet4 (2×2 patches, N=4)**:
- Linear layer: `nn.Linear(1024, 256)`
- Weight shape: `(256, 1024)`
- Applied to concatenated features: `[B, 1024] @ [1024, 256]^T = [B, 256]`

---

### Inference Model (Fused)

#### PyTorch Fused Forward

```python
# CryptoFace/models/pcnn.py (lines 79-109)
@torch.no_grad()
def fuse(self):
    mean = self.bn.running_mean  # (256,)
    var = self.bn.running_var    # (256,)
    weight = self.linear.weight  # (256, N*256)
    bias = self.linear.bias      # (256,)
    e = self.bn.eps

    # Fuse BatchNorm into linear layer
    weight = torch.divide(weight.T, torch.sqrt(var + e))  # (N*256, 256)
    bias = torch.divide(bias - mean, torch.sqrt(var + e))  # (256,)
    weight = weight.T  # Back to (256, N*256)

    # CRITICAL: Split weight matrix into N chunks!
    self.weights = nn.ParameterList(torch.chunk(weight, self.H * self.W, dim=1))
    # For N=4: Creates 4 tensors of shape (256, 256)

    # Divide bias by N for aggregation
    self.bias = nn.Parameter(bias / (self.H * self.W))  # (256,)

def forward_fuse(self, x):
    H, W = self.H, self.W
    N = H * W

    # Process each patch separately
    y = [None for _ in range(N)]
    for i in range(N):
        y[i] = self.nets[i].forward_fuse(x[i])  # Backbone output: (B, 256)

        # Apply per-patch linear transformation with weight chunk i
        y[i] = y[i] @ self.weights[i].T + self.bias  # (B, 256) @ (256, 256) = (B, 256)

    # Stack and sum across patches
    out = torch.stack(y, dim=0)     # (N, B, 256)
    out = rearrange(out, 'n b c -> b n c')  # (B, N, 256)
    out = reduce(out, 'b n c -> b c', 'sum')  # (B, 256)

    return out
```

**For CryptoFaceNet4 (N=4)**:
- Original weight `(256, 1024)` is chunked into:
  - `self.weights[0]`: `(256, 256)` - for patch 0
  - `self.weights[1]`: `(256, 256)` - for patch 1
  - `self.weights[2]`: `(256, 256)` - for patch 2
  - `self.weights[3]`: `(256, 256)` - for patch 3
- Each patch applies its own 256→256 transformation
- Outputs are summed: `Σ(weights[i] @ features[i]) = final_embedding`

---

#### CKKS C++ Implementation

```cpp
// CryptoFace/cnn_ckks/cpu-ckks/single-key/cnn/infer_seal.cpp

// ========== Weight Loading (lines 305-367) ==========
void import_weights_pcnn(size_t num_nets, ...) {
    // Allocate storage for each subnet's weights
    all_linear_weight.resize(num_nets);  // Vector of vectors

    // Load weights for each subnet from separate directories
    for (size_t i = 0; i < num_nets; i++) {
        string net_dir = dir + "/net" + to_string(i);  // e.g., "net0", "net1", ...
        import_weights_pcnn_net(net_dir, all_linear_weight[i], ...);
    }

    // Load shared bias (same across all subnets)
    in.open(dir + "/bias.txt");
    for(long i=0; i<256; i++) {in>>val; linear_bias.emplace_back(val);}
    in.close();
}

// ========== Per-Subnet Weight Loading (lines 369-556) ==========
void import_weights_pcnn_net(string &dir, vector<double> &linear_weight, ...) {
    // Load 256×256 linear weight matrix from this subnet's directory
    in.open(dir + "/weight.txt");  // e.g., "net0/weight.txt"
    for(long i=0; i<256*256; i++) {
        in>>val;
        linear_weight.emplace_back(val);
    }
    in.close();

    // Also loads: conv weights, HerPN coefficients, BatchNorm stats, etc.
}

// ========== Encrypted Inference (lines 1614-1945) ==========
void patchcnn(...) {
    size_t num_nets = (input_size / patch_size) * (input_size / patch_size);  // N = 4 for 64×64

    // Process each subnet/patch in parallel
    vector<Ciphertext> features(num_nets);

    #pragma omp parallel for num_threads(num_nets)
    for(size_t subnet = 0; subnet < num_nets; subnet++) {
        // Extract this subnet's weights
        vector<double> linear_weight(all_linear_weight_share[subnet].begin(),
                                      all_linear_weight_share[subnet].end());  // 256×256
        vector<double> linear_bias(linear_bias_share.begin(),
                                    linear_bias_share.end());  // 256 (shared)

        // ... process backbone layers with HerPN, convolutions, etc. ...

        // HerPNPool (line 1912)
        herpn_print(cnn, cnn, a0[j], a1[j], ...);
        averagepooling_seal_print(cnn, cnn, ...);

        // BatchNorm1D (line 1916)
        batchnorm1d_seal_print(cnn, cnn, ..., bn_weight, bn_bias, bn_running_mean, bn_running_var, ...);

        // Linear layer: 256 → 256 (line 1919)
        fully_connected_seal_print(cnn, cnn, linear_weight, linear_bias, 256, 256, ...);

        // Save encrypted feature vector for this patch
        features[subnet] = cnn.cipher();
    }

    // Aggregate features across all patches (lines 1927-1928 for offline, 1940-1941 for online)
    gallery = features[0];
    for(size_t subnet = 1; subnet < num_nets; subnet++) {
        evaluator.add_inplace_reduced_error(gallery, features[subnet]);
    }

    // L2 Normalization (line 1931)
    l2norm_seal_print(gallery, logn, ..., coef_a, coef_b, coef_c, ...);
}
```

**Weight Storage Structure**:
```
weights/cryptoface_pcnn64/
├── net0/
│   ├── weight.txt          # 256×256 = 65,536 values
│   ├── conv_weight.txt
│   ├── layers_0_a0.txt
│   ├── layers_0_a1.txt
│   ├── ...
│   └── bn_weight.txt
├── net1/
│   ├── weight.txt          # 256×256 = 65,536 values
│   └── ...
├── net2/
│   └── weight.txt          # 256×256 = 65,536 values
├── net3/
│   └── weight.txt          # 256×256 = 65,536 values
└── bias.txt                # 256 values (shared across all subnets)
```

---

## Mathematical Equivalence

### Why Chunking Works

Matrix multiplication is **distributive** over concatenation:

```
Given:
- Features: F = [f0, f1, f2, f3]  where fi ∈ R^256
- Weight:   W = [W0 | W1 | W2 | W3]  where Wi ∈ R^(256×256)

Original (training model):
out = F @ W^T = [f0, f1, f2, f3] @ [W0 | W1 | W2 | W3]^T
    = f0 @ W0^T + f1 @ W1^T + f2 @ W2^T + f3 @ W3^T

Chunked (inference model):
out = Σ(fi @ Wi^T) for i in [0, 3]
    = f0 @ W0^T + f1 @ W1^T + f2 @ W2^T + f3 @ W3^T

Result: IDENTICAL!
```

### Benefits for FHE

1. **Parallelization**: Each patch can be processed independently
   - CKKS uses OpenMP: `#pragma omp parallel for num_threads(num_nets)`
   - Orion uses multiprocessing fork + serialization

2. **Memory efficiency**: Smaller intermediate ciphertexts
   - Process 256-dim vectors instead of 1024-dim concatenated vector

3. **Simpler homomorphic operations**:
   - Matrix-vector multiplication: 256×256 instead of 1024×256
   - Addition for aggregation is cheap in FHE

---

## Implications for Orion Implementation

### Current Orion PCNN

```python
# models/pcnn.py (lines 650-651)
class PatchCNN(on.Module):
    def __init__(self, ...):
        self.linear = nn.ModuleList([on.Linear(dim, 1024) for _ in range(N)])
        # 4 separate 256→1024 linears
        # Output: 4 separate 1024-dim vectors (no aggregation)
```

**Issues**:
1. Wrong output dimension: 1024 vs 256
2. No aggregation step (missing sum across patches)
3. No final normalization (missing BatchNorm1d equivalent)

### Required Changes for CryptoFace Compatibility

```python
class CryptoFacePatchCNN(on.Module):
    def __init__(self, num_patches=4, dim=256, output_dim=256):
        # Per-patch linears (256 → 256)
        self.linear = nn.ModuleList([on.Linear(dim, output_dim) for _ in range(num_patches)])

        # Final normalization (replace BatchNorm1d with ChannelSquare)
        # Need to convert BatchNorm running_mean and running_var to ChannelSquare weights
        self.normalization = ChannelSquare(weight0, weight1)

    def forward(self, x):
        # Extract patches
        patches = extract_patches(x, self.patch_size)  # (N, B, C, H, W)

        # Process each patch
        features = []
        for i in range(num_patches):
            feat = self.nets[i](patches[i])  # (B, 256)
            feat = self.linear[i](feat)      # (B, 256)
            features.append(feat)

        # Aggregate: sum across patches
        out = torch.stack(features, dim=0).sum(dim=0)  # (B, 256)

        # Normalize
        out = self.normalization(out)  # (B, 256)

        return out
```

### Weight Loading Strategy

**Direct loading from CKKS weight files**:

```python
def load_cryptoface_weights(model, weight_dir, num_nets=4):
    for i in range(num_nets):
        # Load linear weight for this subnet
        weight_path = f"{weight_dir}/net{i}/weight.txt"
        weight = np.loadtxt(weight_path).reshape(256, 256)
        model.linear[i].weight.data = torch.from_numpy(weight).T  # Transpose for PyTorch

    # Load shared bias
    bias_path = f"{weight_dir}/bias.txt"
    bias = np.loadtxt(bias_path)
    for i in range(num_nets):
        model.linear[i].bias.data = torch.from_numpy(bias) / num_nets

    # Load normalization parameters (from BatchNorm1d)
    # Convert to ChannelSquare weights
    # ...
```

---

## Recommendations

### For Task 1.2 (Create cryptoface_pcnn.py)

✅ **Use per-patch linear layers** (matches CryptoFace fused model and CKKS)
```python
self.linear = nn.ModuleList([on.Linear(256, 256) for _ in range(N)])
```

✅ **Add aggregation step** (sum across patches)
```python
out = torch.stack(features, dim=0).sum(dim=0)
```

✅ **Add final normalization** (convert BatchNorm1d to ChannelSquare)
```python
self.normalization = ChannelSquare(weight0, weight1)
```

❌ **Don't use single linear** `on.Linear(N*256, 256)`
- Harder to parallelize
- Doesn't match weight storage format
- Less efficient for FHE

### For Task 1.3 (Weight Loading)

✅ **Load from per-subnet directories**
- `net0/weight.txt`, `net1/weight.txt`, etc.
- Each contains 256×256 = 65,536 float values
- Shared `bias.txt` with 256 values

✅ **Convert BatchNorm1d to ChannelSquare**
- Extract `running_mean` and `running_var` from normalization layer stats
- Compute ChannelSquare coefficients (see HerPN fusion formulas)

✅ **Verify loading correctness**
- Run cleartext forward pass
- Compare with CryptoFace reference outputs
- Check cosine similarity > 0.999

---

## References

- **PyTorch Training Model**: `CryptoFace/models/pcnn.py` (lines 40-123)
- **PyTorch Fused Model**: `CryptoFace/models/pcnn.py` (lines 79-109)
- **CKKS Implementation**: `CryptoFace/cnn_ckks/cpu-ckks/single-key/cnn/infer_seal.cpp`
  - Weight loading: lines 305-556
  - Encrypted inference: lines 1614-1945
- **Orion PCNN**: `models/pcnn.py` (lines 580-680)

---

**Last Updated**: 2025-11-22
**Analysis Complete**: Task 1.1 ✓
