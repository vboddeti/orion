# CryptoFace vs Orion PCNN Architecture Comparison

**Task 1.1 Analysis**: Comparing CryptoFace and Orion PCNN implementations

---

## Summary of Key Differences

| Aspect | CryptoFace | Orion | Impact |
|--------|------------|-------|--------|
| **Framework** | Pure PyTorch | PyTorch + Orion layers | Need adapter layer |
| **HerPN Fusion** | Manual fuse() method | init_orion_params() | Different fusion workflow |
| **Parallel Processing** | CUDA streams | Fork + serialization / ThreadPoolExecutor | Different parallelization |
| **Final Layer** | Linear → BatchNorm | Linear → ChannelSquare | Different normalization |
| **Patch Config** | Fixed 2×2 (4 patches) | Configurable (4, 9, 16) | More flexible |
| **Shortcut Scaling** | Multiplies by a2 | Uses ChannelSquare(w1=a2) | Same math, different impl |

---

## 1. Overall Architecture

### **CryptoFace PatchCNN**
```python
class PatchCNN(nn.Module):
    def __init__(self, input_size, patch_size):
        # Fixed H×W grid (typically 2×2 = 4 patches)
        self.H, self.W = input_size // patch_size, input_size // patch_size
        N = self.H * self.W

        # Backbone for each patch
        self.nets = nn.ModuleList([Backbone(output_size=(2,2)) for _ in range(N)])

        # Global linear layer for all patches
        self.linear = nn.Linear(N * dim, 256)

        # Normalization (affine=False)
        self.bn = nn.BatchNorm1d(256, affine=False)

        # Jigsaw task (auxiliary)
        self.jigsaw = nn.Linear(dim, N)
```

### **Orion PatchCNN**
```python
class PatchCNN(on.Module):
    def __init__(self, input_size, patch_size, sqrt_weights, output_size=(4,4)):
        # Configurable H×W grid
        self.H = input_size // patch_size
        self.W = input_size // patch_size
        self.N = self.H * self.W

        # Backbone for each patch
        self.nets = nn.ModuleList([Backbone(output_size) for _ in range(self.N)])

        # Separate linear layer per patch (FHE-friendly)
        self.linear = nn.ModuleList([on.Linear(self.dim, 1024) for _ in range(self.N)])

        # Normalization via ChannelSquare (quadratic for FHE)
        self.normalization = ChannelSquare(sqrt_weights[0], sqrt_weights[1])
```

**Key Differences**:
1. ✗ **Linear Layer**: CryptoFace uses single shared Linear (N×dim → 256), Orion uses per-patch Linear (dim → 1024)
2. ✗ **Normalization**: CryptoFace uses BatchNorm1d, Orion uses ChannelSquare
3. ✓ **Jigsaw**: CryptoFace has auxiliary task, Orion does not (not needed for inference)
4. ✓ **Output dim**: CryptoFace outputs 256, Orion outputs 1024

---

## 2. Backbone Architecture

### **Both Implementations (Nearly Identical)**
```python
# Initial conv
self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)

# 5 HerPNConv blocks
self.layers = nn.Sequential(
    HerPNConv(16, 16),       # layer1
    HerPNConv(16, 32, 2),    # layer2, stride=2
    HerPNConv(32, 32),       # layer3
    HerPNConv(32, 64, 2),    # layer4, stride=2
    HerPNConv(64, 64)        # layer5
)

# HerPN + Pooling
self.herpnpool = HerPNPool(64, output_size=(2, 2))

# Flatten
self.flatten = Flatten()

# Final BatchNorm
self.bn = nn.BatchNorm1d(output_size[0] * output_size[1] * 64)  # 2×2×64 = 256
```

**Identical**: Backbone structure is the same in both implementations!

---

## 3. HerPN Activation

### **CryptoFace HerPN**
```python
class HerPN(nn.Module):
    def __init__(self, planes):
        self.bn0 = nn.BatchNorm2d(planes, affine=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=False)
        self.weight = nn.Parameter(torch.ones(planes, 1, 1))
        self.bias = nn.Parameter(torch.zeros(planes, 1, 1))

    def forward(self, x):
        x0 = self.bn0(torch.ones_like(x))
        x1 = self.bn1(x)
        x2 = self.bn2((torch.square(x) - 1) / math.sqrt(2))
        out = x0/√(2π) + x1/2 + x2/√(4π)
        out = self.weight * out + self.bias
        return out
```

### **Orion HerPN (Part of HerPNConv)**
```python
# Unfused mode (training):
x0 = self.bn0_1(torch.ones_like(x))
x1 = self.bn1_1(x)
x2 = self.bn2_1((torch.square(x) - 1) / math.sqrt(2))
out = x0/√(2π) + x1/2 + x2/√(4π)

# Fused mode (inference):
# Created via HerPN class which computes:
# w2*x² + w1*x + w0
```

**Identical Math**: Both use same HerPN formulation with 3 BatchNorms!

---

## 4. HerPN Fusion (CRITICAL DIFFERENCE)

### **CryptoFace Fusion**
```python
@torch.no_grad()
def fuse(self):
    # Extract BatchNorm stats
    m0, v0 = self.herpn1.bn0.running_mean, self.herpn1.bn0.running_var
    m1, v1 = self.herpn1.bn1.running_mean, self.herpn1.bn1.running_var
    m2, v2 = self.herpn1.bn2.running_mean, self.herpn1.bn2.running_var
    g, b = self.herpn1.weight.squeeze(), self.herpn1.bias.squeeze()
    e = self.herpn1.bn0.eps

    # Compute fused coefficients
    a2 = g / √(8π(v2 + e))
    a1 = g / (2√(v1 + e))
    a0 = b + g * (...)

    # Fuse convolution weights
    weight1 = self.conv1.weight * a2

    # Store for forward_fuse()
    self.weight1 = nn.Parameter(weight1)
    self.a2, self.a1, self.a0 = nn.Parameter(...)

def forward_fuse(self, x):
    # Fused forward: x² + a1*x + a0
    x = torch.square(x) + self.a1 * x + self.a0
    out = F.conv2d(x, self.weight1, ...)
    ...
    out += self.shortcut(x * self.a2)  # Scale shortcut
```

### **Orion Fusion**
```python
def init_orion_params(self):
    # Extract same BatchNorm stats
    bn0_mean_1 = self.bn0_1.running_mean
    bn0_var_1 = self.bn0_1.running_var
    ...

    # Create HerPN object with fused coefficients
    self.herpn1 = HerPN(
        bn0_mean_1, bn0_var_1,
        bn1_mean_1, bn1_var_1,
        bn2_mean_1, bn2_var_1,
        weight_1, bias_1, eps
    )

    # HerPN internally computes w2, w1, w0
    # and provides forward(x) = w2*x² + w1*x + w0

def forward(self, x):
    if self.he_mode:
        # Use fused HerPN
        out = self.herpn1(x)
        out = self.conv1(out)
        ...
```

**Key Difference**:
- ✗ CryptoFace: Manually fuses conv weights (`weight1 = self.conv1.weight * a2`)
- ✗ Orion: Keeps conv weights separate, applies HerPN first then conv
- ✓ Both compute same a2, a1, a0 coefficients
- ✓ Both use same fusion formula from paper

**Impact**: Weight loading will need to handle this difference!

---

## 5. Shortcut Connection Scaling

### **CryptoFace**
```python
# In forward_fuse():
out += self.shortcut(x * self.a2)  # Direct multiplication by a2
```

### **Orion**
```python
# Create ChannelSquare for scaling
self.shortcut_herpn = ChannelSquare(
    weight0=zeros,
    weight1=a2,   # Same a2 coefficient!
    weight2=None  # Only linear term
)

# In forward():
shortcut = self.shortcut_herpn(identity)  # Computes a2*x
shortcut = self.shortcut_conv(shortcut)
```

**Difference**: Implementation style, but **mathematically identical**!
- CryptoFace: `x * a2`
- Orion: ChannelSquare with `w1=a2, w0=0, w2=None` → outputs `a2*x`

---

## 6. Final Aggregation & Normalization

### **CryptoFace**
```python
def forward(self, x):
    # Process patches
    out = self._forward(x)  # Returns (B, N, dim)

    # Concatenate all patches
    out_global = rearrange(out_global, 'b n c -> b (n c)')

    # Single linear layer
    out_global = self.linear(out_global)  # (B, N*dim) → (B, 256)

    # BatchNorm normalization
    out_global = self.bn(out_global)

    return out_global, pred, target

def fuse(self):
    # Fuse BatchNorm into linear layer
    mean = self.bn.running_mean
    var = self.bn.running_var
    weight = self.linear.weight
    bias = self.linear.bias

    # Normalize weights
    weight = weight.T / √(var + e)
    bias = (bias - mean) / √(var + e)

    # Split for per-patch processing
    self.weights = nn.ParameterList(torch.chunk(weight, N, dim=1))
    self.bias = bias / N
```

### **Orion**
```python
def forward(self, x):
    # Process patches
    patches = [...]

    # Per-patch linear layers
    y_outputs = []
    for i in range(N):
        y_i = self.nets[i](patches[i])
        y_i = self.linear[i](y_i)  # Separate linear per patch!
        y_outputs.append(y_i)

    # Tree reduction (parallel sum)
    y = self._tree_reduce_add(y_outputs)

    # ChannelSquare normalization
    out = self.normalization(y)  # w1*y + w0 (or w2*y² + w1*y + w0)

    return out
```

**Major Difference**:
- ✗ **CryptoFace**: Single linear (N*dim → 256) + BatchNorm
- ✗ **Orion**: N separate linears (dim → 1024) + ChannelSquare
- ✗ **Output dim**: 256 vs 1024

**Impact on Weight Loading**:
- Cannot directly load CryptoFace linear weights into Orion!
- Need to split/reshape CryptoFace weights for per-patch linears
- Or modify Orion to use single linear like CryptoFace

---

## 7. Parallel Processing

### **CryptoFace**
```python
def _forward(self, x, fuse=False):
    # CUDA streams for parallel processing
    streams = [torch.cuda.Stream() for _ in range(N)]
    y = [None for _ in range(N)]

    for i in range(N):
        with torch.cuda.stream(streams[i]):
            y[i] = self.nets[i](x[i])
            if fuse:
                y[i] = y[i] @ self.weights[i].T + self.bias

    torch.cuda.synchronize()
    return torch.stack(y, dim=0)
```

### **Orion**
```python
def forward(self, x):
    if self.he_mode:
        # ThreadPoolExecutor for FHE operations
        def process_patch(i):
            y_i = self.nets[i](patches[i])
            y_i = self.linear[i](y_i)
            return y_i

        with ThreadPoolExecutor(max_workers=N) as executor:
            y_outputs = list(executor.map(process_patch, range(N)))

        # Tree reduction
        y = self._tree_reduce_add(y_outputs)
```

**Difference**:
- ✗ CryptoFace: CUDA streams (GPU parallelism)
- ✗ Orion: ThreadPoolExecutor or fork+serialization (CPU/FHE parallelism)

---

## 8. Implications for Weight Loading

### **Direct Compatibility** ✓
1. **Backbone weights**: Can be loaded directly
   - Conv layers: Same structure
   - BatchNorm stats: Same format (needed for fusion)
   - HerPN fusion uses same formula

2. **HerPNConv structure**: Same layers, same math
   - 5 HerPNConv blocks with identical channel progression
   - Shortcut connections use same a2 scaling

3. **HerPNPool**: Same structure
   - HerPN + AdaptiveAvgPool2d
   - Same fusion formula

### **Requires Adaptation** ✗

1. **Final Linear Layer**
   - **CryptoFace**: Single (N×256 → 256) linear
   - **Orion**: N separate (256 → 1024) linears
   - **Solution**:
     - Option A: Reshape CryptoFace weights to match Orion structure
     - Option B: Modify Orion to use single linear like CryptoFace

2. **Final Normalization**
   - **CryptoFace**: BatchNorm1d (affine=False)
   - **Orion**: ChannelSquare (quadratic)
   - **Solution**:
     - Extract BatchNorm stats from CryptoFace
     - Initialize ChannelSquare with equivalent transformation

3. **Output Dimensions**
   - **CryptoFace**: 256-dim embeddings
   - **Orion**: 1024-dim (4×256)
   - **Solution**: Adjust Orion linear layers to output 256 instead of 1024

---

## 9. Recommended Adaptation Strategy

### **Option A: Modify Orion to Match CryptoFace** (Easier)
```python
class PatchCNN_CryptoFace(on.Module):
    def __init__(self, input_size, patch_size, output_size=(2,2)):
        # Keep same Backbone (fully compatible!)
        self.nets = nn.ModuleList([Backbone(output_size) for _ in range(N)])

        # Use single linear like CryptoFace
        dim = output_size[0] * output_size[1] * 64
        self.linear = on.Linear(N * dim, 256)  # Match CryptoFace!

        # Use BatchNorm → ChannelSquare fusion for normalization
        # OR keep as BatchNorm1d for compatibility
        self.bn = on.BatchNorm1d(256, affine=False)
```

**Pros**:
- ✓ Can load weights directly
- ✓ Minimal changes to Orion
- ✓ Easier to verify correctness

**Cons**:
- ✗ Less FHE-friendly (single large linear vs per-patch linears)
- ✗ Harder to parallelize linear layer

### **Option B: Adapt CryptoFace Weights to Orion** (More Complex)
```python
def load_cryptoface_weights(cryptoface_checkpoint, orion_model):
    # Load backbone weights directly (compatible!)
    for i in range(N):
        load_backbone_weights(cryptoface.nets[i], orion_model.nets[i])

    # Reshape linear weights
    # CryptoFace: weight = (256, N*256), bias = (256,)
    # Orion: N × [(1024, 256), (1024,)]
    cryptoface_weight = checkpoint['linear.weight']  # (256, N*256)
    cryptoface_bias = checkpoint['linear.bias']      # (256,)

    # Split into N chunks
    weight_chunks = torch.chunk(cryptoface_weight, N, dim=1)  # N × (256, 256)

    for i in range(N):
        # Pad or project to 1024 dim
        orion_model.linear[i].weight = project_to_1024(weight_chunks[i])
        orion_model.linear[i].bias = cryptoface_bias / N  # Distribute bias
```

**Pros**:
- ✓ Keeps Orion FHE-friendly structure
- ✓ Better parallelization

**Cons**:
- ✗ Complex weight transformation
- ✗ May lose some accuracy due to reshaping

---

## 10. Action Items for Next Steps

### **Task 1.2: Create cryptoface_pcnn.py**
1. Choose adaptation strategy (recommend Option A for initial implementation)
2. Implement CryptoFace-compatible PatchCNN using Orion layers
3. Ensure backbone is identical (already is!)
4. Match final linear + normalization structure

### **Task 1.3: Implement weight_loader.py**
1. Load CryptoFace checkpoint (.pth file)
2. Map backbone weights (should work directly!)
3. Handle linear layer difference (choose strategy)
4. Handle BatchNorm → ChannelSquare normalization

### **Task 1.4: Verify Loading**
1. Load CryptoFaceNet4 (64×64) checkpoint
2. Compare cleartext outputs with CryptoFace reference
3. Verify numerical accuracy (tolerance < 1e-4)

---

## Conclusion

**Summary**:
- ✓ **Backbone**: Fully compatible! Same structure, same math
- ✓ **HerPN Fusion**: Same formula, different implementation style
- ✗ **Final Layers**: Different structure (linear + normalization)
- ✗ **Parallel**: Different mechanisms (CUDA vs ThreadPool/Fork)

**Recommendation**: Use **Option A** (modify Orion to match CryptoFace) for initial implementation. This minimizes complexity and ensures we can verify correctness against CryptoFace baseline.

Once verified, we can optionally implement Option B for better FHE performance if needed.
