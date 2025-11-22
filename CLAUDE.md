# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Orion is a Fully Homomorphic Encryption (FHE) framework for deep learning inference on encrypted data using CKKS encryption. It supports running neural networks (ResNet, MobileNet, PCNN, etc.) entirely on encrypted data without decryption.

**Key Architecture Components:**
- **Backend**: Go-based Lattigo library (primary), with planned support for OpenFHE and HEAAN
- **Python Frontend**: PyTorch-based API for model definition and FHE operations
- **Level Management**: Automatic bootstrap placement and level assignment via DAG optimization
- **Custom Activations**: HerPN (Hermite Polynomial Network) activations for FHE-friendly nonlinear operations

## Environment Setup

**This project uses `uv` for package management, NOT conda.**

### Initial Setup
```bash
# Install dependencies using uv
uv sync

# Build the Go Lattigo backend (required before first use)
uv run python tools/build_lattigo.py

# Activate the virtual environment
source .venv/bin/activate
```

### Running Tests

**IMPORTANT: Always write test output to log files for long-running FHE tests.**

```bash
# Run a single test file (with log output)
uv run python tests/models/test_pcnn_backbone.py > logs/test_pcnn_backbone.log 2>&1

# Run all tests
uv run pytest tests/ > logs/pytest_all.log 2>&1

# Run specific test function
uv run pytest tests/models/test_mlp.py::test_mlp_fhe > logs/test_mlp.log 2>&1

# Read log files afterward
tail -100 logs/test_pcnn_backbone.log
grep -i "error\|panic\|success\|failed" logs/test_pcnn_backbone.log
```

**Why log files**: FHE tests can take several minutes. Logging allows you to:
- Run tests in background without timeout
- Read full output later
- Search for specific errors or success messages
- Keep a record of test runs

### Running Examples
```bash
# ResNet example (standard workflow)
uv run python examples/run_resnet.py

# Other examples
uv run python examples/run_mobilenet.py
uv run python examples/run_alexnet.py
```

## Core Workflow

### Standard FHE Inference Workflow
```python
import orion
import orion.models as models

# 1. Initialize scheme with config
orion.init_scheme("configs/resnet.yml")

# 2. Create and prepare model
model = models.ResNet20()
inp = torch.randn(1, 3, 32, 32)

# 3. Fit: trace model and collect statistics
orion.fit(model, inp)

# 4. Compile: assign levels and generate FHE parameters
input_level = orion.compile(model)

# 5. Encode, encrypt, and run FHE inference
vec_ptxt = orion.encode(inp, input_level)
vec_ctxt = orion.encrypt(vec_ptxt)
model.he()  # Switch to FHE mode
out_ctxt = model(vec_ctxt)
out_fhe = out_ctxt.decrypt().decode()
```

### Custom Fit Workflow (for PCNN/HerPN models)

**CRITICAL**: Models with post-initialization fusion (like PCNN with HerPN) require a different workflow. See `docs/CUSTOM_FIT_WORKFLOW.md` for full details.

```python
# STEP 1: Collect BatchNorm statistics
model.eval()
with torch.no_grad():
    for _ in range(20):
        _ = model(torch.randn(4, 3, 32, 32))

# STEP 2: Fuse operations BEFORE orion.fit()
model.init_orion_params()

# STEP 3: Now fit and compile
inp = torch.randn(1, 3, 32, 32)
orion.fit(model, inp)
input_level = orion.compile(model)
```

**Why**: Fusion must happen before tracing, otherwise the tracer captures the unfused graph with incorrect depth calculations, leading to level assignment errors.

## Configuration Files

Configuration files (`configs/*.yml`) define CKKS parameters for different models:

### Key Parameters
```yaml
ckks_params:
  LogN: 16              # Polynomial degree: 2^16 = 65536
  LogQ: [51, 46, ...]   # Modulus chain (each entry = one level)
  LogP: [51, 51, ...]   # Bootstrapping moduli
  LogScale: 46          # Scale factor: 2^46
  H: 192                # Hamming weight for secret key

orion:
  margin: 2                      # Bootstrap scaling range (usually 2)
  bootstrap_placement_margin: 0  # Minimum output level constraint
  embedding_method: hybrid       # [hybrid, square]
  backend: lattigo              # [lattigo, openfhe, heaan]
```

### Bootstrap Placement Margin
- **Purpose**: Prevents bootstrap operations from being placed at levels where Q/Scale ratio is too low
- **When to use**: Deep networks with infrequent bootstraps (e.g., PCNN with 35 levels, 1 bootstrap)
- **Default**: 0 for ResNet (frequent bootstraps), 2-3 for PCNN (rare bootstraps)
- **Details**: See `BOOTSTRAP_PLACEMENT_MARGIN_EXPLANATION.md`

### Example Configs
- `configs/resnet.yml`: 11 levels, frequent bootstrapping
- `configs/pcnn.yml` / `configs/pcnn_backbone.yml`: 35 levels, strategic bootstrap placement

## Architecture Deep Dive

### Core Pipeline (`orion/core/`)
1. **`orion.py`**: Main Scheme class, entry point for fit/compile
2. **`tracer.py`**: Traces PyTorch model to build computation graph
3. **`network_dag.py`**: Builds directed acyclic graph of operations
4. **`level_dag.py`**: Creates level assignment graph, runs Dijkstra's algorithm
5. **`auto_bootstrap.py`**: Bootstrap solver, places bootstraps optimally
6. **`fuser.py`**: Fuses batch normalization into convolutions

### Backend (`orion/backend/`)
- **`lattigo/`**: Go implementation using Lattigo library (compile with `build_lattigo.py`)
  - `evaluator.go`: Homomorphic operations (Add, Mul, etc.)
  - `bootstrapper.go`: Bootstrap implementation
  - `lineartransform.go`: Convolution/linear layer operations
- **`python/`**: Python wrappers and fallback implementations

### Neural Network Layers (`orion/nn/`)
- **`module.py`**: Base Module class (extends torch.nn.Module)
- **`operations.py`**: FHE operations (Add, Mul, Bootstrap, etc.)
- **`activation.py`**: HerPN and other FHE-friendly activations
- **`linear.py`**: Linear transforms and convolutions
- **`pooling.py`**: AdaptiveAvgPool2d, MaxPool (approximate)

### Models (`models/` and `orion/models/`)
- **`pcnn.py`**: PatchCNN with HerPN activations (custom fit workflow required)
  - `Backbone`: Feature extraction backbone
  - `PatchCNN`: Full patch-based CNN
  - `HerPNConv`: Residual block with HerPN activations
  - `HerPNPool`: HerPN + adaptive pooling
- **`resnet.py`**: ResNet variants (ResNet20, ResNet32, etc.)
- Other: AlexNet, MobileNet, VGG, YOLO, etc.

## Common Development Tasks

### Adding a New Model
1. Create model in `models/your_model.py` using `orion.nn` layers
2. Create config file `configs/your_model.yml` with appropriate CKKS parameters
3. Create test in `tests/models/test_your_model.py`
4. Follow standard or custom fit workflow depending on architecture

### Debugging Level Assignment Issues
```python
# Enable debug mode in config
orion:
  debug: true

# Check traced graph structure
from orion.core import scheme
for name, module in scheme.trace.named_modules():
    if name:
        level = getattr(module, 'level', 'N/A')
        depth = getattr(module, 'depth', 'N/A')
        print(f"{name:40s} level={level:>3}, depth={depth}")
```

### Verifying Shortcut Alignment (Residual Networks)
```python
# Both paths in a residual block must end at the same level
main_level = model.layer2.conv2.level
shortcut_level = model.layer2.shortcut_conv.level
assert main_level == shortcut_level
```

### Common Errors

**"panic: level cannot be larger than max level"**
- Cause: Requested input level exceeds available levels in LogQ
- Solution: Reduce required depth or increase LogQ entries in config

**"cannot BootstrapMany: initial Q/Scale < threshold"**
- Cause: Bootstrap placed at level with insufficient Q/Scale ratio
- Solution: Increase `bootstrap_placement_margin` in config (try 2-3)

**Shortcut level mismatch in residual blocks**
- Cause: Traced unfused graph instead of fused graph
- Solution: Use custom fit workflow - call `init_orion_params()` before `orion.fit()`

## Important Implementation Details

### Level Consumption
- **Multiplication**: Consumes 1 level (requires rescaling)
- **Addition**: No level consumption (if scales match)
- **Convolution**: 1 level (single multiplication depth)
- **HerPN (quadratic)**: 2 levels (x² term)
- **Bootstrap**: Input at level L → Output at max level

### HerPN Activation
HerPN approximates activation functions using Hermite polynomials:
- Fuses 3 BatchNorm layers into a single quadratic operation
- Form: `w2·x² + w1·x + w0`
- Depth: 2 (one for x², one for addition after rescaling)

### Packing and Slots
- CKKS packing packs multiple values into polynomial slots
- Slot count: `poly_modulus_degree / 2` (e.g., 32768 for LogN=16)
- Enables SIMD-style parallel computation on encrypted data

## Key Documentation Files

- `BOOTSTRAP_PLACEMENT_MARGIN_EXPLANATION.md`: Deep dive on bootstrap placement constraints
- `docs/CUSTOM_FIT_WORKFLOW.md`: Required workflow for PCNN and similar models
- `docs/PCNN_LEVEL_ISSUE_FIX.md`: Historical context on PCNN level assignment fixes
- `ISSUES.md`: Current development tasks and known issues

## Testing Strategy

Tests are organized by model type in `tests/models/`:
- Each test follows the fit → compile → encrypt → infer → decrypt workflow
- Tests verify MAE (Mean Absolute Error) between cleartext and FHE outputs
- Tolerance typically: MAE < 1.0 for successful FHE inference

## CryptoFace Integration

The `CryptoFace/` directory contains reference implementations:
- Original SEAL-based CNN implementation
- Parameter configurations for comparison (typically 16 remaining + 14 bootstrap = 30 levels)
- Useful for validating level requirements and performance benchmarks
