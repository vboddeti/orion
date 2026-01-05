# Encrypted Face Recognition with Orion - Implementation Plan

## Overview

Implement end-to-end encrypted face recognition using PCNN with Orion FHE (Lattigo/CKKS backend), adapting the CryptoFace system to leverage our parallel inference capabilities.

**Goal**: Achieve encrypted face verification on standard benchmarks with parallel speedup.

---

## Project Structure

```
face_recognition/
├── PLAN.md                    # This file
├── __init__.py
├── models/                    # Model definitions and weight loading
│   ├── __init__.py
│   ├── cryptoface_pcnn.py    # Adapted PCNN model for Orion
│   ├── weight_loader.py      # Load CryptoFace pretrained weights
│   └── model_utils.py        # Model utilities (fusion, etc.)
├── datasets/                  # Dataset loaders and preprocessing
│   ├── __init__.py
│   ├── face_datasets.py      # Face dataset loaders (lfw, cfp_fp, etc.)
│   ├── data_utils.py         # Data preprocessing utilities
│   └── download.py           # Dataset download scripts
├── evaluation/                # Evaluation protocols and metrics
│   ├── __init__.py
│   ├── verification.py       # Face verification protocol
│   └── metrics.py            # Accuracy, EER, AUC metrics
└── experiments/               # Main experiment scripts
    ├── __init__.py
    ├── eval_cleartext.py     # Baseline cleartext evaluation
    ├── eval_encrypted.py     # Encrypted face recognition
    └── benchmark_fr.py       # Benchmark sequential vs parallel
```

---

## Implementation Phases

### **Phase 1: Model Setup & Weight Loading**

**Goal**: Adapt CryptoFace PCNN to Orion and load pretrained weights.

#### Tasks:
- [ ] 1.1. Study CryptoFace PCNN architecture (`CryptoFace/models/pcnn.py`)
  - Compare with Orion PCNN (`models/pcnn.py`)
  - Identify differences in layer structure, fusion, normalization

- [ ] 1.2. Create `models/cryptoface_pcnn.py`
  - Implement CryptoFace-compatible PCNN using Orion layers
  - Match architecture: 4,9 or 16 patches (2×2, 3x3, or 4x4) chose through config, Backbone with HerPNConv
  - Include fusion methods for BatchNorm + HerPN
  - create 4 config files for CryptoFaceNet4, CryptoFaceNet9, CryptoFaceNet16. include the FHE parameters too in the config

- [ ] 1.3. Implement `models/weight_loader.py`
  - Load CryptoFace checkpoint files (PyTorch .pth)
  - Map weights to Orion model structure
  - Handle BatchNorm statistics and fusion

- [ ] 1.4. Create test script: `experiments/test_model_load.py`
  - Load CryptoFaceNet4 weights (64×64)
  - Run cleartext forward pass
  - Compare with CryptoFace reference outputs

- [ ] 1.5. Verify weight loading correctness
  - Generate embeddings on sample images
  - Compare with CryptoFace embeddings (cosine similarity)
  - Ensure numerical accuracy (tolerance < 1e-4)

**Deliverables**:
- ✓ Working CryptoFace-compatible PCNN model in Orion
- ✓ Weight loader that correctly loads pretrained checkpoints
- ✓ Test script validating cleartext inference matches CryptoFace

**Estimated Time**: 2-3 days

---

### **Phase 2: Dataset Integration**

**Goal**: Setup face verification datasets and evaluation protocol.

#### Tasks:
- [ ] 2.1. Download InsightFace datasets
  - lfw.bin
  - cfp_fp.bin
  - cplfw.bin
  - agedb_30.bin
  - calfw.bin
  - Store in `data/face_recognition/`

- [ ] 2.2. Create `datasets/face_datasets.py`
  - Adapt `CryptoFace/datasets.py` for Orion
  - Implement FaceBinDataset class
  - Support 64×64 input resolution (matching CryptoFaceNet4)
  - Return image pairs + labels (same/different person)

- [ ] 2.3. Implement `datasets/data_utils.py`
  - Image preprocessing (normalize to [-1, 1])
  - Patch extraction (2×2 patches from 64×64 image)
  - Data augmentation (if needed for training)

- [ ] 2.4. Create `evaluation/verification.py`
  - Implement face verification protocol
  - Compute embeddings for image pairs
  - Calculate cosine similarity
  - Apply 10-fold cross-validation

- [ ] 2.5. Implement `evaluation/metrics.py`
  - Accuracy @ different thresholds
  - Equal Error Rate (EER)
  - True Positive Rate @ False Positive Rate
  - ROC curve plotting

- [ ] 2.6. Create `experiments/eval_cleartext.py`
  - Run cleartext face verification on all 5 datasets
  - Report accuracy, EER for each dataset
  - Establish baseline performance

**Deliverables**:
- ✓ All 5 datasets downloaded and accessible
- ✓ Dataset loaders working with Orion preprocessing
- ✓ Cleartext evaluation achieving expected accuracy
- ✓ Baseline metrics for comparison

**Estimated Time**: 1-2 days

---

### **Phase 3: Encrypted Inference**

**Goal**: Implement end-to-end encrypted face verification with Orion FHE.

#### Tasks:
- [ ] 3.1. Setup CKKS parameters for face recognition
  - Create `configs/face_recognition.yml`
  - Determine required levels for PCNN (from CryptoFace paper)
  - Set LogQ, LogP, bootstrap placement margin
  - Test compilation with sample input

- [ ] 3.2. Implement encrypted embedding extraction
  - Encrypt input image patches
  - Run FHE inference through PCNN
  - Get encrypted embedding vector
  - Implement encrypted normalization (if needed)

- [ ] 3.3. Implement encrypted cosine similarity
  - Option A: Decrypt embeddings, compute similarity in cleartext
  - Option B: Encrypted dot product (if feasible)
  - Choose based on security requirements vs. performance

- [ ] 3.4. Create `experiments/eval_encrypted.py`
  - Load pretrained CryptoFaceNet4
  - Compile for FHE with Orion
  - Run encrypted face verification on subset of data
  - Verify accuracy matches cleartext (within FHE error tolerance)

- [ ] 3.5. Integrate parallel inference
  - Reuse `benchmarks/utils.py` utilities
  - Apply fork + serialization to PCNN inference
  - Process 4 patches in parallel per image
  - Measure speedup vs sequential

**Deliverables**:
- ✓ Working encrypted face verification pipeline
- ✓ Accuracy comparable to cleartext (within tolerance)
- ✓ Parallel inference integrated and functional
- ✓ End-to-end encryption maintained

**Estimated Time**: 3-4 days

---

### **Phase 4: Benchmarking & Optimization**

**Goal**: Comprehensive benchmarking and performance optimization.

#### Tasks:
- [ ] 4.1. Create `experiments/benchmark_fr.py`
  - Benchmark sequential encrypted inference
  - Benchmark parallel encrypted inference
  - Measure: latency, memory, accuracy
  - Compare against CryptoFace/SEAL results

- [ ] 4.2. Run full evaluation on all 5 datasets
  - lfw (6000 pairs)
  - cfp_fp (7000 pairs)
  - cplfw (6000 pairs)
  - agedb_30 (6000 pairs)
  - calfw (6000 pairs)
  - Report accuracy, EER for each

- [ ] 4.3. Performance analysis
  - Single image inference time (sequential vs parallel)
  - Throughput (images per second)
  - Memory usage (encrypted data + model)
  - Speedup factor vs sequential

- [ ] 4.4. Comparison with CryptoFace/SEAL
  - Accuracy: Orion/Lattigo vs CryptoFace/SEAL
  - Latency: Compare inference times
  - Implementation differences analysis

- [ ] 4.5. Documentation
  - Write comprehensive README for face_recognition/
  - Document configuration parameters
  - Provide usage examples
  - Record results and findings

**Deliverables**:
- ✓ Complete benchmark results (sequential + parallel)
- ✓ Accuracy results on all 5 datasets
- ✓ Performance comparison with CryptoFace
- ✓ Documentation and usage guide

**Estimated Time**: 2-3 days

---

## Configuration & Dependencies

### CKKS Parameters (Initial Estimate)
Based on CryptoFace paper, estimated parameters for 64×64 input:

```yaml
ckks_params:
  LogN: 16              # Polynomial degree
  LogQ: [51, 46, ...]   # ~30-35 levels (estimate)
  LogP: [51, 51, ...]   # Bootstrap moduli
  LogScale: 46          # Scale factor
  H: 192                # Hamming weight

orion:
  margin: 2
  bootstrap_placement_margin: 2  # Similar to PCNN
  embedding_method: hybrid
  backend: lattigo
```

### Model Specifications

| Model | Input | Patches | Backbone Layers | Embedding Dim | Parameters |
|-------|-------|---------|-----------------|---------------|------------|
| CryptoFaceNet4 | 64×64 | 2×2 (32×32 each) | 5 HerPNConv | 256 | 0.94M |

### Dataset Specifications

| Dataset | Pairs | Protocol | Metric |
|---------|-------|----------|--------|
| lfw | 6000 | 10-fold CV | Accuracy, EER |
| cfp_fp | 7000 | 10-fold CV | Accuracy, EER |
| cplfw | 6000 | 10-fold CV | Accuracy, EER |
| agedb_30 | 6000 | 10-fold CV | Accuracy, EER |
| calfw | 6000 | 10-fold CV | Accuracy, EER |

---

## Key Decisions to Make

### 1. **Checkpoint Source**
- [ ] Option A: Use CryptoFace pretrained weights (from Google Drive)
- [ ] Option B: Train our own model with Orion-compatible fusion
- **Recommendation**: Start with Option A (pretrained) for faster iteration

### 2. **Embedding Comparison**
- [ ] Option A: Decrypt embeddings, compare in cleartext (simpler, faster)
- [ ] Option B: Encrypted cosine similarity (more secure, but complex)
- **Recommendation**: Start with Option A, add Option B if needed

### 3. **Model Size**
- [ ] Option A: Start with CryptoFaceNet4 (64×64, fastest)
- [ ] Option B: Test all 3 sizes (64×64, 96×96, 128×128)
- **Recommendation**: Start with 64×64, expand if time permits

### 4. **Dataset Storage**
- [ ] Path: `/research/hal-vishnu/data/face_recognition/` or project-local?
- **Decision needed**: Where to store ~10GB of face datasets?

### 5. **Parallel Strategy**
- [ ] Per-image parallelism (process 4 patches per image in parallel)
- [ ] Batch parallelism (process multiple images in parallel)
- [ ] Both (parallel patches + batch processing)
- **Recommendation**: Start with per-image (4 patches), proven to work

---

## Success Criteria

### Minimum Viable Product (MVP)
- ✓ Load CryptoFaceNet4 weights into Orion PCNN
- ✓ Run encrypted inference on at least 1 dataset (lfw)
- ✓ Achieve accuracy within 1-2% of cleartext
- ✓ Demonstrate parallel speedup (>1.5x vs sequential)

### Full Success
- ✓ All 5 datasets evaluated with encrypted inference
- ✓ Accuracy matches CryptoFace/cleartext baseline
- ✓ Parallel speedup >2x on encrypted inference
- ✓ Complete benchmarks and documentation
- ✓ Reproducible results with clear instructions

---

## Timeline Estimate

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Model Setup | 2-3 days | CryptoFace checkpoints |
| Phase 2: Datasets | 1-2 days | InsightFace access |
| Phase 3: Encrypted Inference | 3-4 days | Phase 1 complete |
| Phase 4: Benchmarking | 2-3 days | Phase 3 complete |
| **Total** | **8-12 days** | |

---

## Notes & Open Questions

### Questions for Discussion
1. Do we have access to InsightFace datasets already?
2. Where should we store pretrained checkpoints?
3. Should we target matching CryptoFace accuracy exactly, or is within tolerance acceptable?
4. Do we need encrypted similarity, or is decrypting embeddings acceptable?
5. Should we implement training code, or only inference?

### Known Challenges
- Weight loading compatibility between CryptoFace and Orion BatchNorm fusion
- CKKS parameter tuning for optimal accuracy vs latency
- Dataset download may require InsightFace registration
- Ensuring numerical accuracy matches CryptoFace baseline

### Future Extensions
- Multi-resolution support (96×96, 128×128)
- Encrypted gallery search (1:N matching)
- Training with Orion-aware fusion
- GPU-accelerated FHE (when available)
- Real-time encrypted face recognition demo

---

## References

- CryptoFace Paper: CVPR 2025
- CryptoFace Code: `/research/hal-vishnu/code/orion-fhe/CryptoFace/`
- Orion PCNN: `/research/hal-vishnu/code/orion-fhe/models/pcnn.py`
- Parallel Inference: `/research/hal-vishnu/code/orion-fhe/benchmarks/`
- InsightFace: https://github.com/deepinsight/insightface

---

**Last Updated**: 2025-11-22
**Status**: Planning Phase - Awaiting Approval
