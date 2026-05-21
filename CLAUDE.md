# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NANOQUANT is a post-training quantization (PTQ) method that compresses LLMs to sub-1-bit per parameter using low-rank binary factorization: `W ≈ s1 ⊙ (U_{±1} V_{±1}^T) ⊙ s2^T` (where `⊙` is Hadamard product). It uses 128 calibration samples and does not require retraining.

- **Language**: Python 3.9+
- **Framework**: PyTorch 2.0+, HuggingFace Transformers
- **Version**: 0.2.0 (per `__init__.py`)
- **License**: MIT

## Common Commands

### Development Setup
```bash
pip install -e ".[dev]"  # Install package + dev deps (pytest, black, flake8)
```

### Testing
```bash
pytest tests/ -v                              # Run all tests
pytest tests/test_nanoquant.py -k TestADMM    # Run specific test class
python test_integration_v0_2_0.py             # Integration tests for v0.2.0
```

### Running Scripts
```bash
# Demo (no HF auth needed): gpt2, tinyllama, opt
python scripts/quantize_demo.py --model tinyllama --rank 4

# Full quantization
python scripts/quantize.py --model meta-llama/Llama-2-7b-hf --rank 8

# Evaluation
python scripts/evaluate.py --model <model> --quantized-path <path>

# MPS-optimized (macOS)
python scripts/quantize_mps_optimized.py
```

### Code Quality
```bash
black src/              # line-length=100, target py39 (see pyproject.toml)
flake8 src/
```

## Architecture Overview

### Three-Phase Algorithm (Algorithm 1)
The `NanoQuantizer` (`quantization.py`) orchestrates the full pipeline:

1. **Global Calibration** (`calibration.py`): Computes Hessian-aware preconditioners `D_in`, `D_out` for each linear layer using K-FAC approximation with shrinkage regularization. Runs a forward/backward pass on calibration data to collect activations and gradients.
2. **Block Reconstruction** (`reconstruction.py`): Processes each transformer block sequentially with a 3-step optimization:
   - **Step 1** (`error_mitigation.py`): TUNEFP — tunes full-precision weights to mitigate error propagation from previously-quantized blocks using Adam with cosine scheduler (paper: 8 epochs, lr=1e-4, batch=4).
   - **Step 2** (`admm.py`): LB-ADMM — solves `min ||W_f - UV^T||_F^2` via Cholesky decomposition (O(r^3/3)) with SVID-based proxy updates. Supports optional ternary initialization (`use_ternary_init=True`) from QMoE for faster convergence.
   - **Step 3** (`refinement.py`): TUNELATENTSTE — refines factorized components using Straight-Through Estimator (STE) on `U_latent`, `V_latent`, and scales with Adam cosine (paper: 8 epochs, lr=1e-5, batch=1).
3. **Model Reconstruction** (`model_reconstruction.py`): TUNESCALESKD — global scale tuning via KL divergence between original and quantized model logits (paper: 8 epochs, lr=1e-6, batch=1).

### Key Data Structures

- **`NanoQuantConfig`** (`config.py`): Central hyperparameter dataclass. Auto-adapts to model families: `shrinkage_gamma=0.2` for Llama/Qwen, `0.6` for Gemma/Rnj. Also auto-detects MoE models.
- **`FactorizedLinear`** (`reconstruction.py`): Core layer implementing the factorized math. Fields: `U_latent`, `V_latent`, `s1`, `s2`, `U_binary`, `V_binary`. `pack()` freezes to binary and sets `packed=True`.
- **`LatentBinaryADMM`** (`admm.py`): Solver instance with `solve(W_f)` returning `(U, V, s1, s2)`. Uses ternary init when `use_ternary_init=True`.

### Module Map

| Module | Role |
|--------|------|
| `calibration.py` | `GlobalCalibration.compute_preconditioners()` → `(D_in, D_out)` |
| `admm.py` | `LatentBinaryADMM` — binary factorization solver |
| `reconstruction.py` | `BlockReconstructionPipeline`, `FactorizedLinear`, `ModelReconstruction` |
| `error_mitigation.py` | `tune_full_precision_weights()` — Step 1 of block reconstruction |
| `refinement.py` | `StraightThroughEstimator`, `tune_latent_ste()` — Step 3 |
| `model_reconstruction.py` | `tune_scales_kd()` — Phase 3 global tuning |
| `svid.py` | `svid_decompose()` — proxy projection for ADMM update |
| `packing.py` | `pack_binary_tensor()`, `PackedBinaryStorage` — bit-packing {-1,+1} to uint8 |
| `kernels.py` | `OptimizedFactorizedLinear`, `binary_gemv_simple()` — fast inference kernels |
| `inference.py` | `NanoQuantInferenceEngine` — load, generate, benchmark |
| `evaluation.py` | `evaluate_perplexity()`, `evaluate_zero_shot()` — wikitext, hellaswag, etc. |
| `device_utils.py` | `get_optimal_device()` — priority: CUDA > MPS > CPU |
| **v0.2.0 Additions** | |
| `moe_quantization.py` | `MoEExpertQuantizer` — Mixture-of-Experts support |
| `ternary_init.py` | `ternary_svd_init()` — sparse init for ADMM (from QMoE) |
| `group_scale.py` | `apply_group_scaling()` — per-128-weight FP16 scales (from OxiBonsai) |
| `gguf_export.py` | `export_to_gguf()` — Q1_0_g128 format export (from OxiBonsai) |

### Platform-Specific Notes

- **Device**: `get_optimal_device("auto")` picks CUDA > MPS > CPU. Config auto-downgrades `calib_samples` and `calib_seq_len` on MPS/CPU to avoid OOM.
- **MPS (macOS)**: Calibration hooks move tensors to CPU to save MPS memory. Use `scripts/quantize_mps_optimized.py` for MPS-specific memory handling.
- **Model loading**: Always uses `trust_remote_code=True`, `torch_dtype=torch.float16`, `device_map="auto"`, and `low_cpu_mem_usage=True`.

### Testing Conventions

- All tests in `tests/test_nanoquant.py` using `pytest`.
- Classes match module names: `TestSVID`, `TestADMM`, `TestCalibration`, `TestErrorMitigation`, `TestRefinement`, `TestModelReconstruction`, `TestPacking`, `TestKernels`, `TestConfig`, `TestReconstruction`.
- Integration: `test_integration_v0_2_0.py` covers new v0.2.0 imports and features.
