"""
NANOQUANT: Efficient Sub-1-Bit Quantization of Large Language Models

A post-training quantization (PTQ) method capable of compressing LLMs
to binary (1-bit) and sub-1-bit levels using low-rank binary factorization.

Integrations:
- QMoE (IST-DASLab): Support for Mixture-of-Experts models
- OxiBonsai (COOLJAPAN): GGUF Q1_0_g128 export for ultra-fast inference
"""

__version__ = "0.2.0"

from .config import NanoQuantConfig
from .device_utils import get_optimal_device, get_device_info, move_to_device
from .calibration import GlobalCalibration
from .admm import LatentBinaryADMM
from .reconstruction import BlockReconstructionPipeline, ModelReconstruction
from .quantization import NanoQuantizer
from .evaluation import evaluate_perplexity, evaluate_zero_shot
from .packing import (
    pack_binary_tensor,
    unpack_binary_tensor,
    pack_binary_matrix,
    unpack_binary_matrix,
    PackedBinaryStorage,
)
from .ternary_init import (
    ternary_project,
    ternary_svd_init,
    estimate_init_quality,
)
from .group_scale import (
    apply_group_scaling,
    reconstruct_from_group_scales,
    GroupScaledWeights,
    memory_stats,
    GROUP_SIZE,
)
from .kernels import (
    binary_gemv_simple,
    OptimizedFactorizedLinear,
    create_optimized_linear_from_factorized,
)
from .inference import (
    NanoQuantInferenceEngine,
    create_inference_engine,
)
from .svid import (
    svid_decompose,
    svid_rank1_fast,
    project_to_binary_low_rank,
)
from .error_mitigation import (
    tune_full_precision_weights,
    weighted_mse_loss,
    compute_weighted_reconstruction_error,
)
from .refinement import (
    StraightThroughEstimator,
    tune_latent_ste,
    tune_latent_simple,
)
from .model_reconstruction import (
    tune_scales_kd,
    kl_divergence_loss,
    tune_scales_simple,
)
# === Integrazione QMoE (da IST-DASLab) ===
from .moe_quantization import MoEExpertQuantizer
# === Integrazione OxiBonsai (da COOLJAPAN) ===
from .gguf_export import (
    export_to_gguf,
    pack_nanoquant_to_q1_0_g128,
    load_gguf_metadata,
)

__all__ = [
    "NanoQuantConfig",
    "GlobalCalibration",
    "LatentBinaryADMM",
    "BlockReconstructionPipeline",
    "ModelReconstruction",
    "NanoQuantizer",
    "evaluate_perplexity",
    "evaluate_zero_shot",
    "pack_binary_tensor",
    "unpack_binary_tensor",
    "pack_binary_matrix",
    "unpack_binary_matrix",
    "PackedBinaryStorage",
    # ternary_init
    "ternary_project",
    "ternary_svd_init",
    "estimate_init_quality",
    # group_scale
    "apply_group_scaling",
    "reconstruct_from_group_scales",
    "GroupScaledWeights",
    "memory_stats",
    "GROUP_SIZE",
    "binary_gemv_simple",
    "OptimizedFactorizedLinear",
    "create_optimized_linear_from_factorized",
    "NanoQuantInferenceEngine",
    "create_inference_engine",
    "svid_decompose",
    "svid_rank1_fast",
    "project_to_binary_low_rank",
    "tune_full_precision_weights",
    "weighted_mse_loss",
    "compute_weighted_reconstruction_error",
    "StraightThroughEstimator",
    "tune_latent_ste",
    "tune_latent_simple",
    "tune_scales_kd",
    "kl_divergence_loss",
    "tune_scales_simple",
    # === QMoE Integration ===
    "MoEExpertQuantizer",
    # === OxiBonsai Integration ===
    "export_to_gguf",
    "pack_nanoquant_to_q1_0_g128",
    "load_gguf_metadata",
]
