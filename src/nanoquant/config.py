"""
Configuration class for NANOQUANT hyperparameters.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import torch
from .device_utils import get_optimal_device


@dataclass
class NanoQuantConfig:
    """Configuration for NANOQUANT quantization.
    
    Attributes:
        model_name: HuggingFace model name or path
        rank: Low-rank dimension for binary factorization
        bits: Target bit-width (1.0 for 1-bit, <1 for sub-1-bit)
        
        # Calibration
        calib_dataset: Calibration dataset name
        calib_samples: Number of calibration samples
        calib_seq_len: Sequence length for calibration
        
        # ADMM parameters
        admm_iterations: Number of ADMM iterations
        admm_rho: ADMM penalty parameter
        admm_lambda: Ridge regularization coefficient
        admm_epsilon: Convergence threshold
        
        # Shrinkage regularization
        shrinkage_gamma: Shrinkage coefficient for preconditioners
        shrinkage_tau: Clipping threshold for outlier removal
        
        # Refinement
        pre_tune_steps: Steps for error propagation mitigation (TUNEFP)
        post_tune_steps: Steps for factorized component refinement (TUNELATENT)
        glob_tune_steps: Steps for global scale tuning (TUNESCALES)
        
        # Block reconstruction
        block_size: Number of transformer layers per block
        
        # Evaluation
        eval_datasets: List of evaluation datasets
        eval_batch_size: Batch size for evaluation
        device: Device to use (cuda/cpu/mps)
        seed: Random seed
    """
    
    # Model
    model_name: str = "meta-llama/Llama-2-7b-hf"
    rank: int = 8
    bits: float = 1.0
    
    # Calibration
    calib_dataset: str = "wikitext"
    calib_config: str = "wikitext-2-raw-v1"
    calib_samples: int = 128
    calib_seq_len: int = 2048
    calib_batch_size: int = 1
    
    # ADMM parameters (from paper)
    admm_iterations: int = 50
    admm_rho: float = 1.0
    admm_lambda: float = 0.01
    admm_epsilon: float = 1e-5
    
    # Shrinkage regularization (from paper)
    shrinkage_gamma: float = 0.2  # 0.2 for Llama/Qwen, 0.6 for Gemma/Rnj
    shrinkage_tau: float = 10.0   # Clipping threshold
    
    # Block Reconstruction Pipeline parameters (from Appendix C of paper)
    # Phase 2 Step 1: TUNEFP (Error Propagation Mitigation)
    pre_tune_lr: float = 1e-5
    pre_tune_steps: int = 20

    # Phase 2 Step 3: TUNELATENT (Factorized Component Refinement)
    post_tune_lr: float = 1e-4
    post_tune_steps: int = 50

    # Phase 3: TUNESCALES (Global Scale Tuning)
    glob_tune_lr: float = 1e-3
    glob_tune_steps: int = 30
    
    # Block reconstruction (Phase 2 Algorithm 1)
    block_size: int = 1

    # Block I/O forward pass sample limit for memory efficiency
    # Default 8 is from paper; set to calib_samples/4 for large models
    block_io_samples: int = 8

    # Gradient calibration sample limit for preconditioner estimation
    # Asymmetric because gradient backprop is more expensive than forward
    grad_calib_samples: int = 32
    
    # Evaluation
    eval_datasets: List[str] = field(default_factory=lambda: ["wikitext"])
    eval_batch_size: int = 1
    
    # System
    device: str = "auto"  # Auto-detects: cuda > mps > cpu
    seed: int = 42
    output_dir: str = "./outputs"
    
    # === Nuovi parametri MoE (da QMoE - IST-DASLab) ===
    moe_enabled: bool = False              # Abilita modalità MoE
    quantize_only_experts: bool = False    # Quantizza solo gli expert
    tie_hessians: bool = True              # Riusa Hessiano per gate/up (risparmio memoria)
    expert_parallelism: bool = False       # Sharding expert su più GPU
    
    def __post_init__(self):
        """Auto-adjust parameters based on device for memory efficiency."""
        # Resolve device
        self.device = get_optimal_device(self.device)
        
        # Auto-adapt parameters based on model family
        self.adapt_for_model_family(self.model_name)
        
        # Apply MPS-specific optimizations for memory efficiency
        if self.device == "mps":
            # MPS has limited memory bandwidth, use conservative defaults
            if self.calib_samples > 64:
                self.calib_samples = 64
            if self.calib_seq_len > 1024:
                self.calib_seq_len = 1024
            if self.calib_batch_size > 2:
                self.calib_batch_size = 2
        
        # CPU has more RAM but slower computation
        elif self.device == "cpu":
            if self.calib_samples > 32:
                self.calib_samples = 32
            if self.calib_seq_len > 512:
                self.calib_seq_len = 512
    
    # Model family-specific settings
    @property
    def effective_bits(self) -> float:
        """Calculate effective bits per parameter."""
        raise NotImplementedError(
            "Use per-layer BPW from NanoQuantizer._log_compression_stats() "
            "which uses real layer dimensions. See paper eq. 43."
        )
    
    def adapt_for_model_family(self, model_name: str):
        """Adapt hyperparameters based on model family."""
        lower_name = model_name.lower()
        
        # Model family-specific shrinkage gamma
        if "gemma" in lower_name or "rnj" in lower_name:
            self.shrinkage_gamma = 0.6
        elif "llama" in lower_name or "qwen" in lower_name:
            self.shrinkage_gamma = 0.2
        
        # === Nuovo: MoE model detection (da QMoE) ===
        moe_models = ["mixtral", "deepseek", "switch", "qwen2-moe", "output-llm"]
        if any(moe in lower_name for moe in moe_models):
            self.moe_enabled = True
            if "deepseek" in lower_name:
                self.tie_hessians = True  # Ottimizzazione memoria per DeepSeek
        
        # Adjust rank based on model size
        if "70b" in lower_name:
            self.rank = min(self.rank, 16)
        elif "1b" in lower_name or "0.6b" in lower_name:
            self.rank = min(self.rank, 4)
