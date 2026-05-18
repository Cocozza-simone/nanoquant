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
        pre_tune_steps: Steps for error propagation mitigation
        post_tune_steps: Steps for factorized component refinement
        glob_tune_steps: Steps for global scale tuning
        
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
    # "We used a learning rate of 1e-4 and a batch size of 4"
    tune_fp_epochs: int = 8
    tune_fp_lr: float = 1e-4
    tune_fp_batch_size: int = 4
    
    # Phase 2 Step 3: TUNELATENT (Factorized Component Refinement)
    # "We used a unified learning rate of 1e-5 and a batch size of 1"
    tune_latent_epochs: int = 8
    tune_latent_lr: float = 1e-5
    tune_latent_batch_size: int = 1
    
    # Phase 2 Step 2: TUNESCALES (Global Scale Tuning)
    tune_scales_epochs: int = 8
    tune_scales_lr: float = 1e-6
    tune_scales_batch_size: int = 1
    
    # Scheduler
    use_cosine_scheduler: bool = True
    pre_tune_lr: float = 1e-5
    post_tune_lr: float = 1e-4
    glob_tune_lr: float = 1e-3
    
    # Refinement steps (legacy support)
    pre_tune_steps: int = 20
    post_tune_steps: int = 50
    glob_tune_steps: int = 30
    
    # Block reconstruction
    block_size: int = 1  # Process one transformer block at a time
    
    # Evaluation
    eval_datasets: List[str] = field(default_factory=lambda: ["wikitext"])
    eval_batch_size: int = 1
    
    # System
    device: str = "auto"  # Auto-detects: cuda > mps > cpu
    seed: int = 42
    output_dir: str = "./outputs"
    
    def __post_init__(self):
        """Auto-adjust parameters based on device for memory efficiency."""
        # Resolve device
        self.device = get_optimal_device(self.device)
        
        # Apply MPS-specific optimizations for memory efficiency
        if self.device == "mps":
            # MPS has limited memory bandwidth, use conservative defaults
            if self.calib_samples > 64:
                self.calib_samples = 64
            if self.calib_seq_len > 1024:
                self.calib_seq_len = 1024
            if self.tune_fp_batch_size > 2:
                self.tune_fp_batch_size = 2
        
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
        # For rank-r factorization with 2 scales per layer:
        # bits = (2 * r * (din + dout) + din + dout) / (din * dout)
        # This is approximately 2*r / max(din, dout) for large matrices
        return self.bits
    
    def adapt_for_model_family(self, model_name: str):
        """Adapt hyperparameters based on model family."""
        lower_name = model_name.lower()
        if "gemma" in lower_name or "rnj" in lower_name:
            self.shrinkage_gamma = 0.6
        elif "llama" in lower_name or "qwen" in lower_name:
            self.shrinkage_gamma = 0.2
        # Adjust rank based on model size
        if "70b" in lower_name:
            self.rank = min(self.rank, 16)
        elif "1b" in lower_name or "0.6b" in lower_name:
            self.rank = min(self.rank, 4)
