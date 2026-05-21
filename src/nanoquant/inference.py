"""
Optimized inference engine for NANOQUANT quantized models.

Provides efficient model loading and inference with binary weights,
including support for consumer hardware deployment.
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from .device_utils import get_optimal_device
from .reconstruction import FactorizedLinear
from .kernels import OptimizedFactorizedLinear

logger = logging.getLogger(__name__)


class NanoQuantInferenceEngine:
    """Inference engine for NANOQUANT quantized models.
    
    Enables efficient inference on both datacenter and consumer hardware
    by leveraging optimized binary kernels and packed storage.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        use_optimized_kernels: bool = True,
        pack_weights: bool = True,
    ):
        """Initialize inference engine.
        
        Args:
            model_path: Path to quantized model
            device: Device for inference
            use_optimized_kernels: Use optimized binary kernels
            pack_weights: Use packed binary storage
        """
        self.model_path = model_path
        self.device = device
        self.use_optimized_kernels = use_optimized_kernels
        self.pack_weights = pack_weights
        
        self.model = None
        self.tokenizer = None
        self._loaded = False
    
    def load(self):
        """Load quantized model for inference."""
        from .quantization import NanoQuantizer
        
        logger.info(f"Loading quantized model from {self.model_path}")
        
        quantizer = NanoQuantizer.load_quantized_model(
            self.model_path,
            config=None,
        )
        
        self.model = quantizer.quantized_model
        self.tokenizer = quantizer.tokenizer
        self._loaded = True
        
        # Optionally optimize with packed kernels
        if self.pack_weights:
            self._optimize_for_inference()
        
        logger.info("Model loaded and ready for inference")
    
    def _optimize_for_inference(self):
        """Optimize model for inference."""
        from .kernels import create_optimized_linear_from_factorized
        
        logger.info("Optimizing for inference with packed weights...")
        
        # Replace FactorizedLinear with OptimizedFactorizedLinear
        for name, module in self.model.named_modules():
            if isinstance(module, (FactorizedLinear, OptimizedFactorizedLinear)):
                try:
                    optimized = create_optimized_linear_from_factorized(module, packed=True)
                    
                    # Navigate to parent and replace
                    parts = name.split(".")
                    parent = self.model
                    for part in parts[:-1]:
                        if part.isdigit():
                            parent = parent[int(part)]
                        else:
                            parent = getattr(parent, part)
                    
                    setattr(parent, parts[-1], optimized)
                except Exception as e:
                    logger.warning(f"Could not optimize layer {name}: {e}")
        
        logger.info("Optimization complete")
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.95,
        num_return_sequences: int = 1,
        **kwargs
    ) -> list:
        """Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            num_return_sequences: Number of sequences to generate
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated texts
        """
        if not self._loaded:
            self.load()
        
        self.model.eval()
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode
        texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return texts
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics.
        
        Returns:
            Dictionary with memory metrics
        """
        total_params = 0
        binary_params = 0
        float_params = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, (FactorizedLinear, OptimizedFactorizedLinear)):
                binary_params += module.d_out * module.rank + module.d_in * module.rank
                float_params += module.d_out + module.d_in  # scales
            elif isinstance(module, nn.Linear):
                total_params += module.weight.numel()
                if module.bias is not None:
                    float_params += module.bias.numel()
        
        original_bits = total_params * 16  # FP16
        quantized_bits = binary_params * 1 + float_params * 32
        
        return {
            "original_mb": original_bits / 8 / 1024 / 1024,
            "quantized_mb": quantized_bits / 8 / 1024 / 1024,
            "compression_ratio": original_bits / quantized_bits if quantized_bits > 0 else 0,
            "effective_bits": quantized_bits / total_params if total_params > 0 else 0,
            "total_layers": sum(1 for _ in self.model.named_modules()),
        }
    
    @torch.no_grad()
    def benchmark(self, prompt: str = "Hello, world!", num_runs: int = 10) -> Dict[str, float]:
        """Benchmark inference speed.
        
        Args:
            prompt: Test prompt
            num_runs: Number of benchmark runs
            
        Returns:
            Dictionary with benchmark results
        """
        if not self._loaded:
            self.load()
        
        self.model.eval()
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=50,
        ).to(self.device)
        
        # Warmup
        for _ in range(3):
            _ = self.model.generate(
                **inputs,
                max_length=50,
                do_sample=False,
                num_return_sequences=1,
            )
        
        # Benchmark
        import time
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        start_time = time.time()
        for _ in range(num_runs):
            _ = self.model.generate(
                **inputs,
                max_length=50,
                do_sample=False,
                num_return_sequences=1,
            )
            if self.device == "cuda":
                torch.cuda.synchronize()
        
        total_time = time.time() - start_time
        avg_time = total_time / num_runs
        
        # Measure memory
        if self.device == "cuda":
            peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            peak_memory_mb = 0
        
        return {
            "avg_time_sec": avg_time,
            "tokens_per_sec": 50 / avg_time,
            "peak_memory_mb": peak_memory_mb,
            "num_runs": num_runs,
        }


def create_inference_engine(
    model_path: str,
    device: Optional[str] = None,
) -> NanoQuantInferenceEngine:
    """Create inference engine with automatic device detection.
    
    Args:
        model_path: Path to quantized model
        device: Optional device override
        
    Returns:
        Configured NanoQuantInferenceEngine
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return NanoQuantInferenceEngine(
        model_path=model_path,
        device=device,
        use_optimized_kernels=True,
        pack_weights=True,
    )
