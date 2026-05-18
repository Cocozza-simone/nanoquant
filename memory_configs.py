#!/usr/bin/env python3
"""
Pre-configured NANOQUANT settings optimized for different GPU/CPU setups.
Use these as templates for your hardware.

Example usage:
    from memory_configs import get_optimal_config
    config = get_optimal_config("Llama-2-7b", gpu_type="rtx_4090")
"""

import torch
from dataclasses import dataclass, asdict
from nanoquant.config import NanoQuantConfig


@dataclass
class HardwareProfile:
    """Hardware constraints profile."""
    gpu_mem_gb: int
    cpu_mem_gb: int
    gpu_type: str
    
    def __str__(self):
        return f"{self.gpu_type} ({self.gpu_mem_gb}GB GPU, {self.cpu_mem_gb}GB CPU)"


# Hardware profiles
PROFILES = {
    "laptop": HardwareProfile(gpu_mem_gb=8, cpu_mem_gb=16, gpu_type="RTX 4050"),
    "gaming": HardwareProfile(gpu_mem_gb=12, cpu_mem_gb=32, gpu_type="RTX 4070"),
    "workstation": HardwareProfile(gpu_mem_gb=24, cpu_mem_gb=64, gpu_type="RTX 4090"),
    "data_center": HardwareProfile(gpu_mem_gb=80, cpu_mem_gb=256, gpu_type="A100"),
    "multi_gpu_a100": HardwareProfile(gpu_mem_gb=160, cpu_mem_gb=512, gpu_type="2x A100"),
}


def get_hardware_profile() -> HardwareProfile:
    """Auto-detect hardware profile."""
    
    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        if gpu_mem_gb >= 160:
            return PROFILES["multi_gpu_a100"]
        elif gpu_mem_gb >= 70:
            return PROFILES["data_center"]
        elif gpu_mem_gb >= 22:
            return PROFILES["workstation"]
        elif gpu_mem_gb >= 11:
            return PROFILES["gaming"]
        else:
            return PROFILES["laptop"]
    else:
        return PROFILES["laptop"]


# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

def gpt2_minimal() -> NanoQuantConfig:
    """GPT-2 (124M) - Minimal footprint for testing."""
    return NanoQuantConfig(
        model_name="gpt2",
        rank=2,
        bits=1.0,
        calib_samples=8,
        calib_seq_len=128,
        calib_batch_size=1,
        tune_fp_batch_size=1,
        tune_fp_epochs=2,
        tune_latent_epochs=2,
        tune_scales_epochs=2,
        device="auto",
    )


def gpt2_optimized() -> NanoQuantConfig:
    """GPT-2 (124M) - Balanced quality/memory."""
    return NanoQuantConfig(
        model_name="gpt2",
        rank=4,
        bits=1.0,
        calib_samples=32,
        calib_seq_len=256,
        calib_batch_size=1,
        tune_fp_batch_size=1,
        tune_fp_epochs=4,
        tune_latent_epochs=4,
        tune_scales_epochs=4,
        device="auto",
    )


def opt_125m_rtx4090() -> NanoQuantConfig:
    """OPT-125M - RTX 4090 optimized."""
    return NanoQuantConfig(
        model_name="facebook/opt-125m",
        rank=4,
        bits=1.0,
        calib_samples=64,
        calib_seq_len=512,
        calib_batch_size=1,
        tune_fp_batch_size=2,
        tune_latent_batch_size=1,
        tune_scales_batch_size=1,
        device="cuda",
    )


def opt_6_7b_laptop() -> NanoQuantConfig:
    """OPT-6.7B - Laptop with GPU (12GB)."""
    return NanoQuantConfig(
        model_name="facebook/opt-6.7b",
        rank=4,
        bits=1.0,
        calib_samples=16,
        calib_seq_len=256,
        calib_batch_size=1,
        tune_fp_batch_size=1,
        tune_latent_batch_size=1,
        tune_scales_batch_size=1,
        device="cuda",
    )


def tinyllama_workstation() -> NanoQuantConfig:
    """TinyLlama-1.1B - Workstation (24GB GPU)."""
    return NanoQuantConfig(
        model_name="TinyLlama/TinyLlama-1.1B",
        rank=4,
        bits=1.0,
        calib_samples=32,
        calib_seq_len=512,
        calib_batch_size=1,
        tune_fp_batch_size=2,
        tune_latent_batch_size=1,
        tune_scales_batch_size=1,
        device="cuda",
    )


def llama2_7b_gaming() -> NanoQuantConfig:
    """Llama-2-7B - Gaming GPU (12GB) - Conservative."""
    return NanoQuantConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        rank=4,  # ← Basso rango per memoria
        bits=1.0,
        calib_samples=16,  # ← Molto ridotto
        calib_seq_len=256,
        calib_batch_size=1,
        tune_fp_batch_size=1,
        tune_latent_batch_size=1,
        tune_scales_batch_size=1,
        device="cuda",
    )


def llama2_7b_workstation() -> NanoQuantConfig:
    """Llama-2-7B - Workstation (24GB GPU) - Balanced."""
    return NanoQuantConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        rank=8,
        bits=1.0,
        calib_samples=32,  # ← Ridotto da 128
        calib_seq_len=512,  # ← Ridotto da 2048
        calib_batch_size=1,
        tune_fp_batch_size=1,
        tune_latent_batch_size=1,
        tune_scales_batch_size=1,
        device="cuda",
    )


def llama2_7b_server() -> NanoQuantConfig:
    """Llama-2-7B - A100 Server (80GB) - Full quality."""
    return NanoQuantConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        rank=8,
        bits=1.0,
        calib_samples=64,
        calib_seq_len=1024,
        calib_batch_size=2,
        tune_fp_batch_size=2,
        tune_latent_batch_size=1,
        tune_scales_batch_size=1,
        device="cuda",
    )


def llama2_13b_server() -> NanoQuantConfig:
    """Llama-2-13B - A100 Server (80GB) - Conservative."""
    return NanoQuantConfig(
        model_name="meta-llama/Llama-2-13b-hf",
        rank=8,
        bits=1.0,
        calib_samples=32,
        calib_seq_len=512,
        calib_batch_size=1,
        tune_fp_batch_size=1,
        tune_latent_batch_size=1,
        tune_scales_batch_size=1,
        device="cuda",
    )


def llama2_70b_multi_a100() -> NanoQuantConfig:
    """Llama-2-70B - Multi-GPU A100 (160GB total) - Full."""
    return NanoQuantConfig(
        model_name="meta-llama/Llama-2-70b-hf",
        rank=16,
        bits=1.0,
        calib_samples=128,  # ← Full
        calib_seq_len=2048,  # ← Full
        calib_batch_size=4,  # ← Aumentato
        tune_fp_batch_size=4,
        tune_latent_batch_size=2,
        tune_scales_batch_size=2,
        device="cuda",
    )


def gemma_7b_workstation() -> NanoQuantConfig:
    """Gemma-7B - Workstation (24GB GPU)."""
    return NanoQuantConfig(
        model_name="google/gemma-7b",
        rank=8,
        bits=1.0,
        calib_samples=32,
        calib_seq_len=512,
        calib_batch_size=1,
        tune_fp_batch_size=1,
        tune_latent_batch_size=1,
        tune_scales_batch_size=1,
        shrinkage_gamma=0.6,  # ← Gemma-specific
        device="cuda",
    )


# ============================================================================
# AUTO-SELECTION FUNCTIONS
# ============================================================================

MODEL_PRESETS = {
    # GPT-2
    "gpt2-minimal": gpt2_minimal,
    "gpt2": gpt2_optimized,
    
    # OPT
    "opt-125m": opt_125m_rtx4090,
    "opt-6.7b": opt_6_7b_laptop,
    
    # TinyLlama
    "tinyllama": tinyllama_workstation,
    
    # Llama-2
    "llama2-7b": llama2_7b_gaming,  # Default to conservative
    "llama2-7b-conservative": llama2_7b_gaming,
    "llama2-7b-balanced": llama2_7b_workstation,
    "llama2-7b-quality": llama2_7b_server,
    "llama2-13b": llama2_13b_server,
    "llama2-70b": llama2_70b_multi_a100,
    
    # Gemma
    "gemma-7b": gemma_7b_workstation,
}


def get_optimal_config(
    model: str,
    gpu_type: str = "auto",
    quality_preset: str = "balanced",
) -> NanoQuantConfig:
    """Get optimal config for model + hardware combination.
    
    Args:
        model: Model name (e.g., "gpt2", "llama2-7b", "opt-125m")
        gpu_type: GPU type ("auto", "laptop", "gaming", "workstation", "server")
        quality_preset: "conservative" | "balanced" | "quality"
    
    Returns:
        Optimized NanoQuantConfig
    """
    
    # Auto-detect GPU if needed
    if gpu_type == "auto":
        profile = get_hardware_profile()
        print(f"🔍 Detected hardware: {profile}")
        
        if profile.gpu_mem_gb >= 70:
            gpu_type = "server"
        elif profile.gpu_mem_gb >= 22:
            gpu_type = "workstation"
        elif profile.gpu_mem_gb >= 11:
            gpu_type = "gaming"
        else:
            gpu_type = "laptop"
    
    # Build preset key
    if quality_preset != "balanced":
        preset_key = f"{model}-{quality_preset}"
        if preset_key in MODEL_PRESETS:
            return MODEL_PRESETS[preset_key]()
    
    # Default to available preset
    if model in MODEL_PRESETS:
        return MODEL_PRESETS[model]()
    
    raise ValueError(f"Unknown model preset: {model}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_config_comparison():
    """Print comparison of different presets."""
    
    print("\n" + "="*80)
    print("NANOQUANT Memory-Optimized Configurations")
    print("="*80 + "\n")
    
    configs = [
        ("GPT-2 (Laptop)", gpt2_minimal()),
        ("Llama-2-7B (Gaming 12GB)", llama2_7b_gaming()),
        ("Llama-2-7B (Workstation 24GB)", llama2_7b_workstation()),
        ("Llama-2-7B (Server A100)", llama2_7b_server()),
        ("Llama-2-70B (Multi-GPU)", llama2_70b_multi_a100()),
    ]
    
    for name, config in configs:
        print(f"📊 {name}")
        print(f"   - Rank: {config.rank}")
        print(f"   - Calib samples: {config.calib_samples}")
        print(f"   - Calib seq len: {config.calib_seq_len}")
        print(f"   - Batch size: {config.calib_batch_size}")
        est_mem = estimate_memory_usage(config)
        print(f"   - Est. memory: ~{est_mem:.1f} GB")
        print()


def estimate_memory_usage(config: NanoQuantConfig) -> float:
    """Rough estimate of memory usage in GB."""
    
    # Model sizes (rough)
    model_sizes = {
        "gpt2": 0.5,
        "facebook/opt-125m": 0.5,
        "facebook/opt-6.7b": 13,
        "TinyLlama/TinyLlama-1.1B": 4,
        "meta-llama/Llama-2-7b-hf": 13,
        "meta-llama/Llama-2-13b-hf": 26,
        "meta-llama/Llama-2-70b-hf": 138,
        "google/gemma-7b": 13,
    }
    
    model_size = model_sizes.get(config.model_name, 10)
    
    # Memory formula:
    # Model (float16) + Activations (2.5x) + Calibration data + Gradients
    estimated = model_size + (model_size * 2.5) + (
        config.calib_samples * config.calib_seq_len * config.calib_batch_size 
        * 2 / 1e9
    )
    
    return estimated


def create_custom_config(
    model_name: str,
    rank: int,
    calib_samples: int,
    calib_seq_len: int,
    batch_size: int = 1,
    device: str = "cuda",
) -> NanoQuantConfig:
    """Create a custom config with full control."""
    
    return NanoQuantConfig(
        model_name=model_name,
        rank=rank,
        bits=1.0,
        calib_samples=calib_samples,
        calib_seq_len=calib_seq_len,
        calib_batch_size=batch_size,
        tune_fp_batch_size=max(1, batch_size // 2),
        tune_latent_batch_size=1,
        tune_scales_batch_size=1,
        device=device,
    )


# ============================================================================
# MAIN - Example usage
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NANOQUANT Memory-Optimized Configs")
    parser.add_argument("--list", action="store_true", help="List all presets")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--gpu", type=str, default="auto", help="GPU type")
    parser.add_argument("--quality", type=str, default="balanced", 
                       help="Quality preset (conservative|balanced|quality)")
    parser.add_argument("--info", type=str, help="Show info for preset")
    
    args = parser.parse_args()
    
    if args.list:
        print("\n📋 Available Model Presets:\n")
        for key in sorted(MODEL_PRESETS.keys()):
            print(f"  - {key}")
        print()
    
    elif args.info:
        config = get_optimal_config(args.info)
        print(f"\n📋 Configuration for '{args.info}':\n")
        for key, value in asdict(config).items():
            print(f"  {key}: {value}")
        print()
    
    else:
        print(f"\n🔍 Getting optimal config for '{args.model}' on {args.gpu} GPU...")
        config = get_optimal_config(args.model, args.gpu, args.quality)
        
        print(f"\n✅ Configuration:\n")
        for key, value in asdict(config).items():
            if value is not None:
                print(f"  {key}: {value}")
        
        print(f"\n📊 Estimated memory: {estimate_memory_usage(config):.1f} GB\n")
        
        print("💡 Usage:\n")
        print("  from memory_configs import get_optimal_config")
        print(f"  config = get_optimal_config('{args.model}')")
        print("  quantizer = NanoQuantizer(config)")
        print("  quantizer.quantize()")
        print()
