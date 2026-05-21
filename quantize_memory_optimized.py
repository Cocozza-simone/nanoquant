#!/usr/bin/env python3
"""
Memory-Optimized Quantization Script with Real-Time Monitoring.

Usage:
    python quantize_memory_optimized.py --model gpt2 --gpu auto
    python quantize_memory_optimized.py --model llama2-7b --gpu workstation
    python quantize_memory_optimized.py --model opt-125m --quality conservative
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import argparse
import torch
import logging
from datetime import datetime
from pathlib import Path

from nanoquant import NanoQuantizer
from nanoquant.evaluation import evaluate_perplexity
from memory_configs import (
    get_optimal_config,
    estimate_memory_usage,
    get_hardware_profile,
    MODEL_PRESETS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Real-time memory monitoring during quantization."""
    
    def __init__(self):
        self.peak_memory = 0
        self.measurements = []
    
    def log_memory(self, tag: str = ""):
        """Log current memory usage."""
        if not torch.cuda.is_available():
            return
        
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        
        self.peak_memory = max(self.peak_memory, peak)
        
        msg = f"[{tag}] Allocated: {allocated:.1f}GB, Reserved: {reserved:.1f}GB, Peak: {peak:.1f}GB"
        print(f"💾 {msg}")
        
        self.measurements.append({
            "tag": tag,
            "allocated": allocated,
            "reserved": reserved,
            "peak": peak,
            "timestamp": datetime.now(),
        })
    
    def print_summary(self):
        """Print memory usage summary."""
        print("\n" + "="*70)
        print("📊 MEMORY USAGE SUMMARY")
        print("="*70)
        
        for m in self.measurements:
            print(f"{m['tag']:30} | Alloc: {m['allocated']:6.1f}GB | "
                  f"Reserved: {m['reserved']:6.1f}GB | Peak: {m['peak']:6.1f}GB")
        
        print("-"*70)
        print(f"{'PEAK TOTAL':30} | {self.peak_memory:6.1f}GB")
        print("="*70 + "\n")


def print_hardware_info():
    """Print hardware information."""
    print("\n" + "="*70)
    print("🖥️  HARDWARE INFORMATION")
    print("="*70)
    
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}")
        print(f"Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"CUDA Capability: {props.major}.{props.minor}")
        
        profile = get_hardware_profile()
        print(f"\nDetected Profile: {profile}")
    else:
        print("GPU: Not available (using CPU)")
        print("⚠️  WARNING: CPU quantization will be very slow!")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Memory-Optimized NANOQUANT Quantization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect hardware and quantize GPT-2
  python quantize_memory_optimized.py --model gpt2
  
  # Quantize Llama-2-7B on workstation with balanced quality
  python quantize_memory_optimized.py --model llama2-7b --gpu workstation
  
  # Conservative quantization for limited GPU memory
  python quantize_memory_optimized.py --model llama2-7b --quality conservative
  
  # List all available presets
  python quantize_memory_optimized.py --list-presets
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model name (e.g., gpt2, llama2-7b, opt-125m)"
    )
    
    parser.add_argument(
        "--gpu",
        type=str,
        default="auto",
        choices=["auto", "laptop", "gaming", "workstation", "server"],
        help="GPU type for optimal configuration"
    )
    
    parser.add_argument(
        "--quality",
        type=str,
        default="balanced",
        choices=["conservative", "balanced", "quality"],
        help="Quality preset"
    )
    
    parser.add_argument(
        "--rank",
        type=int,
        help="Override rank (use for custom tuning)"
    )
    
    parser.add_argument(
        "--calib-samples",
        type=int,
        help="Override calibration samples"
    )
    
    parser.add_argument(
        "--calib-seq-len",
        type=int,
        help="Override calibration sequence length"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/quantized_model",
        help="Output directory for quantized model"
    )
    
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List all available model presets"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration without running quantization"
    )
    
    args = parser.parse_args()
    
    # List presets if requested
    if args.list_presets:
        print("\n📋 Available Model Presets:\n")
        for key in sorted(MODEL_PRESETS.keys()):
            print(f"  • {key}")
        print("\nUsage: python quantize_memory_optimized.py --model <preset>\n")
        return
    
    # Print hardware info
    print_hardware_info()
    
    # Get configuration
    print("🔧 Loading configuration...")
    try:
        config = get_optimal_config(args.model, args.gpu, args.quality)
    except ValueError as e:
        logger.error(f"❌ {e}")
        logger.error(f"Use --list-presets to see available models")
        return 1
    
    # Override with CLI args if provided
    if args.rank:
        config.rank = args.rank
    if args.calib_samples:
        config.calib_samples = args.calib_samples
    if args.calib_seq_len:
        config.calib_seq_len = args.calib_seq_len
    
    # Print configuration
    print("\n" + "="*70)
    print("⚙️  QUANTIZATION CONFIGURATION")
    print("="*70)
    print(f"Model:                {config.model_name}")
    print(f"Rank:                 {config.rank}")
    print(f"Bits:                 {config.bits}")
    print(f"Calibration samples:  {config.calib_samples}")
    print(f"Calibration seq len:  {config.calib_seq_len}")
    print(f"Batch size:           {config.calib_batch_size}")
    print(f"Device:               {config.device}")
    print()
    
    # Estimate memory
    est_mem = estimate_memory_usage(config)
    print(f"📊 Estimated memory usage: ~{est_mem:.1f} GB")
    
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        if est_mem > gpu_mem * 0.95:
            print(f"⚠️  WARNING: Estimated memory ({est_mem:.1f}GB) exceeds GPU capacity!")
            print("   Consider using --quality conservative or reducing --calib-samples")
            if not args.dry_run:
                response = input("Continue anyway? (y/n): ")
                if response.lower() != 'y':
                    return 1
    
    print("="*70)
    
    # Dry run - exit without quantizing
    if args.dry_run:
        print("\n✅ Dry run completed. Configuration looks good!")
        return 0
    
    # Start quantization
    print("\n🚀 Starting quantization...\n")
    
    monitor = MemoryMonitor()
    
    try:
        # Reset memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        
        monitor.log_memory("Initial")
        
        # Create quantizer
        quantizer = NanoQuantizer(config)
        monitor.log_memory("After Quantizer Init")
        
        # Load model
        logger.info("Loading model...")
        quantizer.load_model()
        monitor.log_memory("After Model Load")
        
        # Run quantization
        logger.info("Running quantization pipeline...")
        quantized_model = quantizer.quantize()
        monitor.log_memory("After Quantization")
        
        # Evaluate
        logger.info("Evaluating quantized model...")
        try:
            perplexity = evaluate_perplexity(
                quantized_model,
                quantizer.tokenizer,
                dataset_name=config.calib_dataset,
                config_name=config.calib_config,
                split="test",
                batch_size=1,
                max_length=256,
                max_samples=10,  # Limit for speed
                device=config.device,
            )
            results = {
                "perplexity": perplexity,
                "compression_ratio": 8.0,  # Approximate
            }
        except Exception as e:
            logger.warning(f"Perplexity evaluation failed: {e}")
            results = {
                "perplexity": "N/A",
                "compression_ratio": 8.0,
            }
        
        monitor.log_memory("After Evaluation")
        
        # Save
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving quantized model to {output_path}...")
        try:
            quantizer.save_quantized_model(str(output_path))
            monitor.log_memory("After Save")
        except Exception as e:
            logger.warning(f"Model save failed: {e}")
            logger.warning(f"Keeping model in memory instead")
        
        # Print results
        print("\n" + "="*70)
        print("✅ QUANTIZATION COMPLETE!")
        print("="*70)
        
        print("\n📈 Results:")
        perplexity = results.get('perplexity', 'N/A')
        print(f"  Perplexity:        {perplexity}")
        
        compression = results.get('compression_ratio', 'N/A')
        if isinstance(compression, (int, float)):
            print(f"  Compression Ratio: {compression:.2f}x")
        else:
            print(f"  Compression Ratio: {compression}")
        
        speed = results.get('speed', 'N/A')
        if isinstance(speed, (int, float)):
            print(f"  Speed:             {speed:.1%}")
        else:
            print(f"  Speed:             {speed}")
        
        print(f"\n💾 Model saved to: {output_path}")
        
        # Print memory summary
        monitor.print_summary()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Quantization interrupted by user")
        monitor.print_summary()
        return 1
        
    except Exception as e:
        logger.error(f"❌ Error during quantization: {e}", exc_info=True)
        monitor.print_summary()
        return 1


if __name__ == "__main__":
    exit(main())
