#!/usr/bin/env python3
"""
NANOQUANT Quantization Script - MPS/macOS Optimized

This script provides memory-optimized quantization for Apple Silicon Macs.
Automatically adjusts parameters based on available memory.

Usage (recommended for M4 Pro):
    python scripts/quantize_mps_optimized.py --model gpt2 --rank 4
    python scripts/quantize_mps_optimized.py --model TinyLlama/TinyLlama-1.1B --rank 4
    python scripts/quantize_mps_optimized.py --model facebook/opt-125m --rank 8
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import logging
import torch
from nanoquant.config import NanoQuantConfig
from nanoquant.quantization import NanoQuantizer
from nanoquant.evaluation import evaluate_perplexity
from nanoquant.device_utils import get_optimal_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# MPS-optimized model configurations (tested on M4 Pro/Max)
MPS_MODELS = {
    "gpt2": {
        "name": "gpt2",
        "rank": 4,
        "bits": 1.0,
        "calib_samples": 32,
        "calib_seq_len": 512,
        "memory_gb": 8,
    },
    "tinyllama": {
        "name": "TinyLlama/TinyLlama-1.1B",
        "rank": 4,
        "bits": 1.0,
        "calib_samples": 32,
        "calib_seq_len": 512,
        "memory_gb": 10,
    },
    "opt-125m": {
        "name": "facebook/opt-125m",
        "rank": 4,
        "bits": 1.0,
        "calib_samples": 32,
        "calib_seq_len": 512,
        "memory_gb": 10,
    },
    "opt-350m": {
        "name": "facebook/opt-350m",
        "rank": 8,
        "bits": 1.0,
        "calib_samples": 16,
        "calib_seq_len": 256,
        "memory_gb": 14,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="NANOQUANT: MPS-Optimized Sub-1-Bit Quantization for macOS",
        epilog="""
Examples:
    python scripts/quantize_mps_optimized.py --model gpt2 --rank 4
    python scripts/quantize_mps_optimized.py --model tinyllama --rank 8
    python scripts/quantize_mps_optimized.py --model opt-125m --rank 4 --eval
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        choices=list(MPS_MODELS.keys()),
        help="Pre-optimized model for MPS (no auth needed)",
    )
    parser.add_argument(
        "--rank", type=int, default=None, help="Low-rank dimension (default: model-specific)"
    )
    parser.add_argument("--bits", type=float, default=None, help="Target bits (default: 1.0)")
    parser.add_argument(
        "--calib-samples",
        type=int,
        default=None,
        help="Calibration samples (default: model-specific)",
    )
    parser.add_argument(
        "--calib-seq-len",
        type=int,
        default=None,
        help="Calibration sequence length (default: model-specific)",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate perplexity after quantization (adds 5-10 min)",
    )
    parser.add_argument("--output", type=str, default="./outputs/quantized_mps",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def estimate_memory_usage(model_params_m: float, calib_samples: int, seq_len: int) -> float:
    """Estimate peak memory usage in GB."""
    # Model weights
    model_memory = (model_params_m * 4) / 1024  # 4 bytes per float32
    
    # Activation memory during calibration
    # Rough estimate: hidden_dim * seq_len * calib_samples * (data + hooks buffer)
    activation_memory = (model_params_m / 10) * seq_len * calib_samples * 1.5 / 1024
    
    total = model_memory + activation_memory
    return total


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Get model config
    if args.model not in MPS_MODELS:
        logger.error(f"Unknown model: {args.model}")
        logger.info(f"Available models: {list(MPS_MODELS.keys())}")
        sys.exit(1)

    model_config = MPS_MODELS[args.model].copy()
    model_name = model_config.pop("name")
    required_memory_gb = model_config.pop("memory_gb")

    # Override defaults with CLI args
    if args.rank is not None:
        model_config["rank"] = args.rank
    if args.bits is not None:
        model_config["bits"] = args.bits
    if args.calib_samples is not None:
        model_config["calib_samples"] = args.calib_samples
    if args.calib_seq_len is not None:
        model_config["calib_seq_len"] = args.calib_seq_len

    # Show header
    logger.info("=" * 80)
    logger.info("🍎 NANOQUANT - MPS Optimized Quantization for macOS")
    logger.info("=" * 80)

    # Detect device
    device = get_optimal_device("auto")
    logger.info(f"Device: {device.upper()}")

    if device != "mps":
        logger.warning(f"⚠️  MPS not available. Using {device.upper()} instead.")
        logger.warning("Performance may be significantly slower.")

    logger.info(f"Model: {model_name}")
    logger.info(f"Rank: {model_config['rank']}")
    logger.info(f"Bits: {model_config['bits']}")
    logger.info(f"Calibration samples: {model_config['calib_samples']}")
    logger.info(f"Sequence length: {model_config['calib_seq_len']}")
    logger.info(f"Estimated peak memory: ~{required_memory_gb}GB")
    logger.info("=" * 80)

    try:
        # Create config
        config = NanoQuantConfig(
            model_name=model_name,
            rank=model_config["rank"],
            bits=model_config["bits"],
            calib_samples=model_config["calib_samples"],
            calib_seq_len=model_config["calib_seq_len"],
            device=device,
            output_dir=args.output,
            seed=args.seed,
        )

        logger.info(f"\n✅ Config created with MPS optimizations")
        logger.info(f"   Adjusted calib_samples: {config.calib_samples}")
        logger.info(f"   Adjusted calib_seq_len: {config.calib_seq_len}")
        logger.info(f"   Adjusted tune_fp_batch_size: {config.tune_fp_batch_size}")

        logger.info(f"\n📦 Loading model: {model_name}")
        logger.info("   (First run will download model ~250MB-1GB)\n")

        # Initialize quantizer
        quantizer = NanoQuantizer(config)

        # Run quantization
        logger.info("🔄 Starting quantization pipeline...")
        quantized_model = quantizer.quantize()

        # Save
        logger.info(f"\n💾 Saving quantized model to {args.output}")
        quantizer.save_quantized_model(args.output)

        # Evaluate
        if args.eval:
            logger.info("\n📊 Evaluating perplexity...")
            ppl = evaluate_perplexity(
                quantized_model,
                quantizer.tokenizer,
                dataset_name="wikitext",
                config_name="wikitext-2-raw-v1",
                split="test",
                device=config.device,
                max_samples=100,
            )
            logger.info(f"✅ Final perplexity: {ppl:.2f}")

        logger.info("\n" + "=" * 80)
        logger.info("✅ Quantization completed successfully!")
        logger.info(f"📁 Output: {args.output}")
        logger.info("=" * 80)

    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error(f"\n❌ Out of Memory Error!")
            logger.error("\nTroubleshooting steps:")
            logger.error("  1. Close other applications to free up memory")
            logger.error("  2. Use a smaller model (--model gpt2)")
            logger.error("  3. Reduce calibration samples (--calib-samples 16)")
            logger.error("  4. Reduce sequence length (--calib-seq-len 256)")
            logger.error("  5. Switch to CPU (will be slower, but more memory)")
            sys.exit(1)
        else:
            raise

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
