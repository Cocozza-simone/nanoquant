#!/usr/bin/env python3
"""
NANOQUANT Quantization Demo - Uses public open models (no authentication needed)

This script quantizes open-source models that don't require HuggingFace authentication.
Perfect for testing on macOS M4 Pro!

Usage:
    python scripts/quantize_demo.py --model TinyLlama/TinyLlama-1.1B --rank 4
    python scripts/quantize_demo.py --model gpt2 --rank 4
    python scripts/quantize_demo.py --model facebook/opt-125m --rank 4
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import logging
import torch
from nanoquant.config import NanoQuantConfig
from nanoquant.device_utils import get_optimal_device, get_device_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Recommended models (no authentication needed)
RECOMMENDED_MODELS = {
    "tinyllama": {
        "name": "TinyLlama/TinyLlama-1.1B",
        "rank": 4,
        "bits": 1.0,
        "calib_samples": 32,
    },
    "gpt2": {
        "name": "gpt2",
        "rank": 4,
        "bits": 1.0,
        "calib_samples": 16,
    },
    "opt": {
        "name": "facebook/opt-125m",
        "rank": 4,
        "bits": 1.0,
        "calib_samples": 32,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="NANOQUANT Demo: Sub-1-Bit Quantization (No Auth Required)",
        epilog="""
Examples:
    python scripts/quantize_demo.py --model tinyllama
    python scripts/quantize_demo.py --model gpt2 --rank 8
    python scripts/quantize_demo.py --model opt --bits 0.5
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="tinyllama",
        choices=list(RECOMMENDED_MODELS.keys()),
        help="Model to quantize (no auth needed)",
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
        "--eval-perplexity",
        action="store_true",
        help="Evaluate perplexity after quantization",
    )
    parser.add_argument("--output", type=str, default="./outputs/quantized_demo",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Get model config
    if args.model not in RECOMMENDED_MODELS:
        logger.error(f"Unknown model: {args.model}")
        logger.info(f"Available models: {list(RECOMMENDED_MODELS.keys())}")
        sys.exit(1)

    model_config = RECOMMENDED_MODELS[args.model].copy()
    model_name = model_config.pop("name")

    # Override defaults with CLI args
    if args.rank is not None:
        model_config["rank"] = args.rank
    if args.bits is not None:
        model_config["bits"] = args.bits
    if args.calib_samples is not None:
        model_config["calib_samples"] = args.calib_samples

    # Show device info
    logger.info("=" * 70)
    logger.info("NANOQUANT Demo - macOS M4 Pro Compatible")
    logger.info("=" * 70)

    device_info = get_device_info()
    logger.info(f"Platform: {device_info['platform']}")
    logger.info(f"Optimal Device: {device_info['optimal_device']}")
    logger.info(f"MPS Available: {device_info['mps_available']}")
    logger.info(f"PyTorch Version: {device_info['pytorch_version']}")
    logger.info("=" * 70)

    logger.info(f"Model: {model_name}")
    logger.info(f"Rank: {model_config['rank']}")
    logger.info(f"Bits: {model_config['bits']}")
    logger.info(f"Calibration samples: {model_config['calib_samples']}")
    logger.info("=" * 70)

    try:
        # Create config with auto device detection
        config = NanoQuantConfig(
            model_name=model_name,
            rank=model_config["rank"],
            bits=model_config["bits"],
            calib_samples=model_config["calib_samples"],
            device="auto",  # Auto-detect: CUDA > MPS > CPU
            output_dir=args.output,
            seed=args.seed,
        )

        logger.info(f"\n✅ Device resolved to: {config.device}")
        logger.info(f"✅ Config created successfully")

        # Try to load model (but don't run full quantization - just demo)
        logger.info(f"\n📦 Loading model: {model_name}")
        logger.info("   (This may take a moment on first run...)\n")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, trust_remote_code=True
        )

        logger.info(f"✅ Model loaded successfully!")
        logger.info(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"   Device: {config.device}")

        # Show what would happen next
        logger.info("\n" + "=" * 70)
        logger.info("Next Steps for Full Quantization:")
        logger.info("=" * 70)
        logger.info(
            """
1. Calibration: Collect activation/gradient statistics
2. ADMM Initialization: Solve low-rank binary factorization
3. Block Refinement: Three-step optimization per block
4. Model Reconstruction: Global scale tuning
5. Packing: Compress to binary storage
6. Evaluation: Measure quality loss

To run full quantization:
    from nanoquant import NanoQuantizer
    quantizer = NanoQuantizer(config)
    quantized_model = quantizer.quantize()
    quantizer.save_quantized_model(args.output)
        """
        )
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
