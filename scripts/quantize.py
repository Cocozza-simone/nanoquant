#!/usr/bin/env python3
"""
Main script to run NANOQUANT quantization on a language model.

Usage:
    python quantize.py --model meta-llama/Llama-2-7b-hf --rank 8 --output ./outputs/quantized
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


def parse_args():
    parser = argparse.ArgumentParser(description="NANOQUANT: Sub-1-Bit Quantization")
    
    # Model
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name or local path")
    parser.add_argument("--rank", type=int, default=8,
                        help="Low-rank dimension for binary factorization")
    parser.add_argument("--bits", type=float, default=1.0,
                        help="Target bit-width (1.0 for 1-bit, <1 for sub-1-bit)")
    
    # Calibration
    parser.add_argument("--calib-dataset", type=str, default="wikitext",
                        help="Calibration dataset")
    parser.add_argument("--calib-config", type=str, default="wikitext-2-raw-v1",
                        help="Calibration dataset config")
    parser.add_argument("--calib-samples", type=int, default=128,
                        help="Number of calibration samples (auto-reduced on MPS/CPU)")
    parser.add_argument("--calib-seq-len", type=int, default=2048,
                        help="Sequence length for calibration (auto-reduced on MPS/CPU)")
    
    # ADMM
    parser.add_argument("--admm-iters", type=int, default=50,
                        help="ADMM iterations")
    parser.add_argument("--admm-rho", type=float, default=1.0,
                        help="ADMM penalty parameter")
    parser.add_argument("--admm-lambda", type=float, default=0.01,
                        help="ADMM regularization")
    
    # Refinement
    parser.add_argument("--pre-tune-steps", type=int, default=20,
                        help="Error propagation mitigation steps")
    parser.add_argument("--post-tune-steps", type=int, default=50,
                        help="Factorized component refinement steps")
    parser.add_argument("--glob-tune-steps", type=int, default=30,
                        help="Global scale tuning steps")
    
    # Evaluation
    parser.add_argument("--eval-perplexity", action="store_true",
                        help="Evaluate perplexity after quantization")
    parser.add_argument("--eval-datasets", nargs="+",
                        default=["wikitext"],
                        help="Evaluation datasets")
    
    # System
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cuda/mps/cpu/auto) - auto-detects optimal device")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output", type=str, default="./outputs/quantized",
                        help="Output directory")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Warn about gated models
    if "llama" in args.model.lower() and "7b" in args.model.lower():
        logger.warning(
            f"\n{'='*70}\n"
            f"WARNING: Meta Llama-2 models are GATED (require HuggingFace authentication)\n"
            f"{'='*70}\n\n"
            f"If you get 401 Unauthorized error, try:\n"
            f"  1. Run:  huggingface-cli login\n"
            f"  2. Paste your HF token from https://huggingface.co/settings/tokens\n"
            f"  3. Accept terms at: https://huggingface.co/meta-llama/Llama-2-7b-hf\n\n"
            f"Alternative open models (no auth needed):\n"
            f"  - python scripts/quantize.py --model TinyLlama/TinyLlama-1.1B --rank 4\n"
            f"  - python scripts/quantize.py --model gpt2 --rank 4\n"
            f"  - python scripts/quantize.py --model facebook/opt-125m --rank 4\n"
            f"{'='*70}\n"
        )
    
    # Create config
    config = NanoQuantConfig(
        model_name=args.model,
        rank=args.rank,
        bits=args.bits,
        calib_dataset=args.calib_dataset,
        calib_config=args.calib_config,
        calib_samples=args.calib_samples,
        calib_seq_len=args.calib_seq_len,
        admm_iterations=args.admm_iters,
        admm_rho=args.admm_rho,
        admm_lambda=args.admm_lambda,
        pre_tune_steps=args.pre_tune_steps,
        post_tune_steps=args.post_tune_steps,
        glob_tune_steps=args.glob_tune_steps,
        device=args.device,
        seed=args.seed,
        output_dir=args.output,
    )
    
    logger.info("=" * 60)
    logger.info("NANOQUANT: Efficient Sub-1-Bit Quantization")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Rank: {args.rank}")
    logger.info(f"Target bits: {args.bits}")
    logger.info(f"Device (requested): {args.device}")
    logger.info("=" * 60)
    
    # Auto-detect optimal device
    optimal_device = get_optimal_device(args.device)
    if optimal_device != args.device and args.device != "auto":
        logger.info(f"Device (resolved to available): {optimal_device}")
    elif args.device == "auto":
        logger.info(f"Device (auto-detected): {optimal_device}")
    config.device = optimal_device
    
    # Warn about memory constraints
    if optimal_device == "mps":
        logger.warning(
            f"\n{'='*70}\n"
            f"⚠️  macOS Metal Performance Shaders (MPS) Memory Constraints\n"
            f"{'='*70}\n"
            f"Auto-adjusted parameters for MPS device:\n"
            f"  - Calibration samples: {config.calib_samples} (reduced from 128)\n"
            f"  - Sequence length: {config.calib_seq_len} (reduced from 2048)\n\n"
            f"If you still see memory errors:\n"
            f"  1. Use smaller model: --model gpt2 or --model TinyLlama/TinyLlama-1.1B\n"
            f"  2. Reduce samples: --calib-samples 32\n"
            f"  3. Reduce sequence: --calib-seq-len 512\n"
            f"  4. Use CPU: --device cpu (slower but more memory)\n"
            f"  5. Close other applications to free up memory\n"
            f"{'='*70}\n"
        )
    elif optimal_device == "cpu":
        logger.warning(
            f"\n{'='*70}\n"
            f"ℹ️  Using CPU - This will be significantly slower\n"
            f"{'='*70}\n"
            f"Adjusted parameters for CPU:\n"
            f"  - Calibration samples: {config.calib_samples}\n"
            f"  - Sequence length: {config.calib_seq_len}\n"
            f"Expected duration: 30-60 minutes for {args.model}\n"
            f"{'='*70}\n"
        )
    
    # Initialize quantizer
    quantizer = NanoQuantizer(config)
    
    # Run quantization
    quantized_model = quantizer.quantize()
    
    # Save
    quantizer.save_quantized_model(args.output)
    
    # Evaluate
    if args.eval_perplexity:
        logger.info("\nEvaluating perplexity...")
        ppl = evaluate_perplexity(
            quantized_model,
            quantizer.tokenizer,
            dataset_name="wikitext",
            config_name="wikitext-2-raw-v1",
            split="test",
            device=config.device,
            max_samples=100,
        )
        logger.info(f"Final perplexity: {ppl:.2f}")
    
    logger.info("\nDone!")


if __name__ == "__main__":
    main()
