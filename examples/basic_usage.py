#!/usr/bin/env python3
"""
Basic usage example of NANOQUANT.

This example demonstrates quantizing a small GPT-2 model
and evaluating the results.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import logging
from nanoquant.config import NanoQuantConfig
from nanoquant.quantization import NanoQuantizer
from nanoquant.evaluation import evaluate_perplexity

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Run basic NANOQUANT example."""
    
    print("=" * 60)
    print("NANOQUANT Basic Usage Example")
    print("=" * 60)
    
    # Configuration for a small model
    config = NanoQuantConfig(
        model_name="gpt2",  # Small model for demonstration
        rank=4,             # Low-rank dimension
        bits=1.0,           # 1-bit quantization
        calib_samples=8,    # Small for demo (use 128 for real)
        calib_seq_len=128,  # Shorter for demo (use 2048 for real)
        admm_iterations=10, # Fewer for demo (use 50 for real)
        pre_tune_steps=5,   # Demo (use 20 for real)
        post_tune_steps=10, # Demo (use 50 for real)
        glob_tune_steps=5,  # Demo (use 30 for real)
        device="cpu",       # CPU for demo
        output_dir="./outputs/demo",
    )
    
    logger.info(f"Configuration:")
    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  Rank: {config.rank}")
    logger.info(f"  Bits: {config.bits}")
    logger.info(f"  Device: {config.device}")
    
    # Initialize and run quantization
    logger.info("\nInitializing NANOQUANT...")
    quantizer = NanoQuantizer(config)
    
    logger.info("\nLoading model...")
    quantizer.load_model()
    
    logger.info("\nRunning quantization pipeline...")
    logger.info("  Phase 1: Global Calibration")
    logger.info("  Phase 2: Block Reconstruction")
    logger.info("  Phase 3: Model Reconstruction")
    
    try:
        quantized_model = quantizer.quantize()
        logger.info("\nQuantization complete!")
        
        # Save
        quantizer.save_quantized_model(config.output_dir)
        
        # Evaluate
        logger.info("\nEvaluating perplexity...")
        ppl = evaluate_perplexity(
            quantized_model,
            quantizer.tokenizer,
            max_samples=10,
            max_length=128,
            device=config.device,
        )
        logger.info(f"Perplexity: {ppl:.2f}")
        
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        logger.info("This is expected for demo with limited resources.")
        logger.info("For production use, use a GPU and increase calibration samples.")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
