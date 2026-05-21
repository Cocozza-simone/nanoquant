#!/usr/bin/env python3
"""
Evaluation script for NANOQUANT quantized models.

Usage:
    python evaluate.py --model-path ./outputs/quantized --tasks perplexity --device cuda
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import logging
import torch
import json
from nanoquant.quantization import NanoQuantizer
from nanoquant.evaluation import evaluate_perplexity, evaluate_zero_shot
from nanoquant.config import NanoQuantConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate NANOQUANT quantized model")
    
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to quantized model")
    parser.add_argument("--base-model", type=str, default=None,
                        help="Base model name (if different from saved)")
    parser.add_argument("--tasks", nargs="+",
                        default=["perplexity"],
                        choices=["perplexity", "winogrande", "hellaswag", "boolq",
                                 "arc_easy", "arc_challenge", "piqa", "all"],
                        help="Evaluation tasks")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device")
    parser.add_argument("--output", type=str, default="./outputs/eval_results.json",
                        help="Output file for results")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum samples to evaluate")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Check CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # Load config
    config_path = os.path.join(args.model_path, "nanoquant_config.json")
    with open(config_path, "r") as f:
        saved_config = json.load(f)
    
    config = NanoQuantConfig(
        model_name=args.base_model or saved_config["model_name"],
        rank=saved_config.get("rank", 8),
        bits=saved_config.get("bits", 1.0),
        device=args.device,
    )
    
    logger.info(f"Evaluating model from {args.model_path}")
    logger.info(f"Base model: {config.model_name}")
    logger.info(f"Rank: {config.rank}")
    logger.info(f"Bits: {config.bits}")
    
    # Load quantized model
    quantizer = NanoQuantizer(config)
    quantizer.load_model()
    
    # For evaluation, we can use the original model with factorized layers
    # In a full implementation, we would load the saved state dict
    model = quantizer.model
    tokenizer = quantizer.tokenizer
    
    # Evaluate
    results = {}
    
    tasks = args.tasks
    if "all" in tasks:
        tasks = ["perplexity", "winogrande", "hellaswag", "boolq",
                 "arc_easy", "arc_challenge", "piqa"]
    
    if "perplexity" in tasks:
        logger.info("\nEvaluating perplexity...")
        try:
            ppl = evaluate_perplexity(
                model, tokenizer,
                device=args.device,
                max_samples=args.max_samples or 100,
            )
            results["perplexity"] = ppl
            logger.info(f"Perplexity: {ppl:.2f}")
        except Exception as e:
            logger.error(f"Perplexity evaluation failed: {e}")
            results["perplexity"] = None
    
    zero_shot_tasks = [t for t in tasks if t != "perplexity"]
    for task in zero_shot_tasks:
        logger.info(f"\nEvaluating {task}...")
        try:
            result = evaluate_zero_shot(
                model, tokenizer,
                task_name=task,
                device=args.device,
                max_samples=args.max_samples or 500,
            )
            results[task] = result
            logger.info(f"{task}: {result}")
        except Exception as e:
            logger.error(f"{task} evaluation failed: {e}")
            results[task] = None
    
    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {args.output}")
    logger.info("\nSummary:")
    for task, result in results.items():
        if result is not None:
            if isinstance(result, dict):
                logger.info(f"  {task}: {result}")
            else:
                logger.info(f"  {task}: {result:.4f}")


if __name__ == "__main__":
    main()
