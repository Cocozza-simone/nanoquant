"""
Evaluation module for NANOQUANT.

Implements perplexity and zero-shot accuracy evaluation on standard benchmarks.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from .device_utils import get_optimal_device

logger = logging.getLogger(__name__)


def evaluate_perplexity(
    model: nn.Module,
    tokenizer,
    dataset_name: str = "wikitext",
    config_name: str = "wikitext-2-raw-v1",
    split: str = "test",
    batch_size: int = 1,
    max_length: int = 2048,
    stride: int = 512,
    device: str = "auto",
    max_samples: Optional[int] = None,
) -> float:
    """Evaluate perplexity on a language modeling dataset.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        dataset_name: Dataset name
        config_name: Dataset config
        split: Dataset split
        batch_size: Evaluation batch size
        max_length: Maximum sequence length
        stride: Stride for sliding window
        device: Device
        max_samples: Maximum number of samples to evaluate
        
    Returns:
        Perplexity score
    """
    logger.info(f"Evaluating perplexity on {dataset_name}/{config_name} ({split})")
    
    model.eval()
    
    # Load dataset
    try:
        dataset = load_dataset(
            dataset_name,
            config_name,
            split=split,
            trust_remote_code=True,
        )
    except Exception as e:
        logger.warning(f"Failed to load {config_name}, trying default: {e}")
        dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
    
    # Prepare texts
    texts = []
    for i, example in enumerate(dataset):
        if max_samples and i >= max_samples:
            break
        text = example.get("text", "")
        if text.strip():
            texts.append(text)
    
    logger.info(f"Evaluating on {len(texts)} texts")
    
    # Calculate perplexity using sliding window
    total_nll = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i, text in enumerate(texts):
            if i % 100 == 0:
                logger.info(f"  Processing text {i}/{len(texts)}")
            
            # Tokenize
            encodings = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length * 4,  # Allow longer texts
            )
            
            input_ids = encodings["input_ids"].to(device)
            
            # Sliding window
            seq_len = input_ids.size(1)
            
            if seq_len <= max_length:
                # Single forward pass
                outputs = model(input_ids, labels=input_ids)
                nll = outputs.loss.item() * seq_len
                total_nll += nll
                total_tokens += seq_len
            else:
                # Sliding window
                num_windows = (seq_len - max_length + stride - 1) // stride + 1
                
                for j in range(num_windows):
                    start_idx = j * stride
                    end_idx = min(start_idx + max_length, seq_len)
                    
                    window_ids = input_ids[:, start_idx:end_idx]
                    
                    # Only compute loss on the last stride tokens (except first window)
                    labels = window_ids.clone()
                    if j > 0:
                        labels[:, :-stride] = -100  # Ignore previous tokens
                    
                    outputs = model(window_ids, labels=labels)
                    
                    # Count actual tokens
                    valid_tokens = (labels != -100).sum().item() - 1  # Exclude first token
                    if valid_tokens > 0:
                        total_nll += outputs.loss.item() * valid_tokens
                        total_tokens += valid_tokens
    
    # Calculate perplexity
    avg_nll = total_nll / max(total_tokens, 1)
    perplexity = np.exp(avg_nll)
    
    logger.info(f"Perplexity: {perplexity:.2f}")
    
    return float(perplexity)


def evaluate_zero_shot(
    model: nn.Module,
    tokenizer,
    task_name: str,
    num_fewshot: int = 0,
    batch_size: int = 1,
    device: str = "auto",
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate zero-shot accuracy on commonsense reasoning tasks.
    
    Currently supports: winogrande, hellaswag, boolq, arc_easy, arc_challenge, piqa
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        task_name: Task name
        num_fewshot: Number of few-shot examples
        batch_size: Batch size
        device: Device
        max_samples: Max samples to evaluate
        
    Returns:
        Dictionary of metrics
    """
    logger.info(f"Evaluating zero-shot on {task_name}")
    
    model.eval()
    
    # Map task names to dataset configs
    task_map = {
        "winogrande": ("winogrande", "winogrande_xl"),
        "hellaswag": ("hellaswag", None),
        "boolq": ("boolq", None),
        "arc_easy": ("ai2_arc", "ARC-Easy"),
        "arc_challenge": ("ai2_arc", "ARC-Challenge"),
        "piqa": ("piqa", None),
    }
    
    if task_name not in task_map:
        logger.warning(f"Unknown task: {task_name}")
        return {"accuracy": 0.0}
    
    dataset_name, config_name = task_map[task_name]
    
    try:
        if config_name:
            dataset = load_dataset(dataset_name, config_name, split="validation", trust_remote_code=True)
        else:
            dataset = load_dataset(dataset_name, split="validation", trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return {"accuracy": 0.0}
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, example in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            
            if i % 100 == 0:
                logger.info(f"  Processing {i}/{min(len(dataset), max_samples or len(dataset))}")
            
            result = _evaluate_example(model, tokenizer, example, task_name, device)
            
            if result is not None:
                correct += result
                total += 1
    
    accuracy = correct / max(total, 1)
    logger.info(f"{task_name} accuracy: {accuracy:.4f} ({correct}/{total})")
    
    return {"accuracy": accuracy}


def _evaluate_example(
    model: nn.Module,
    tokenizer,
    example: dict,
    task_name: str,
    device: str,
) -> Optional[int]:
    """Evaluate a single example.
    
    Args:
        model: Model
        tokenizer: Tokenizer
        example: Dataset example
        task_name: Task name
        device: Device
        
    Returns:
        1 if correct, 0 if incorrect, None if skipped
    """
    try:
        if task_name == "boolq":
            return _eval_boolq(model, tokenizer, example, device)
        elif task_name == "piqa":
            return _eval_piqa(model, tokenizer, example, device)
        elif task_name in ["arc_easy", "arc_challenge"]:
            return _eval_arc(model, tokenizer, example, device)
        elif task_name == "hellaswag":
            return _eval_hellaswag(model, tokenizer, example, device)
        elif task_name == "winogrande":
            return _eval_winogrande(model, tokenizer, example, device)
        else:
            return None
    except Exception as e:
        logger.debug(f"Error evaluating example: {e}")
        return None


def _eval_boolq(model, tokenizer, example, device):
    """Evaluate BoolQ example."""
    passage = example.get("passage", "")
    question = example.get("question", "")
    label = example.get("answer", 0)
    
    prompt = f"Passage: {passage}\nQuestion: {question}?\nAnswer:"
    
    choices = ["No", "Yes"]
    scores = []
    
    for choice in choices:
        full_prompt = prompt + " " + choice
        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            scores.append(-outputs.loss.item())
    
    pred = 1 if scores[1] > scores[0] else 0
    return 1 if pred == label else 0


def _eval_piqa(model, tokenizer, example, device):
    """Evaluate PIQA example."""
    goal = example.get("goal", "")
    sol1 = example.get("sol1", "")
    sol2 = example.get("sol2", "")
    label = example.get("label", 0)
    
    scores = []
    for sol in [sol1, sol2]:
        prompt = f"Question: {goal}\nAnswer: {sol}"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            scores.append(-outputs.loss.item())
    
    pred = 0 if scores[0] > scores[1] else 1
    return 1 if pred == label else 0


def _eval_arc(model, tokenizer, example, device):
    """Evaluate ARC example."""
    question = example.get("question", "")
    choices = example.get("choices", {})
    
    if isinstance(choices, dict):
        labels = choices.get("label", [])
        texts = choices.get("text", [])
    else:
        return None
    
    answer_key = example.get("answerKey", "")
    
    if not labels or not texts:
        return None
    
    scores = []
    for text in texts:
        prompt = f"Question: {question}\nAnswer: {text}"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            scores.append(-outputs.loss.item())
    
    pred_idx = int(np.argmax(scores))
    
    # Map answer key to index
    if answer_key in labels:
        true_idx = labels.index(answer_key)
    else:
        try:
            true_idx = int(answer_key) - 1
        except:
            return None
    
    return 1 if pred_idx == true_idx else 0


def _eval_hellaswag(model, tokenizer, example, device):
    """Evaluate HellaSwag example."""
    ctx = example.get("ctx", "")
    endings = example.get("endings", [])
    label = example.get("label", 0)
    
    if not endings:
        return None
    
    scores = []
    for ending in endings:
        prompt = ctx + " " + ending
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            scores.append(-outputs.loss.item())
    
    pred = int(np.argmax(scores))
    return 1 if pred == label else 0


def _eval_winogrande(model, tokenizer, example, device):
    """Evaluate WinoGrande example."""
    sentence = example.get("sentence", "")
    option1 = example.get("option1", "")
    option2 = example.get("option2", "")
    answer = example.get("answer", "")
    
    # Replace _ with options
    scores = []
    for option in [option1, option2]:
        prompt = sentence.replace("_", option)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            scores.append(-outputs.loss.item())
    
    pred = 1 if scores[1] > scores[0] else 0
    
    try:
        label = int(answer) - 1
    except:
        return None
    
    return 1 if pred == label else 0
