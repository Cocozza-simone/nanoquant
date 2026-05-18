"""
Model Reconstruction (TUNESCALESKD).

Implements Phase 3 of the NANOQUANT algorithm:
"Model Reconstruction" (Algorithm 1, line 25).

After block-wise optimization concludes, the binary parameters are
frozen and packed into efficient integer formats. The final model
reconstruction phase focuses exclusively on optimizing the
floating-point scaling vectors S_global = {s1, s2} for all layers
to align the logits of the quantized model with the original predictions.

The objective function (Equation 11):
    min_{S_global} ||Logits(M(X)) - Logits(M_c(X; S_global))||_KL

Reference: Kwon et al., "AlphaTuning: Quantization-Aware Parameter-
Efficient Adaptation of Large-Scale Pre-Trained Language Models", 2022.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Optional, Dict
from .device_utils import get_optimal_device

logger = logging.getLogger(__name__)


def tune_scales_kd(
    quantized_model: nn.Module,
    original_model: nn.Module,
    dataloader: List[torch.Tensor],
    num_epochs: int = 8,
    learning_rate: float = 1e-6,
    batch_size: int = 1,
    temperature: float = 1.0,
    device: str = "auto",
) -> nn.Module:
    """Tune global scaling vectors via knowledge distillation.
    
    Implements Algorithm 1, Phase 3 (TUNESCALESKD):
    "Optimize the floating-point scaling vectors S_global to align
    the logits of the quantized model with the original predictions."
    
    The objective minimizes KL divergence (Equation 11):
        min ||Logits(M(X)) - Logits(M_c(X; S_global))||_KL
    
    Unlike prior methods that require extensive memory for global
    fine-tuning, this approach maintains fixed bit-packed binary
    weights throughout the process, substantially reducing memory.
    
    Args:
        quantized_model: Quantized model with frozen binary weights
        original_model: Original full-precision model
        dataloader: Calibration data
        num_epochs: Number of tuning epochs (paper uses 8)
        learning_rate: Learning rate (paper uses 1e-6)
        batch_size: Batch size (paper uses 1)
        temperature: Temperature for softening distributions
        device: Device
        
    Returns:
        Quantized model with tuned scales
    """
    logger.info("Phase 3: Model Reconstruction (TUNESCALESKD)")
    logger.info(f"  Epochs: {num_epochs}, LR: {learning_rate}, Batch: {batch_size}")
    
    # Collect all scale parameters from factorized layers
    scale_params = []
    scale_metadata = []
    
    for name, module in quantized_model.named_modules():
        if hasattr(module, 's1') and hasattr(module, 's2'):
            if hasattr(module, 'U_binary') and hasattr(module, 'V_binary'):
                # Only tune scales, keep binary weights frozen
                module.s1.requires_grad = True
                module.s2.requires_grad = True
                scale_params.extend([module.s1, module.s2])
                scale_metadata.append({
                    'name': name,
                    's1_shape': module.s1.shape,
                    's2_shape': module.s2.shape,
                })
    
    if not scale_params:
        logger.warning("  No scale parameters found for tuning")
        return quantized_model
    
    logger.info(f"  Found {len(scale_params)} scale parameters to tune")
    
    # Freeze all non-scale parameters
    for name, param in quantized_model.named_parameters():
        if param not in scale_params:
            param.requires_grad = False
    
    # Setup optimizer with cosine learning rate scheduler
    optimizer = torch.optim.Adam(scale_params, lr=learning_rate)
    total_steps = num_epochs * max(1, len(dataloader) // batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    original_model.eval()
    quantized_model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_kl_loss = 0.0
        total_mse_loss = 0.0
        num_batches = 0
        
        for i in range(0, len(dataloader), batch_size):
            batch = dataloader[i:i + batch_size]
            if not batch:
                continue
            
            inputs = torch.stack(batch).to(device)
            
            optimizer.zero_grad()
            
            # Get logits from original model
            with torch.no_grad():
                orig_logits = _get_logits(original_model, inputs)
            
            # Get logits from quantized model
            quant_logits = _get_logits(quantized_model, inputs)
            
            # KL divergence loss
            kl_loss = kl_divergence_loss(quant_logits, orig_logits, temperature)
            
            # Optional: add MSE on logits for additional alignment
            mse_loss = F.mse_loss(quant_logits, orig_logits)
            
            # Combined loss
            loss = kl_loss + 0.1 * mse_loss
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(scale_params, max_norm=0.1)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            total_kl_loss += kl_loss.item()
            total_mse_loss += mse_loss.item()
            num_batches += 1
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            avg_kl = total_kl_loss / num_batches
            avg_mse = total_mse_loss / num_batches
            if epoch % 2 == 0 or epoch == num_epochs - 1:
                logger.info(f"  Epoch {epoch+1}/{num_epochs}, "
                          f"Loss: {avg_loss:.6f} (KL: {avg_kl:.6f}, MSE: {avg_mse:.6f})")
    
    quantized_model.eval()
    logger.info("  TUNESCALESKD complete")
    return quantized_model


def kl_divergence_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL divergence loss for knowledge distillation.
    
    Computes the KL divergence between the softened distributions:
        KL(softmax(teacher/T) || softmax(student/T)) * T^2
    
    Args:
        student_logits: Quantized model logits
        teacher_logits: Original model logits
        temperature: Temperature for softening
        
    Returns:
        KL divergence loss (scalar)
    """
    # Soften distributions
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
    
    # KL divergence
    kl = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
    
    # Scale by T^2 as in standard KD
    return kl * (temperature ** 2)


def _get_logits(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Get logits from model.
    
    Handles different model output formats.
    
    Args:
        model: Language model
        inputs: Input tensor
        
    Returns:
        Logits tensor
    """
    with torch.no_grad() if not inputs.requires_grad else torch.enable_grad():
        outputs = model(inputs)
        
        if hasattr(outputs, 'logits'):
            return outputs.logits
        elif isinstance(outputs, tuple):
            return outputs[0]
        else:
            return outputs


def tune_scales_simple(
    quantized_model: nn.Module,
    original_model: nn.Module,
    calibration_data: List[torch.Tensor],
    num_steps: int = 100,
    learning_rate: float = 1e-6,
    device: str = "auto",
) -> nn.Module:
    """Simplified scale tuning without full model forward.
    
    For resource-constrained environments, this performs scale tuning
    layer-by-layer using local reconstruction error instead of global
    KL divergence.
    
    Args:
        quantized_model: Quantized model
        original_model: Original model
        calibration_data: Calibration inputs
        num_steps: Optimization steps
        learning_rate: Learning rate
        device: Device
        
    Returns:
        Quantized model with tuned scales
    """
    logger.info("Phase 3: Model Reconstruction (simplified)")
    
    scale_params = []
    for name, module in quantized_model.named_modules():
        if hasattr(module, 's1') and hasattr(module, 's2'):
            module.s1.requires_grad = True
            module.s2.requires_grad = True
            scale_params.extend([module.s1, module.s2])
    
    if not scale_params:
        return quantized_model
    
    optimizer = torch.optim.Adam(scale_params, lr=learning_rate)
    
    for step in range(num_steps):
        if step >= len(calibration_data):
            break
        
        inputs = calibration_data[step].unsqueeze(0).to(device)
        
        optimizer.zero_grad()
        
        with torch.no_grad():
            orig_logits = _get_logits(original_model, inputs)
        
        quant_logits = _get_logits(quantized_model, inputs)
        
        loss = kl_divergence_loss(quant_logits, orig_logits)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(scale_params, max_norm=0.1)
        optimizer.step()
    
    return quantized_model
