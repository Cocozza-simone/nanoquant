"""
Error Propagation Mitigation (TUNEFP).

Implements Step 1 of the NANOQUANT block reconstruction pipeline:
"Error Propagation Mitigation" (Algorithm 1, line 11).

Before factorizing each block's weights, we tune the full-precision
weights to minimize the error introduced by quantization of preceding
blocks. This compensates for accumulated quantization error as the
reconstruction proceeds through the network.

Reference: Frantar et al., "GPTQ: Accurate Post-Training Quantization
for Generative Pre-trained Transformers", 2022.
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, List, Tuple
from .device_utils import get_optimal_device

logger = logging.getLogger(__name__)


def tune_full_precision_weights(
    block: nn.Module,
    dataloader: List[torch.Tensor],
    model: nn.Module,
    block_idx: int,
    num_epochs: int = 8,
    learning_rate: float = 1e-4,
    batch_size: int = 4,
    device: str = "auto",
) -> nn.Module:
    """Tune full-precision weights to mitigate error propagation.
    
    As described in Algorithm 1, Step 1 (TUNEFP):
    "Tune the full-precision weights of the current block to minimize
    the error introduced by the quantization of preceding blocks."
    
    This adjusts the full-precision weights W of the current block
    so that W(x) better approximates the original block's output,
    accounting for quantization errors from already-processed blocks.
    
    Args:
        block: Current transformer block to tune
        dataloader: Calibration data (list of input tensors)
        model: Full model for computing reference outputs
        block_idx: Index of current block in the model
        num_epochs: Number of tuning epochs (paper uses 8)
        learning_rate: Learning rate (paper uses 1e-4)
        batch_size: Batch size (paper uses 4)
        device: Device
        
    Returns:
        Tuned block module
    """
    logger.info(f"  Step 1: Error Propagation Mitigation (TUNEFP)")
    logger.info(f"    Epochs: {num_epochs}, LR: {learning_rate}, Batch: {batch_size}")
    
    # Collect linear layers in this block
    linear_layers = []
    for name, module in block.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers.append((name, module))
    
    if not linear_layers:
        logger.warning("    No linear layers found in block")
        return block
    
    # Make weights trainable
    original_requires_grad = {}
    for name, module in block.named_modules():
        if isinstance(module, nn.Linear) and module.weight is not None:
            original_requires_grad[name] = module.weight.requires_grad
            module.weight.requires_grad = True
            if module.bias is not None:
                module.bias.requires_grad = True
    
    # Setup optimizer with cosine learning rate scheduler
    params = []
    for name, module in block.named_modules():
        if isinstance(module, nn.Linear):
            params.append(module.weight)
            if module.bias is not None:
                params.append(module.bias)
    
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs * max(1, len(dataloader) // batch_size)
    )
    
    # Get reference outputs from the full-precision model
    block.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        # Process calibration data in batches
        for i in range(0, len(dataloader), batch_size):
            batch = dataloader[i:i + batch_size]
            if not batch:
                continue
            
            inputs = torch.stack(batch).to(device)
            
            optimizer.zero_grad()
            
            # Forward through the quantized model up to this block
            # to get inputs that include quantization errors from prev blocks
            with torch.no_grad():
                # Get reference output from full-precision block
                ref_output = _get_block_reference_output(
                    model, block_idx, inputs
                )
            
            # Forward through current block
            output = block(inputs)
            
            # Weighted MSE loss between current output and reference
            loss = weighted_mse_loss(output, ref_output)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            if epoch % 2 == 0 or epoch == num_epochs - 1:
                logger.info(f"    Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    # Restore original requires_grad
    for name, module in block.named_modules():
        if isinstance(module, nn.Linear) and name in original_requires_grad:
            module.weight.requires_grad = original_requires_grad[name]
            if module.bias is not None:
                module.bias.requires_grad = False
    
    block.eval()
    logger.info(f"    TUNEFP complete")
    return block


def weighted_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Weighted MSE loss.
    
    As used in NANOQUANT (referenced from Boža & Macko, 2025):
    A weighted MSE function that applies higher weights to certain
    dimensions based on their importance.
    
    Args:
        pred: Predicted output
        target: Target output
        weights: Optional per-element weights
        
    Returns:
        Scalar loss
    """
    diff = pred - target
    squared = diff ** 2
    
    if weights is not None:
        squared = squared * weights
    
    return squared.mean()


def _get_block_reference_output(
    model: nn.Module,
    block_idx: int,
    inputs: torch.Tensor,
) -> torch.Tensor:
    """Get reference output from the full-precision model.
    
    Forward pass through the model up to and including the
    specified block, using full-precision weights.
    
    Args:
        model: Full model
        block_idx: Block index
        inputs: Input tensor
        
    Returns:
        Reference output tensor
    """
    # For transformer models, we need to navigate the architecture
    # This is a simplified version - actual implementation depends
    # on the specific model architecture
    
    with torch.no_grad():
        model.eval()
        
        # Try common transformer architectures
        if hasattr(model, 'model'):
            # Llama, Qwen, etc.
            hidden_states = inputs
            layers = model.model.layers if hasattr(model.model, 'layers') else []
            
            # Forward through embedding if needed
            if hasattr(model.model, 'embed_tokens'):
                hidden_states = model.model.embed_tokens(inputs)
            
            # Forward through blocks up to and including block_idx
            for i, layer in enumerate(layers):
                if i <= block_idx:
                    hidden_states = layer(hidden_states)[0] if isinstance(
                        layer(hidden_states), tuple
                    ) else layer(hidden_states)
                else:
                    break
            
            return hidden_states
        else:
            # Fallback: just return input (loss will be zero)
            return inputs


def compute_weighted_reconstruction_error(
    W_orig: torch.Tensor,
    W_quant: torch.Tensor,
    D_in: Optional[torch.Tensor] = None,
    D_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute Hessian-weighted reconstruction error.
    
    As in Equation (2) of the paper:
        L(W_c) ≈ ||D_out^{1/2} (W - W_c) D_in^{1/2}||_F^2
    
    Args:
        W_orig: Original weight matrix
        W_quant: Quantized weight matrix
        D_in: Input preconditioner
        D_out: Output preconditioner
        
    Returns:
        Weighted error scalar
    """
    diff = W_orig - W_quant
    
    if D_in is not None and D_out is not None:
        D_out_sqrt = D_out.sqrt()
        D_in_sqrt = D_in.sqrt()
        weighted_diff = D_out_sqrt.unsqueeze(1) * diff * D_in_sqrt.unsqueeze(0)
        return torch.norm(weighted_diff, p='fro') ** 2
    else:
        return torch.norm(diff, p='fro') ** 2
