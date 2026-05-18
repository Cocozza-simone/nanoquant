"""
Global Calibration module for NANOQUANT.

Implements Hessian-aware preconditioning using K-FAC approximation
with shrinkage regularization for robust diagonal estimation.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import logging
import numpy as np

logger = logging.getLogger(__name__)


def robust_diag_estimator(
    values: torch.Tensor,
    tau_percentile: Optional[float] = None,
    tau: Optional[float] = None,
    gamma: float = 0.2,
    shrinkage: float = 0.1,
) -> torch.Tensor:
    """Robust diagonal estimator with outlier clipping.
    
    Estimates robust diagonal values with clipping for outliers.
    Can use either percentile-based or absolute threshold.
    
    Args:
        values: Input values [d]
        tau_percentile: Percentile threshold (0-100) for determining outlier limit
        tau: Absolute threshold for clipping
        gamma: Scaling factor applied with tau
        shrinkage: Shrinkage coefficient for regularization (0-1)
        
    Returns:
        Clipped diagonal values [d]
    """
    # Determine clipping threshold
    if tau_percentile is not None:
        # Use percentile-based threshold
        threshold = values.abs().quantile(tau_percentile / 100.0) * (1 + gamma)
    elif tau is not None:
        # Use absolute threshold with gamma scaling
        threshold = tau * (1 + gamma)
    else:
        # No clipping
        threshold = float('inf')
    
    # Clip values at threshold
    result = torch.clamp(values, -threshold, threshold)
    
    # Ensure positive (take absolute values and shift)
    result = torch.abs(result) + 1e-8
    
    # Apply shrinkage for stability
    mean_val = result.mean()
    result = (1 - shrinkage) * result + shrinkage * mean_val
    
    return result


class GlobalCalibration:
    """Global calibration for computing preconditioning matrices.
    
    Uses K-FAC (Kronecker-factored Approximate Curvature) to estimate
    diagonal preconditioners D_in and D_out for each linear layer.
    Applies shrinkage regularization for robustness.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = config.device
        
        # Storage for preconditioners
        self.D_in: Dict[str, torch.Tensor] = {}
        self.D_out: Dict[str, torch.Tensor] = {}
        
        # Storage for activations and gradients
        self._activations: Dict[str, torch.Tensor] = {}
        self._gradients: Dict[str, torch.Tensor] = {}
        
    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients."""
        self._handles = []
        
        def get_activation(name):
            def hook(module, input, output):
                # Store input activations
                if isinstance(input, tuple):
                    inp = input[0]
                else:
                    inp = input
                if inp is not None:
                    self._activations[name] = inp.detach()
            return hook
        
        def get_gradient(name):
            def hook(module, grad_input, grad_output):
                # Store output gradients
                if isinstance(grad_output, tuple):
                    grad = grad_output[0]
                else:
                    grad = grad_output
                if grad is not None:
                    self._gradients[name] = grad.detach()
            return hook
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                handle_fwd = module.register_forward_hook(get_activation(name))
                handle_bwd = module.register_full_backward_hook(get_gradient(name))
                self._handles.extend([handle_fwd, handle_bwd])
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles = []
        
    def _load_calibration_data(self):
        """Load calibration dataset."""
        logger.info(f"Loading calibration dataset: {self.config.calib_dataset}/{self.config.calib_config}")
        
        dataset = load_dataset(
            self.config.calib_dataset,
            self.config.calib_config,
            split="train",
            trust_remote_code=True
        )
        
        # Tokenize
        texts = []
        for i, example in enumerate(dataset):
            if i >= self.config.calib_samples:
                break
            text = example.get("text", "")
            if text.strip():
                texts.append(text)
        
        # Tokenize all texts
        encodings = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.config.calib_seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        
        return encodings["input_ids"], encodings["attention_mask"]
    
    def robust_diag(
        self,
        samples: torch.Tensor,
        tau: float = 10.0,
        gamma: float = 0.2,
    ) -> torch.Tensor:
        """Compute robust diagonal with shrinkage regularization.
        
        Implements Equation (3) from the paper:
        [D]ii <- (1-gamma)[D]ii + gamma * mean(D)
        
        Args:
            samples: Tensor of shape [num_samples, feature_dim]
            tau: Clipping threshold for outlier removal
            gamma: Shrinkage coefficient
            
        Returns:
            Regularized diagonal matrix
        """
        # Compute empirical diagonal (second moment)
        D = torch.mean(samples ** 2, dim=0)  # [feature_dim]
        
        # Clip outliers
        median = torch.median(D)
        clip_val = tau * median
        D = torch.clamp(D, max=clip_val)
        
        # Shrinkage regularization
        D_mean = torch.mean(D)
        D_reg = (1 - gamma) * D + gamma * D_mean
        
        return D_reg
    
    @torch.no_grad()
    def compute_preconditioners(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Compute diagonal preconditioners D_in and D_out for all linear layers.
        
        Returns:
            Tuple of (D_in, D_out) dictionaries mapping layer names to diagonal vectors.
        """
        logger.info("Starting global calibration...")
        self.model.eval()
        
        # Load calibration data
        input_ids, attention_mask = self._load_calibration_data()
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # Storage for accumulated statistics
        act_stats: Dict[str, list] = {}
        grad_stats: Dict[str, list] = {}
        
        num_samples = min(self.config.calib_samples, len(input_ids))
        
        logger.info(f"Processing {num_samples} calibration samples...")
        
        for i in range(num_samples):
            if i % 10 == 0:
                logger.info(f"Calibration sample {i}/{num_samples}")
            
            # Enable gradient computation for this sample
            with torch.enable_grad():
                self.model.zero_grad()
                
                # Forward pass
                inp = input_ids[i:i+1]
                mask = attention_mask[i:i+1]
                
                outputs = self.model(inp, attention_mask=mask, labels=inp)
                loss = outputs.loss
                
                # Backward pass to get gradients
                loss.backward()
                
                # Collect statistics
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.Linear):
                        if name not in act_stats:
                            act_stats[name] = []
                            grad_stats[name] = []
                        
                        # Get activation (input to linear layer)
                        if hasattr(module, "weight"):
                            # Hook-based collection would be cleaner
                            # For simplicity, we compute from the model's activations
                            weight = module.weight  # [d_out, d_in]
                            d_out, d_in = weight.shape
                            
                            # We need to get the actual activation for this layer
                            # Use a forward hook approach
                            pass
        
        # Alternative: Use a hook-based approach for cleaner implementation
        self._compute_preconditioners_with_hooks(input_ids[:num_samples], attention_mask[:num_samples])
        
        logger.info("Global calibration complete.")
        return self.D_in, self.D_out
    
    def _compute_preconditioners_with_hooks(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """Compute preconditioners using forward/backward hooks with memory optimization."""
        import gc
        
        act_accumulator: Dict[str, list] = {}
        grad_accumulator: Dict[str, list] = {}
        
        handles = []
        
        def make_hook(name, is_input=True):
            def hook(module, input, output):
                if is_input:
                    if isinstance(input, tuple) and input[0] is not None:
                        x = input[0].detach()
                        # Move to CPU for MPS to save GPU memory
                        if str(x.device) == "mps":
                            x = x.cpu()
                        # x shape: [batch, seq_len, d_in]
                        if name not in act_accumulator:
                            act_accumulator[name] = []
                        act_accumulator[name].append(x)
                else:
                    if output is not None:
                        if isinstance(output, tuple):
                            g = output[0].detach() if output[0] is not None else None
                        else:
                            g = output.detach()
                        if g is not None:
                            if str(g.device) == "mps":
                                g = g.cpu()
                            if name not in grad_accumulator:
                                grad_accumulator[name] = []
                            grad_accumulator[name].append(g)
            return hook
        
        # Register forward hooks
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                h = module.register_forward_hook(make_hook(name, is_input=True))
                handles.append(h)
        
        num_samples = len(input_ids)
        logger.info(f"Collecting activation statistics from {num_samples} samples...")
        
        # Collect activations with memory management
        for i in range(num_samples):
            if i % 20 == 0:
                logger.info(f"  Sample {i}/{num_samples}")
            
            with torch.no_grad():
                inp = input_ids[i:i+1]
                mask = attention_mask[i:i+1]
                
                # Move to device if needed
                if str(inp.device) != str(self.model.device):
                    inp = inp.to(self.model.device)
                if str(mask.device) != str(self.model.device):
                    mask = mask.to(self.model.device)
                
                outputs = self.model(inp, attention_mask=mask)
            
            # Periodic memory cleanup for MPS
            if self.config.device == "mps" and (i + 1) % 10 == 0:
                gc.collect()
                torch.mps.empty_cache()
        
        # Remove forward hooks
        for h in handles:
            h.remove()
        
        # Now compute preconditioners from collected data
        logger.info("Computing robust diagonal preconditioners...")
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if name in act_accumulator and len(act_accumulator[name]) > 0:
                    # Concatenate all activations
                    acts = torch.cat(act_accumulator[name], dim=0)  # [total_tokens, d_in]
                    
                    # Ensure acts is on CPU
                    if str(acts.device) != "cpu":
                        acts = acts.cpu()
                    
                    # Flatten to [num_samples * seq_len, d_in]
                    acts_flat = acts.reshape(-1, acts.shape[-1])
                    
                    # Compute D_in (input-side preconditioner)
                    D_in = self.robust_diag(
                        acts_flat,
                        tau=self.config.shrinkage_tau,
                        gamma=self.config.shrinkage_gamma,
                    )
                    self.D_in[name] = D_in.to(self.device)
                    
                    # For D_out, we use gradient statistics
                    # Since we don't have gradients from the no_grad context,
                    # we use the empirical activation statistics to estimate D_out
                    # as a proxy (activations carry curvature information)
                    weight = module.weight.data
                    d_out, d_in = weight.shape
                    
                    # Approximate D_out using activation covariance projected through weights
                    # This approximates the gradient curvature E[gg^T] ≈ E[(Wa)(Wa)^T]
                    act_cov = (acts_flat.T @ acts_flat) / acts_flat.shape[0]  # [d_in, d_in]
                    # Ensure act_cov is on the same device as weight for computation
                    act_cov = act_cov.to(weight.device)
                    weight_cov = weight @ act_cov @ weight.T  # [d_out, d_out]
                    D_out_diag = torch.diagonal(weight_cov).abs().clamp(min=1e-8)
                    
                    # Apply shrinkage to D_out as well
                    D_out_median = torch.median(D_out_diag)
                    D_out_clipped = torch.clamp(D_out_diag, max=self.config.shrinkage_tau * D_out_median)
                    D_out_mean = torch.mean(D_out_clipped)
                    D_out = (1 - self.config.shrinkage_gamma) * D_out_clipped + self.config.shrinkage_gamma * D_out_mean
                    self.D_out[name] = D_out.to(self.device)
                else:
                    # Fallback: use identity preconditioners
                    d_out, d_in = module.weight.shape
                    self.D_in[name] = torch.ones(d_in, device=self.device)
                    self.D_out[name] = torch.ones(d_out, device=self.device)
        
        # Second pass: compute D_out using gradients
        logger.info("Computing output-side preconditioners...")
        
        grad_accumulator.clear()
        
        def make_bwd_hook(name):
            def hook(module, grad_input, grad_output):
                if isinstance(grad_output, tuple):
                    g = grad_output[0]
                else:
                    g = grad_output
                if g is not None:
                    if name not in grad_accumulator:
                        grad_accumulator[name] = []
                    grad_accumulator[name].append(g.detach())
            return hook
        
        handles = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                h = module.register_full_backward_hook(make_bwd_hook(name))
                handles.append(h)
        
        for i in range(min(num_samples, 32)):  # Limit for memory
            if i % 10 == 0:
                logger.info(f"  Gradient sample {i}/{min(num_samples, 32)}")
            
            with torch.enable_grad():
                self.model.zero_grad()
                inp = input_ids[i:i+1]
                mask = attention_mask[i:i+1]
                outputs = self.model(inp, attention_mask=mask, labels=inp)
                loss = outputs.loss
                loss.backward()
        
        # Remove hooks
        for h in handles:
            h.remove()
        
        # Update D_out
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if name in grad_accumulator and len(grad_accumulator[name]) > 0:
                    grads = torch.cat(grad_accumulator[name], dim=0)
                    grads_flat = grads.reshape(-1, grads.shape[-1])
                    
                    D_out = self.robust_diag(
                        grads_flat,
                        tau=self.config.shrinkage_tau,
                        gamma=self.config.shrinkage_gamma,
                    )
                    self.D_out[name] = D_out.to(self.device)
        
        logger.info(f"Computed preconditioners for {len(self.D_in)} layers")
