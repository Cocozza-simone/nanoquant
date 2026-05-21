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
import gc

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
    if tau_percentile is not None:
        threshold = values.abs().quantile(tau_percentile / 100.0) * (1 + gamma)
    elif tau is not None:
        threshold = tau * (1 + gamma)
    else:
        threshold = float('inf')

    result = torch.clamp(values, -threshold, threshold)
    result = torch.abs(result) + 1e-8

    mean_val = result.mean()
    result = (1 - shrinkage) * result + shrinkage * mean_val

    return result


class _OnlineSecondMoment:
    """Accumulates second moments online with minimal memory footprint.

    Instead of storing all activation/gradient tensors, only tracks ``count``
    and ``sum_sq``, reducing memory from GB per layer to a few MB.
    """

    __slots__ = ("count", "sum_sq", "target_device")

    def __init__(self, target_device: str = "cpu"):
        self.count = 0
        self.sum_sq = None
        self.target_device = target_device

    def update(self, x: torch.Tensor) -> None:
        """Accumulate second moment of *x*.

        *x* has shape ``[..., d]`` where last dimension is the feature dim.
        All other dimensions are marginalised.
        """
        d = x.shape[-1]
        if self.sum_sq is None:
            self.sum_sq = torch.zeros(d, device=self.target_device, dtype=torch.float32)

        sq = (x.float() ** 2).sum(dim=tuple(range(x.ndim - 1)))

        if str(sq.device) != self.target_device:
            sq = sq.to(self.target_device)

        self.sum_sq = self.sum_sq.to(sq.device) + sq
        self.count += x.shape[:-1].numel()

    def mean_sq(self):
        """Return mean per-feature E[x^2] or ``None`` if empty."""
        if self.sum_sq is None or self.count == 0:
            return None
        return self.sum_sq / self.count


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
                if isinstance(input, tuple):
                    inp = input[0]
                else:
                    inp = input
                if inp is not None:
                    self._activations[name] = inp.detach()
            return hook

        def get_gradient(name):
            def hook(module, grad_input, grad_output):
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

        texts = []
        for i, example in enumerate(dataset):
            if i >= self.config.calib_samples:
                break
            text = example.get("text", "")
            if text.strip():
                texts.append(text)

        encodings = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.config.calib_seq_len,
            padding="max_length",
            return_tensors="pt",
        )

        return encodings["input_ids"], encodings["attention_mask"]

    def _post_process_diag(self, D: torch.Tensor, tau: float, gamma: float) -> torch.Tensor:
        """Apply outlier clipping and shrinkage regularisation to a diagonal estimate.

        Extracted from :py:meth:`robust_diag` so it can be reused for
        online-statistic estimates as well.

        Args:
            D: Raw diagonal estimate [feature_dim]
            tau: Clipping threshold for outlier removal
            gamma: Shrinkage coefficient

        Returns:
            Post-processed diagonal [feature_dim]
        """
        median = torch.median(D)
        clip_val = tau * median
        D = torch.clamp(D, max=clip_val)
        D_mean = torch.mean(D)
        return (1 - gamma) * D + gamma * D_mean

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
        D = torch.mean(samples ** 2, dim=0)  # [feature_dim]
        return self._post_process_diag(D, tau, gamma)

    @torch.no_grad()
    def compute_preconditioners(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Compute diagonal preconditioners D_in and D_out for all linear layers.

        Returns:
            Tuple of (D_in, D_out) dictionaries mapping layer names to diagonal vectors.
        """
        logger.info("Starting global calibration...")
        self.model.eval()

        input_ids, attention_mask = self._load_calibration_data()
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        self._compute_preconditioners_with_hooks(input_ids, attention_mask)

        logger.info("Global calibration complete.")
        return self.D_in, self.D_out

    def _compute_preconditioners_with_hooks(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """Compute preconditioners using forward/backward hooks with online stats."""
        target_device = "cpu"  # Accumulate on CPU to save GPU memory

        # Phase 1: collect activation stats via forward hooks (online, O(MB) memory)
        act_stats: Dict[str, _OnlineSecondMoment] = {}

        def make_fwd_hook(name):
            def hook(module, input, output):
                if isinstance(input, tuple) and input[0] is not None:
                    x = input[0].detach()
                    if str(x.device) == "mps":
                        x = x.cpu()
                    # x shape: [batch, seq_len, d_in]
                    if name not in act_stats:
                        act_stats[name] = _OnlineSecondMoment(target_device)
                    act_stats[name].update(x)
            return hook

        handles = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                h = module.register_forward_hook(make_fwd_hook(name))
                handles.append(h)

        num_samples = len(input_ids)
        logger.info(f"Collecting activation statistics from {num_samples} samples...")

        for i in range(num_samples):
            if i % 20 == 0:
                logger.info(f"  Sample {i}/{num_samples}")

            inp = input_ids[i:i+1]
            mask = attention_mask[i:i+1]

            # Move to device if needed
            if str(inp.device) != str(self.model.device):
                inp = inp.to(self.model.device)
            if str(mask.device) != str(self.model.device):
                mask = mask.to(self.model.device)

            self.model(inp, attention_mask=mask)

            if self.config.device == "mps" and (i + 1) % 10 == 0:
                gc.collect()
                torch.mps.empty_cache()

        for h in handles:
            h.remove()

        # Compute D_in from online statistics
        logger.info("Computing input-side preconditioners...")
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if name in act_stats:
                mean_sq = act_stats[name].mean_sq()
                # mean_sq is None only if stats were never updated (shouldn't happen)
                if mean_sq is not None:
                    D_in = self._post_process_diag(
                        mean_sq,
                        tau=self.config.shrinkage_tau,
                        gamma=self.config.shrinkage_gamma,
                    )
                    self.D_in[name] = D_in.to(self.device)
                else:
                    d_out, d_in = module.weight.shape
                    self.D_in[name] = torch.ones(d_in, device=self.device)
            else:
                d_out, d_in = module.weight.shape
                self.D_in[name] = torch.ones(d_in, device=self.device)

        # Clean up activation stats before gradient pass
        del act_stats
        gc.collect()

        # Phase 2: collect gradient stats via backward hooks (online, O(MB) memory)
        grad_stats: Dict[str, _OnlineSecondMoment] = {}

        def make_bwd_hook(name):
            def hook(module, grad_input, grad_output):
                if isinstance(grad_output, tuple):
                    g = grad_output[0]
                else:
                    g = grad_output
                if g is not None:
                    g = g.detach()
                    if str(g.device) == "mps":
                        g = g.cpu()
                    if name not in grad_stats:
                        grad_stats[name] = _OnlineSecondMoment(target_device)
                    grad_stats[name].update(g)
            return hook

        handles = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                h = module.register_full_backward_hook(make_bwd_hook(name))
                handles.append(h)

        # Limit gradient samples to avoid excessive memory/time
        grad_num_samples = min(num_samples, 32)
        logger.info(f"Collecting gradient statistics from {grad_num_samples} samples...")

        for i in range(grad_num_samples):
            if i % 10 == 0:
                logger.info(f"  Gradient sample {i}/{grad_num_samples}")

            with torch.enable_grad():
                self.model.zero_grad()
                inp = input_ids[i:i+1]
                mask = attention_mask[i:i+1]
                outputs = self.model(inp, attention_mask=mask, labels=inp)
                loss = outputs.loss
                loss.backward()

        for h in handles:
            h.remove()

        # Compute D_out from online statistics
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if name in grad_stats:
                mean_sq = grad_stats[name].mean_sq()
                if mean_sq is not None:
                    D_out = self._post_process_diag(
                        mean_sq,
                        tau=self.config.shrinkage_tau,
                        gamma=self.config.shrinkage_gamma,
                    )
                    self.D_out[name] = D_out.to(self.device)
                else:
                    d_out, d_in = module.weight.shape
                    self.D_out[name] = torch.ones(d_out, device=self.device)
            else:
                d_out, d_in = module.weight.shape
                self.D_out[name] = torch.ones(d_out, device=self.device)

        logger.info(f"Computed preconditioners for {len(self.D_in)} layers")
