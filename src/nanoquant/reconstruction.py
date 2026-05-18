"""
Block Reconstruction Pipeline and Model Reconstruction for NANOQUANT.

Implements the three-step optimization process for each transformer block:
1. Error Propagation Mitigation
2. Low-Rank Binary Initialization (ADMM)
3. Factorized Component Refinement (STE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple, Any
from .admm import LatentBinaryADMM

logger = logging.getLogger(__name__)


def straight_through_sign(x: torch.Tensor) -> torch.Tensor:
    """Straight-Through Estimator (STE) for sign function.
    
    Forward: returns sign(x) in {-1, +1}
    Backward: passes gradient through unchanged
    
    Args:
        x: Input tensor
        
    Returns:
        Binarized tensor with straight-through gradients
    """
    # STE: forward uses sign, backward passes gradient through identity
    # Forward: sign(x), Backward: grad flows through x
    y = torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
    return x + (y - x).detach()


class FactorizedLinear(nn.Module):
    """Factorized linear layer with binary matrices and scale vectors.
    
    Implements W ≈ s1 ⊙ (U±1 V±1^T) ⊙ s2^T
    """
    
    def __init__(
        self,
        d_out: int,
        d_in: int,
        rank: int,
        U: torch.Tensor,
        V: torch.Tensor,
        s1: torch.Tensor,
        s2: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.d_out = d_out
        self.d_in = d_in
        self.rank = rank
        
        # Continuous latent variables (will be optimized with STE)
        self.U_latent = nn.Parameter(U.clone())
        self.V_latent = nn.Parameter(V.clone())
        
        # Scale vectors
        self.s1 = nn.Parameter(s1.clone())
        self.s2 = nn.Parameter(s2.clone())
        
        # Bias
        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
        else:
            self.register_parameter("bias", None)
        
        # Store binary weights (updated after refinement)
        self.register_buffer("U_binary", torch.sign(U))
        self.register_buffer("V_binary", torch.sign(V))
        self.packed = False
    
    def get_binary_weight(self) -> torch.Tensor:
        """Get the binary approximation of the weight matrix."""
        if self.packed:
            U = self.U_binary
            V = self.V_binary
        else:
            U = straight_through_sign(self.U_latent)
            V = straight_through_sign(self.V_latent)
        
        # W ≈ s1 ⊙ (U V^T) ⊙ s2^T
        W = self.s1.unsqueeze(1) * (U @ V.T) * self.s2.unsqueeze(0)
        return W
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using factorized weights.
        
        Args:
            x: Input [..., d_in]
            
        Returns:
            Output [..., d_out]
        """
        W = self.get_binary_weight()
        return F.linear(x, W, self.bias)
    
    def pack(self):
        """Pack binary weights, freeze continuous variables."""
        with torch.no_grad():
            self.U_binary.copy_(torch.sign(self.U_latent))
            self.V_binary.copy_(torch.sign(self.V_latent))
        self.packed = True
        self.U_latent.requires_grad = False
        self.V_latent.requires_grad = False
    
    def get_packed_size(self) -> int:
        """Get memory size in bits."""
        # U_binary: d_out * rank bits
        # V_binary: d_in * rank bits
        # s1: d_out * 32 bits (float32)
        # s2: d_in * 32 bits (float32)
        binary_bits = self.d_out * self.rank + self.d_in * self.rank
        scale_bits = (self.d_out + self.d_in) * 32
        return binary_bits + scale_bits


class BlockReconstructionPipeline:
    """Block-wise reconstruction pipeline for NANOQUANT.
    
    Implements the three-step process from Section 3.2:
    1. Error Propagation Mitigation
    2. Low-Rank Binary Initialization (ADMM + Magnitude Balancing)
    3. Factorized Component Refinement (STE)
    """
    
    def __init__(
        self,
        config,
        D_in: Dict[str, torch.Tensor],
        D_out: Dict[str, torch.Tensor],
    ):
        self.config = config
        self.D_in = D_in
        self.D_out = D_out
        self.admm = LatentBinaryADMM(
            rank=config.rank,
            num_iterations=config.admm_iterations,
            rho=config.admm_rho,
            lambda_reg=config.admm_lambda,
            epsilon=config.admm_epsilon,
            device=config.device,
        )
    
    def _get_effective_rank(self, layer_name: str, d_out: int, d_in: int) -> int:
        """Calculate effective rank for target bit-width.
        
        For sub-1-bit compression (bits < 1.0), adjusts the rank
        to achieve the target effective bit rate.
        
        effective_bits = (2 * rank * (d_out + d_in) * 1 + (d_out + d_in) * 32) / (d_out * d_in * 16)
        
        For target_bits < 1, we solve for rank:
        rank ≈ (target_bits * d_out * d_in * 16 - (d_out + d_in) * 32) / (2 * (d_out + d_in))
        
        Args:
            layer_name: Name of the layer
            d_out: Output dimension
            d_in: Input dimension
            
        Returns:
            Adjusted rank
        """
        target_bits = self.config.bits
        
        if target_bits >= 1.0:
            return self.config.rank
        
        # Calculate rank for target effective bits
        total_params = d_out * d_in
        scale_overhead = (d_out + d_in) * 32  # scale params in bits
        
        # Solve: target_bits = (2 * rank * (d_out + d_in) + scale_overhead) / total_params
        # rank = (target_bits * total_params - scale_overhead) / (2 * (d_out + d_in))
        rank_for_bits = (target_bits * total_params - scale_overhead) / (2 * (d_out + d_in))
        rank = max(1, int(rank_for_bits))
        rank = min(rank, self.config.rank)
        
        if rank != self.config.rank:
            logger.info(f"  {layer_name}: rank adjusted {self.config.rank} -> {rank} for {target_bits:.2f} bits")
        
        return rank
    
    def reconstruct_block(
        self,
        model: nn.Module,
        block: nn.Module,
        block_name: str,
        X: torch.Tensor,
        Y_star: torch.Tensor,
        linear_layers: List[Tuple[str, nn.Linear]],
    ) -> List[FactorizedLinear]:
        """Reconstruct a single transformer block.
        
        Args:
            model: Full model
            block: Current transformer block
            block_name: Name of the block
            X: Block inputs
            Y_star: Target block outputs
            linear_layers: List of (name, layer) tuples in this block
            
        Returns:
            List of FactorizedLinear layers
        """
        logger.info(f"Reconstructing block: {block_name}")
        
        # Step 1: Error Propagation Mitigation
        self._error_propagation_mitigation(block, X, Y_star)
        
        # Step 2: Low-Rank Binary Initialization
        factorized_layers = []
        for layer_name, linear_layer in linear_layers:
            full_name = f"{block_name}.{layer_name}" if block_name else layer_name
            fl = self._initialize_binary_factors(linear_layer, full_name)
            factorized_layers.append(fl)
        
        # Step 3: Factorized Component Refinement
        self._factorized_component_refinement(
            block, factorized_layers, linear_layers, X, Y_star
        )
        
        return factorized_layers
    
    def _error_propagation_mitigation(
        self,
        block: nn.Module,
        X: torch.Tensor,
        Y_star: torch.Tensor,
    ):
        """Step 1: Tune full-precision weights to minimize accumulated error.
        
        This step adjusts the weights of the current block to account for
        quantization errors from previously processed blocks.
        """
        logger.info("  Step 1: Error Propagation Mitigation")
        
        # Make block parameters trainable
        for param in block.parameters():
            param.requires_grad = True
        
        optimizer = torch.optim.Adam(block.parameters(), lr=self.config.pre_tune_lr)
        
        block.train()
        for step in range(self.config.pre_tune_steps):
            optimizer.zero_grad()
            
            # Forward pass through block
            if isinstance(X, tuple):
                Y = block(*X)
            else:
                Y = block(X)
            
            if isinstance(Y, tuple):
                Y = Y[0]
            
            # Compute loss against target
            loss = F.mse_loss(Y, Y_star)
            
            loss.backward()
            optimizer.step()
            
            if step % 10 == 0:
                logger.debug(f"    Pre-tune step {step}: loss={loss.item():.6f}")
        
        block.eval()
        
        # Freeze parameters
        for param in block.parameters():
            param.requires_grad = False
    
    def _initialize_binary_factors(
        self,
        linear_layer: nn.Linear,
        full_name: str,
    ) -> FactorizedLinear:
        """Step 2: Initialize low-rank binary factors via ADMM.
        
        Args:
            linear_layer: Original linear layer
            full_name: Full layer name for looking up preconditioners
            
        Returns:
            FactorizedLinear with initialized parameters
        """
        logger.info(f"  Step 2: Binary Initialization for {full_name}")
        
        W = linear_layer.weight.data  # [d_out, d_in]
        d_out, d_in = W.shape
        
        # Get effective rank for sub-1-bit compression
        effective_rank = self._get_effective_rank(full_name, d_out, d_in)
        
        # Get preconditioners
        D_in = self.D_in.get(full_name, torch.ones(d_in, device=self.config.device))
        D_out = self.D_out.get(full_name, torch.ones(d_out, device=self.config.device))
        
        # Apply preconditioning: W_f = D_out^{1/2} W D_in^{1/2}
        D_out_sqrt = D_out.sqrt()
        D_in_sqrt = D_in.sqrt()
        
        W_f = D_out_sqrt.unsqueeze(1) * W * D_in_sqrt.unsqueeze(0)
        
        # Solve via ADMM with effective rank
        if effective_rank != self.admm.rank:
            # Create temporary ADMM solver with adjusted rank
            temp_admm = LatentBinaryADMM(
                rank=effective_rank,
                num_iterations=self.config.admm_iterations,
                rho=self.config.admm_rho,
                lambda_reg=self.config.admm_lambda,
                epsilon=self.config.admm_epsilon,
                device=self.config.device,
            )
            U, V, s1, s2 = temp_admm.solve(W_f, D_in, D_out)
        else:
            U, V, s1, s2 = self.admm.solve(W_f, D_in, D_out)
        
        # De-precondition the scales
        s1 = s1 / (D_out_sqrt + 1e-8)
        s2 = s2 / (D_in_sqrt + 1e-8)
        
        logger.info(f"    Initialized: U{U.shape}, V{V.shape}, ranks={self.config.rank}")
        
        # Create factorized layer
        factorized = FactorizedLinear(
            d_out=d_out,
            d_in=d_in,
            rank=self.config.rank,
            U=U,
            V=V,
            s1=s1,
            s2=s2,
            bias=linear_layer.bias.data if linear_layer.bias is not None else None,
        )
        
        return factorized
    
    def _factorized_component_refinement(
        self,
        block: nn.Module,
        factorized_layers: List[FactorizedLinear],
        linear_layers: List[Tuple[str, nn.Linear]],
        X: torch.Tensor,
        Y_star: torch.Tensor,
    ):
        """Step 3: Refine factorized components using STE.
        
        Jointly tunes U, V and s1, s2 to align with full-precision block outputs.
        Implements Equation (10) from the paper.
        """
        logger.info("  Step 3: Factorized Component Refinement")
        
        # Replace linear layers with factorized versions
        name_to_factorized = {}
        for (layer_name, _), fl in zip(linear_layers, factorized_layers):
            name_to_factorized[layer_name] = fl
        
        # Replace modules in block
        for layer_name, fl in name_to_factorized.items():
            parent_name = ".".join(layer_name.split(".")[:-1])
            child_name = layer_name.split(".")[-1]
            
            if parent_name:
                parent = block.get_submodule(parent_name)
            else:
                parent = block
            
            setattr(parent, child_name, fl)
        
        # Make factorized parameters trainable
        for fl in factorized_layers:
            fl.U_latent.requires_grad = True
            fl.V_latent.requires_grad = True
            fl.s1.requires_grad = True
            fl.s2.requires_grad = True
        
        optimizer = torch.optim.Adam(
            [p for fl in factorized_layers for p in [fl.U_latent, fl.V_latent, fl.s1, fl.s2]],
            lr=self.config.post_tune_lr,
        )
        
        block.train()
        for step in range(self.config.post_tune_steps):
            optimizer.zero_grad()
            
            # Forward pass
            if isinstance(X, tuple):
                Y = block(*X)
            else:
                Y = block(X)
            
            if isinstance(Y, tuple):
                Y = Y[0]
            
            # MSE loss (Equation 10)
            loss = F.mse_loss(Y, Y_star)
            
            loss.backward()
            optimizer.step()
            
            if step % 10 == 0:
                logger.debug(f"    Post-tune step {step}: loss={loss.item():.6f}")
        
        # Pack binary weights
        for fl in factorized_layers:
            fl.pack()
        
        block.eval()
        logger.info("  Block reconstruction complete")


class ModelReconstruction:
    """Global model reconstruction phase.
    
    Optimizes floating-point scaling vectors to align logits
    of quantized model with original predictions.
    Implements Equation (11) from the paper.
    """
    
    def __init__(self, config):
        self.config = config
    
    def reconstruct(
        self,
        quantized_model: nn.Module,
        original_model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """Global scale tuning via KL divergence minimization.
        
        Args:
            quantized_model: Quantized model with packed binary weights
            original_model: Original full-precision model
            input_ids: Input tokens for calibration
            attention_mask: Attention mask
        """
        logger.info("Starting Model Reconstruction (Global Scale Tuning)...")
        
        original_model.eval()
        quantized_model.train()
        
        # Collect all scale parameters from factorized layers
        scale_params = []
        for module in quantized_model.modules():
            if isinstance(module, FactorizedLinear):
                module.s1.requires_grad = True
                module.s2.requires_grad = True
                scale_params.extend([module.s1, module.s2])
        
        # If no FactorizedLinear modules found, skip reconstruction
        if len(scale_params) == 0:
            logger.warning("No FactorizedLinear modules found in model. Skipping Model Reconstruction.")
            return
        
        optimizer = torch.optim.Adam(scale_params, lr=self.config.glob_tune_lr)
        
        # Get original logits
        with torch.no_grad():
            orig_outputs = original_model(input_ids, attention_mask=attention_mask)
            orig_logits = orig_outputs.logits
        
        for step in range(self.config.glob_tune_steps):
            optimizer.zero_grad()
            
            # Forward pass through quantized model
            outputs = quantized_model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # KL divergence loss (Equation 11)
            # KL(orig || quant) = sum(orig_probs * log(orig_probs / quant_probs))
            orig_probs = F.softmax(orig_logits, dim=-1)
            log_quant_probs = F.log_softmax(logits, dim=-1)
            
            loss = F.kl_div(
                log_quant_probs,
                orig_probs,
                reduction="batchmean",
                log_target=False,
            )
            
            loss.backward()
            optimizer.step()
            
            if step % 10 == 0:
                logger.info(f"  Global tuning step {step}: KL={loss.item():.6f}")
        
        quantized_model.eval()
        
        # Freeze scales
        for module in quantized_model.modules():
            if isinstance(module, FactorizedLinear):
                module.s1.requires_grad = False
                module.s2.requires_grad = False
        
        logger.info("Model reconstruction complete")
