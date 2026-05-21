"""
Latent Binary ADMM (LB-ADMM) for Low-Rank Binary Factorization.

Implements the ADMM solver for initializing low-rank binary matrices
and scaling vectors, as described in Section 3.2 of the paper.

Integrations:
- Ternary initialization from QMoE (IST-DASLab) for better convergence
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Tuple, Optional
from .svid import svid_rank1_fast

logger = logging.getLogger(__name__)


class LatentBinaryADMM:
    """ADMM solver for latent binary factorization.
    
    Solves the optimization problem (Equation 4):
    min_{U,V,Z} ||W_f - UV^T||_F^2 + lambda/2 (||U||_F^2 + ||V||_F^2)
    s.t. U = Z_U, V = Z_V
    
    where Z are auxiliary variables that will be binarized.
    """
    
    def __init__(
        self,
        rank: int,
        num_iterations: int = 50,
        rho: float = 1.0,
        lambda_reg: float = 0.01,
        epsilon: float = 1e-5,
        device: Optional[str] = None,
        use_ternary_init: bool = True,       # ispirato a QMoE (IST-DASLab)
        ternary_sparsity: float = 0.9,       # 90% dei pesi -> zero prima dell'SVD
        seed: Optional[int] = None,          # seed per riproducibilità (paper Appendix C)
    ):
        self.rank = rank
        self.num_iterations = num_iterations
        self.rho = rho
        self.lambda_reg = lambda_reg
        self.epsilon = epsilon
        self.use_ternary_init = use_ternary_init
        self.ternary_sparsity = ternary_sparsity
        self.seed = seed
        # Auto-detect device: use CPU on macOS (no CUDA support)
        if device is None:
            self.device = "cpu"
        else:
            self.device = device
    
    def solve(
        self,
        W_f: torch.Tensor,
        D_in: Optional[torch.Tensor] = None,
        D_out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve the latent binary factorization via ADMM.
        
        Args:
            W_f: Preconditioned weight matrix [d_out, d_in]
            D_in: Input preconditioner [d_in]
            D_out: Output preconditioner [d_out]
            
        Returns:
            Tuple of (U, V, s1, s2) where:
                U: Latent binary factor [d_out, rank]
                V: Latent binary factor [d_in, rank]
                s1: Output channel scale [d_out]
                s2: Input channel scale [d_in]
        """
        # Fix seed for reproducibility (paper Appendix C)
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
        
        d_out, d_in = W_f.shape
        r = min(self.rank, min(d_out, d_in))
        
        # Inizializzazione U, V
        # DEFAULT (originale): rumore gaussiano casuale
        # NUOVO con ternary_init: SVD sulla proiezione ternaria di W_f
        # (ispirato a QMoE: la sparsita' ternaria ~90% porta l'init
        #  vicino alla soluzione, riducendo le iterazioni ADMM necessarie)
        if self.use_ternary_init:
            try:
                from .ternary_init import ternary_svd_init, estimate_init_quality
                U, V = ternary_svd_init(W_f, rank=r, sparsity=self.ternary_sparsity)
                U = U.to(self.device)
                V = V.to(self.device)
                init_err = estimate_init_quality(W_f, U, V)
                logger.debug(f"Ternary SVD init: errore relativo iniziale = {init_err:.4f}")
            except ImportError:
                logger.warning("ternary_init not available, using Gaussian initialization")
                U = torch.randn(d_out, r, device=self.device) * 0.01
                V = torch.randn(d_in, r, device=self.device) * 0.01
        else:
            # Comportamento originale
            U = torch.randn(d_out, r, device=self.device) * 0.01
            V = torch.randn(d_in, r, device=self.device) * 0.01
        
        # Auxiliary variables
        Z_U = U.clone()
        Z_V = V.clone()
        
        # Dual variables
        Lambda_U = torch.zeros_like(U)
        Lambda_V = torch.zeros_like(V)
        
        logger.debug(f"ADMM: W_f shape {W_f.shape}, rank {r}")
        
        for k in range(self.num_iterations):

            # Step 1: Update U (solve linear system)
            # (V^T V + (rho + lambda)I) U^T = V^T W_f^T + rho(Z_U - Lambda_U)^T
            # Equation (5)
            VtV = V.T @ V  # [r, r]
            A = VtV + (self.rho + self.lambda_reg) * torch.eye(r, device=self.device)

            B = V.T @ W_f.T + self.rho * (Z_U - Lambda_U).T  # [r, d_out]

            # Solve using Cholesky decomposition (O(r^3/3) vs O(2r^3/3))
            try:
                L = torch.linalg.cholesky(A)
                U_T = torch.cholesky_solve(B, L)
                U = U_T.T  # [d_out, r]
            except Exception as e:
                logger.warning(f"Cholesky failed at iter {k}: {e}, using pinv")
                U = (torch.linalg.pinv(A) @ B).T

            # Step 2: Update V (symmetric to U)
            UtU = U.T @ U  # [r, r]
            A_v = UtU + (self.rho + self.lambda_reg) * torch.eye(r, device=self.device)

            B_v = U.T @ W_f + self.rho * (Z_V - Lambda_V).T  # [r, d_in]

            try:
                L_v = torch.linalg.cholesky(A_v)
                V_T = torch.cholesky_solve(B_v, L_v)
                V = V_T.T  # [d_in, r]
            except Exception as e:
                logger.warning(f"Cholesky failed for V at iter {k}: {e}, using pinv")
                V = (torch.linalg.pinv(A_v) @ B_v).T
            
            # Step 3: Update Z using SVID (Equation 6)
            # Use svid_rank1_fast from svid.py (correct implementation)
            P_U = U + Lambda_U
            P_V = V + Lambda_V
            
            Z_U = svid_rank1_fast(P_U)
            Z_V = svid_rank1_fast(P_V)
            
            # Step 4: Update dual variables
            Lambda_U = Lambda_U + U - Z_U
            Lambda_V = Lambda_V + V - Z_V
            
            # Check convergence
            primal_residual_U = torch.norm(U - Z_U, p='fro').item()
            primal_residual_V = torch.norm(V - Z_V, p='fro').item()
            
            if k % 10 == 0:
                obj = torch.norm(W_f - U @ V.T, p='fro').item()
                logger.debug(f"  Iter {k}: obj={obj:.6f}, res_U={primal_residual_U:.6f}, res_V={primal_residual_V:.6f}")
            
            if primal_residual_U < self.epsilon and primal_residual_V < self.epsilon:
                logger.debug(f"ADMM converged at iteration {k}")
                break
        
        # Step 2-3: Latent Magnitude Balancing (Equation 7, 8)
        U, V, s1, s2 = self._magnitude_balancing(U, V, D_out, D_in)
        
        return U, V, s1, s2
    
    def _magnitude_balancing(
        self,
        U: torch.Tensor,
        V: torch.Tensor,
        D_out: Optional[torch.Tensor] = None,
        D_in: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply latent magnitude balancing.
        
        Implements Equations (7), (8), and (9) from the paper.
        
        Args:
            U: ADMM output U [d_out, rank]
            V: ADMM output V [d_in, rank]
            D_out: Output preconditioner [d_out]
            D_in: Input preconditioner [d_in]
            
        Returns:
            Balanced (U, V, s1, s2)
        """
        # shape: U [d_out, rank], V [d_in, rank]

        # Apply preconditioner inverse (Equation 9: depreconditioning)
        # Paper: Û = P_U^(K) · D̃_out^{-1}, so divide by sqrt(D), not multiply
        if D_out is not None:
            U_b = U / (D_out.sqrt().unsqueeze(1) + 1e-8)
        else:
            U_b = U
            
        if D_in is not None:
            V_b = V / (D_in.sqrt().unsqueeze(1) + 1e-8)
        else:
            V_b = V
        
        # Compute equilibrium factor (Equation 7)
        norm_V = torch.norm(V_b, p='fro')
        norm_U = torch.norm(U_b, p='fro')
        
        if norm_U > 1e-8:
            eta = torch.sqrt(norm_V / norm_U)
        else:
            eta = torch.tensor(1.0, device=self.device)
        
        # Compute scales from mean absolute values (Equation 8)
        u_b_rows = U_b * eta  # [d_out, rank]
        v_b_rows = V_b / eta  # [d_in, rank]
        
        s1 = torch.mean(torch.abs(u_b_rows), dim=1)  # [d_out]
        s2 = torch.mean(torch.abs(v_b_rows), dim=1)  # [d_in]
        
        # Ensure scales are positive
        s1 = torch.clamp(s1, min=1e-8)
        s2 = torch.clamp(s2, min=1e-8)
        
        # Define final latent variables (Equation 9)
        U_final = u_b_rows / (s1.unsqueeze(1) + 1e-8)
        V_final = v_b_rows / (s2.unsqueeze(1) + 1e-8)
        
        return U_final, V_final, s1, s2
    
    def solve_simple(
        self,
        W: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Simplified ADMM solver without preconditioning.
        
        Args:
            W: Weight matrix [d_out, d_in]
            
        Returns:
            (U, V, s1, s2)
        """
        logger.info(f"Running ADMM on weight matrix {W.shape}")
        U, V, s1, s2 = self.solve(W, None, None)
        logger.info(f"ADMM complete. U:{U.shape}, V:{V.shape}")
        return U, V, s1, s2
