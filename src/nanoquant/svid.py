"""
Sign-Value Independent Decomposition (SVID).

Implements SVID as described in the NANOQUANT paper for deriving
the optimal rank-1 approximation that preserves the sign structure
during ADMM proxy updates.

References:
    - Pouransari et al., "Least Squares Binary Quantization of Neural Networks", 2020
    - Xu et al., "OneBit: Towards Extremely Low-Bit Large Language Models", 2024
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


def svid_decompose(P: torch.Tensor, rank: int = 1) -> torch.Tensor:
    """Sign-Value Independent Decomposition (SVID).
    
    Decomposes a matrix P into sign and magnitude components,
    preserving the sign structure while extracting the optimal
    rank-R approximation.
    
    For rank-1 SVID (used in NANOQUANT):
        Z = sign(P) * |sigma_1|  (best rank-1 sign-preserving approximation)
    
    For general rank-R SVID:
        Z = SVD_reconstruct(sign(P), rank)
    
    Args:
        P: Consensus matrix [m, n]
        rank: Target rank for approximation (default 1 for NANOQUANT)
        
    Returns:
        Z: SVID approximation [m, n]
    """
    if rank == 1:
        # Rank-1 SVID: optimal sign-preserving rank-1 approximation
        # For rank-1, we find the best approximation Z = u * v^T
        # that preserves the sign structure of P
        
        # Method: use power iteration on sign(P)
        # This finds the dominant singular vectors of the signed matrix
        signed_P = torch.sign(P)
        
        # Handle zeros in sign
        signed_P = torch.where(signed_P == 0, 
                               torch.ones_like(signed_P), 
                               signed_P)
        
        # Fast rank-1 approximation via power iteration
        m, n = signed_P.shape
        u = torch.randn(m, 1, device=P.device, dtype=P.dtype)
        u = u / torch.norm(u)
        
        for _ in range(10):
            v = signed_P.T @ u
            v = v / torch.norm(v)
            u = signed_P @ v
            u = u / torch.norm(u)
        
        # Scale by dominant singular value
        sigma = (u.T @ signed_P @ v).squeeze()
        Z = sigma * u @ v.T
        
    else:
        # General rank-R SVID via truncated SVD
        signed_P = torch.sign(P)
        signed_P = torch.where(signed_P == 0,
                               torch.ones_like(signed_P),
                               signed_P)
        
        U, S, Vh = torch.linalg.svd(signed_P, full_matrices=False)
        # Truncate to rank-R
        U_r = U[:, :rank]
        S_r = S[:rank]
        Vh_r = Vh[:rank, :]
        Z = U_r @ torch.diag(S_r) @ Vh_r
    
    return Z


def svid_rank1_fast(P: torch.Tensor) -> torch.Tensor:
    """Fast rank-1 SVID using direct computation.
    
    For the special case of rank-1, computes the SVID approximation
    efficiently without full SVD.
    
    Args:
        P: Consensus matrix [m, n]
        
    Returns:
        Z: Rank-1 SVID approximation [m, n]
    """
    signed_P = torch.sign(P)
    signed_P = torch.where(signed_P == 0,
                           torch.ones_like(signed_P),
                           signed_P)
    
    # Direct rank-1 approximation: use column/row sums
    col_sums = signed_P.sum(dim=0, keepdim=True)  # [1, n]
    row_sums = signed_P.sum(dim=1, keepdim=True)  # [m, 1]
    
    # Normalize
    col_sums = col_sums / torch.norm(col_sums)
    row_sums = row_sums / torch.norm(row_sums)
    
    # Reconstruct
    sigma = (row_sums.T @ signed_P @ col_sums.T).squeeze()
    Z = sigma * row_sums @ col_sums
    
    return Z


def project_to_binary_low_rank(P: torch.Tensor, rank: int = 1) -> torch.Tensor:
    """Project matrix to binary low-rank set via SVID.
    
    This is the projection operation used in the ADMM proxy update:
        Z^(k+1) = SVID(P^(k+1))
    
    where P^(k+1) = U^(k+1) + Lambda^(k) is the consensus variable.
    
    Args:
        P: Consensus variable [m, n]
        rank: Target rank
        
    Returns:
        Z: Projected binary low-rank matrix [m, n]
    """
    return svid_decompose(P, rank=rank)
