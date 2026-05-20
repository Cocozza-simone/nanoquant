"""
Group Scaling per 128 pesi (Q1_0_g128, ispirato a OxiBonsai, COOLJAPAN).

COSA FA E PERCHÉ:
    NANOQUANT usa scale per riga/colonna (s1 shape=[d_out], s2 shape=[d_in]).
    OxiBonsai usa 1 scala FP16 ogni 128 pesi nello stesso vettore packed.
    La granularità più fine di OxiBonsai cattura variazioni locali nei pesi
    che le scale globali per riga non possono rappresentare, riducendo
    l'errore di quantizzazione specialmente per modelli con distribuzioni
    di pesi non uniformi (es. Llama, Mistral con outlier localizzati).

STRATEGIA:
    Post-processing dopo la fattorizzazione ADMM:
    1. Ricostruisci W_approx = s1 ⊙ (U @ V^T) ⊙ s2^T
    2. Dividi W_approx in gruppi da GROUP_SIZE colonne
    3. Per ogni gruppo, calcola una scala locale più precisa
    4. Rimpiazza s2 (per colonna) con group_scales (per 128 pesi)

    Questo NON cambia la struttura U, V — si limita a raffinare
    le scale come post-processing a costo quasi zero.

INTEGRAZIONE:
    Chiamare apply_group_scaling() dopo LatentBinaryADMM.solve()
    in reconstruction.py (nella BlockReconstructionPipeline).

Fonte: https://github.com/cool-japan/oxibonsai (COOLJAPAN OU, 2026)
"""

import torch
import logging
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Dimensione gruppo come in Q1_0_g128 di OxiBonsai
GROUP_SIZE = 128


@dataclass
class GroupScaledWeights:
    """Contenitore per pesi binarizzati con group scales."""
    U_binary: torch.Tensor      # [d_out, rank] in {-1, +1}
    V_binary: torch.Tensor      # [d_in, rank] in {-1, +1}
    s1: torch.Tensor            # [d_out] scale per riga
    group_scales: torch.Tensor  # [d_out, num_groups] scale di gruppo
    group_size: int = GROUP_SIZE
    d_out: int = 0
    d_in: int = 0
    rank: int = 0


def compute_group_scales(
    W_approx: torch.Tensor,
    W_original: torch.Tensor,
    group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    """Calcola le scale di gruppo minimizzando l'errore di quantizzazione locale.

    Per ogni gruppo di `group_size` colonne, trova la scala ottimale che
    minimizza ||W_group_original - scale * sign(W_group_approx)||_F^2.

    La soluzione in forma chiusa è:
        scale* = <W_original_group, sign(W_approx_group)> / group_size

    Args:
        W_approx:    Ricostruzione ADMM W ≈ s1 ⊙ (U @ V^T) ⊙ s2^T [d_out, d_in]
        W_original:  Pesi FP originali [d_out, d_in]
        group_size:  Numero di pesi per gruppo (default 128)

    Returns:
        group_scales: Scale ottimali [d_out, num_groups] in float32
    """
    d_out, d_in = W_approx.shape
    num_groups = (d_in + group_size - 1) // group_size

    group_scales = torch.zeros(d_out, num_groups, dtype=torch.float32, device=W_approx.device)

    for g in range(num_groups):
        start = g * group_size
        end = min(start + group_size, d_in)
        
        W_approx_group = W_approx[:, start:end]  # [d_out, group_size o meno]
        W_orig_group = W_original[:, start:end]
        
        # Soluzione ottimale: <W_orig, sign(W_approx)> / group_size
        signs = torch.sign(W_approx_group)
        numerator = (W_orig_group * signs).sum(dim=1)  # [d_out]
        group_scales[:, g] = numerator / max(W_approx_group.shape[1], 1)

    logger.debug(
        f"Group scales: shape={group_scales.shape}, "
        f"mean={group_scales.mean().item():.4f}, "
        f"std={group_scales.std().item():.4f}"
    )
    return group_scales.to(torch.float32)


def apply_group_scaling(
    U: torch.Tensor,
    V: torch.Tensor,
    s1: torch.Tensor,
    s2: torch.Tensor,
    W_original: torch.Tensor,
    group_size: int = GROUP_SIZE,
) -> GroupScaledWeights:
    """Applica il group scaling Q1_0_g128 ai risultati ADMM.

    Ricostruisce W_approx dalla fattorizzazione ADMM, poi calcola scale
    di gruppo più fini che minimizzano l'errore di quantizzazione locale.
    Restituisce un oggetto GroupScaledWeights con U/V binarizzati + group scales.

    Args:
        U:           Latent U da ADMM [d_out, rank]
        V:           Latent V da ADMM [d_in, rank]
        s1:          Scale output da ADMM [d_out]
        s2:          Scale input da ADMM [d_in]
        W_original:  Pesi FP32 originali [d_out, d_in]
        group_size:  Pesi per gruppo (default 128 come OxiBonsai)

    Returns:
        GroupScaledWeights con U/V binarizzati e group scales ottimali
    """
    d_out, rank = U.shape
    d_in = V.shape[0]

    # Binarizza U e V: {-1, +1}
    U_binary = torch.sign(U)
    U_binary = torch.where(U_binary == 0, torch.ones_like(U_binary), U_binary)

    V_binary = torch.sign(V)
    V_binary = torch.where(V_binary == 0, torch.ones_like(V_binary), V_binary)

    # Ricostruisci W_approx con scale ADMM originali
    # W ≈ s1 ⊙ (U_bin @ V_bin^T) ⊙ s2^T
    UV = U_binary @ V_binary.T                      # [d_out, d_in]
    W_approx = s1.unsqueeze(1) * UV * s2.unsqueeze(0)

    # Calcola group scales più fini
    group_scales = compute_group_scales(W_approx, W_original, group_size=group_size)

    # Calcola guadagno di errore
    error_before = _reconstruction_error(W_approx, W_original)
    W_group_reconstructed = reconstruct_from_group_scales(U_binary, V_binary, s1, group_scales, group_size)
    error_after = _reconstruction_error(W_group_reconstructed, W_original)

    logger.info(
        f"Group scaling: errore relativo {error_before:.4f} → {error_after:.4f} "
        f"({(error_before - error_after) / error_before * 100:.1f}% miglioramento)"
    )

    return GroupScaledWeights(
        U_binary=U_binary,
        V_binary=V_binary,
        s1=s1,
        group_scales=group_scales,
        group_size=group_size,
        d_out=d_out,
        d_in=d_in,
        rank=rank,
    )


def reconstruct_from_group_scales(
    U_binary: torch.Tensor,
    V_binary: torch.Tensor,
    s1: torch.Tensor,
    group_scales: torch.Tensor,  # [d_out, num_groups]
    group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    """Ricostruisce W da U, V binari e group scales.

    W[:, g*gs:(g+1)*gs] = s1 ⊙ (U @ V[:, g*gs:(g+1)*gs]^T) ⊙ group_scales[:, g]

    Args:
        U_binary:     [d_out, rank] in {-1, +1}
        V_binary:     [d_in, rank] in {-1, +1}
        s1:           [d_out]
        group_scales: [d_out, num_groups] in float16
        group_size:   pesi per gruppo

    Returns:
        W_reconstructed: [d_out, d_in]
    """
    d_out = U_binary.shape[0]
    d_in = V_binary.shape[0]
    num_groups = group_scales.shape[1]

    W_out = torch.zeros(d_out, d_in, dtype=torch.float32, device=U_binary.device)
    gs32 = group_scales.to(torch.float32)

    for g in range(num_groups):
        start = g * group_size
        end = min(start + group_size, d_in)
        
        # Ricostruzione per questo gruppo
        V_g = V_binary[start:end, :]  # [group_size, rank]
        UV_g = U_binary @ V_g.T       # [d_out, group_size]
        W_out[:, start:end] = s1.unsqueeze(1) * UV_g * gs32[:, g].unsqueeze(1)

    return W_out


def pack_with_group_scales(gsw: GroupScaledWeights) -> Dict:
    """Serializza GroupScaledWeights in un dizionario salvabile.

    Formato di storage (compatibile con OxiBonsai Q1_0_g128):
        - U/V: packed bit (1 bit per peso)
        - s1: FP32 per riga
        - group_scales: FP16 (1 scala ogni GROUP_SIZE pesi per riga)

    Returns:
        Dizionario con tutti i tensori serializzabili
    """
    from .packing import pack_binary_matrix  # import locale per evitare circular

    return {
        "U_packed": pack_binary_matrix(gsw.U_binary),
        "V_packed": pack_binary_matrix(gsw.V_binary),
        "s1": gsw.s1.to(torch.float32),
        "group_scales": gsw.group_scales.to(torch.float16),
        "group_size": gsw.group_size,
        "d_out": gsw.d_out,
        "d_in": gsw.d_in,
        "rank": gsw.rank,
    }


def memory_stats(gsw: GroupScaledWeights) -> Dict:
    """Calcola le statistiche di memoria con e senza group scaling.

    Returns:
        Dizionario con bytes usati e compression ratio
    """
    d_out, rank = gsw.U_binary.shape
    d_in = gsw.d_in
    num_groups = (d_in + gsw.group_size - 1) // gsw.group_size

    # Originale FP16
    bytes_fp16 = d_out * d_in * 2

    # NANOQUANT originale (senza group scaling)
    bits_U = d_out * rank                 # 1 bit per elemento
    bits_V = d_in * rank                  # 1 bit per elemento
    bytes_s1 = d_out * 4                  # FP32
    bytes_s2 = d_in * 4                   # FP32
    bytes_nanoquant = (bits_U + bits_V) // 8 + bytes_s1 + bytes_s2

    # Con group scaling (s2 → group_scales FP16)
    bytes_group_scales = d_out * num_groups * 2  # FP16
    bytes_with_groups = (bits_U + bits_V) // 8 + bytes_s1 + bytes_group_scales

    return {
        "original_bytes": bytes_fp16,
        "nanoquant_bytes": bytes_nanoquant,
        "with_group_scales_bytes": bytes_with_groups,
        "nanoquant_ratio": bytes_fp16 / bytes_nanoquant if bytes_nanoquant > 0 else float('inf'),
        "group_scales_ratio": bytes_fp16 / bytes_with_groups if bytes_with_groups > 0 else float('inf'),
        "group_scales_overhead": bytes_group_scales - bytes_s2,  # Differenza vs s2 standard
    }


def _reconstruction_error(W_approx: torch.Tensor, W_original: torch.Tensor) -> float:
    """Calcola l'errore relativo di ricostruzione."""
    error = torch.norm(W_original - W_approx, p="fro") / (torch.norm(W_original, p="fro") + 1e-8)
    return error.item()
