"""
Ternary Initialization for ADMM (ispirato a QMoE, IST-DASLab).

COSA FA E PERCHÉ:
    NANOQUANT inizializza U e V con rumore gaussiano casuale prima di ADMM.
    QMoE ha dimostrato che la quantizzazione ternaria {-max, 0, +max} crea
    ~90% di sparsità nei pesi dei MoE, ma il principio si applica anche a
    modelli densi: forzare a zero i pesi piccoli (rumore) prima della
    fattorizzazione dà ad ADMM un punto di partenza molto migliore,
    riducendo le iterazioni necessarie e migliorando la qualità finale.

STRATEGIA:
    1. Projetta W in forma ternaria: mantieni solo i top-k% per magnitudine
    2. Fattorizza la matrice sparse risultante via SVD troncata
    3. Usa i vettori singolari come init di U, V invece del rumore gaussiano

INTEGRAZIONE:
    Modificare LatentBinaryADMM.solve() in admm.py per usare
    ternary_svd_init() invece di torch.randn come inizializzazione.

Fonte: https://github.com/IST-DASLab/qmoe (Frantar & Alistarh, MLSys 2024)
"""

import torch
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def ternary_project(W: torch.Tensor, sparsity: float = 0.9) -> torch.Tensor:
    """Proietta W in forma ternaria: {-max_row, 0, +max_row}.

    Per ogni riga, mantieni solo i valori il cui |w| supera la soglia
    definita dal percentile (1 - sparsity). Il resto diventa zero.
    Questo è esattamente il meccanismo di QMoE che produce ~90% sparsità.

    Args:
        W:         Matrice dei pesi [d_out, d_in]
        sparsity:  Frazione di pesi da azzerare (default 0.9 = 90%)

    Returns:
        W_ternary: Matrice ternaria {-max, 0, +max} per riga [d_out, d_in]
    """
    # Soglia per riga: percentile (sparsity * 100) dei valori assoluti
    threshold = torch.quantile(W.abs(), sparsity, dim=1, keepdim=True)  # [d_out, 1]

    # Maschera booleana dei valori "significativi"
    mask = W.abs() >= threshold  # [d_out, d_in]

    # Per ogni riga, il valore ternario è il max assoluto della riga
    row_max = W.abs().max(dim=1, keepdim=True).values  # [d_out, 1]
    row_max = row_max.clamp(min=1e-8)

    # Applica: sign(w) * row_max dove mask=True, 0 altrove
    W_ternary = torch.sign(W) * row_max * mask.float()

    sparsity_actual = (W_ternary == 0).float().mean().item()
    logger.debug(f"Ternary projection: sparsity={sparsity_actual:.1%} (target={sparsity:.1%})")

    return W_ternary


def ternary_svd_init(
    W: torch.Tensor,
    rank: int,
    sparsity: float = 0.9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inizializzazione di U, V via SVD sulla proiezione ternaria di W.

    Invece di partire da rumore gaussiano (come fa ADMM di default),
    parte dai vettori singolari dominanti di W ternaria. Questo dà ad
    ADMM un punto di partenza vicino alla soluzione ottimale, riducendo
    le iterazioni e migliorando la qualità della fattorizzazione finale.

    Pipeline:
        W  →  W_ternary (ternary_project)
           →  SVD troncata (top-rank vettori singolari)
           →  U_init [d_out, rank],  V_init [d_in, rank]

    Args:
        W:        Matrice dei pesi [d_out, d_in]
        rank:     Rango target per la fattorizzazione
        sparsity: Sparsità ternaria (0.0 = nessuna, 0.9 = 90% zero)

    Returns:
        U_init: Matrice iniziale U [d_out, rank]
        V_init: Matrice iniziale V [d_in, rank]
    """
    device = W.device
    dtype = W.dtype

    # Step 1: proiezione ternaria
    if sparsity > 0.0:
        W_proj = ternary_project(W, sparsity=sparsity)
    else:
        W_proj = W  # nessuna sparsificazione, usa W direttamente

    # Step 2: SVD troncata sulla matrice sparse
    try:
        # Usa SVD economica: per matrici grandi molto più veloce della full SVD
        U_svd, S_svd, Vh_svd = torch.linalg.svd(W_proj, full_matrices=False)

        r = min(rank, U_svd.shape[1])  # non superare il rango disponibile

        # Prendi i top-r vettori singolari
        U_init = U_svd[:, :r] * S_svd[:r].sqrt().unsqueeze(0)  # [d_out, r]
        V_init = Vh_svd[:r, :].T * S_svd[:r].sqrt().unsqueeze(0)  # [d_in, r]

        # Se r < rank richiesto, padda con rumore piccolo
        if r < rank:
            noise_scale = S_svd[r - 1].item() * 0.01 if r > 0 else 0.01
            U_pad = torch.randn(U_init.shape[0], rank - r, device=device, dtype=dtype) * noise_scale
            V_pad = torch.randn(V_init.shape[0], rank - r, device=device, dtype=dtype) * noise_scale
            U_init = torch.cat([U_init, U_pad], dim=1)
            V_init = torch.cat([V_init, V_pad], dim=1)

        logger.debug(
            f"Ternary SVD init: W{list(W.shape)} → rank={r}, "
            f"S[0]={S_svd[0].item():.4f}, S[{r-1}]={S_svd[r-1].item():.4f}"
        )
        return U_init, V_init

    except Exception as e:
        logger.warning(f"Ternary SVD init fallback a gaussian: {e}")
        # Fallback sicuro: rumore gaussiano (comportamento originale)
        U_init = torch.randn(W.shape[0], rank, device=device, dtype=dtype) * 0.01
        V_init = torch.randn(W.shape[1], rank, device=device, dtype=dtype) * 0.01
        return U_init, V_init


def estimate_init_quality(
    W: torch.Tensor,
    U_init: torch.Tensor,
    V_init: torch.Tensor,
) -> float:
    """Calcola l'errore relativo dell'inizializzazione rispetto a W.

    Utile per confrontare la qualità di ternary_svd_init vs gaussian init.

    Returns:
        Errore relativo ||W - U @ V^T||_F / ||W||_F  (più basso = meglio)
    """
    W_approx = U_init @ V_init.T
    error = torch.norm(W - W_approx, p="fro") / (torch.norm(W, p="fro") + 1e-8)
    return error.item()
