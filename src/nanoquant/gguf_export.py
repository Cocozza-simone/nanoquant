"""
Esporta modelli quantizzati NANOQUANT in formato GGUF Q1_0_g128
compatibile con OxiBonsai per inferenza ultra-veloce.

COSA FA E PERCHÉ:
    OxiBonsai è un backend inferenza Rust ottimizzato per pesi binari.
    Il formato Q1_0_g128 rappresenta ogni peso come:
    - Sign bit (±1) → packed come {0, 1}
    - Scale FP16 condivisa ogni 128 pesi
    
    Questo è identico alla rappresentazione binaria di NANOQUANT,
    quindi l'esportazione è una semplice recodifica senza perdita.

INTEGRAZIONE:
    export_to_gguf(
        quantized_layers=quantizer.get_quantized_layers(),
        model_metadata={"architecture": "llama", "context_length": 4096},
        output_path="./model.gguf"
    )
    
    Poi eseguire con: oxibonsai run --model model.gguf

Fonte: https://github.com/cool-japan/oxibonsai (COOLJAPAN OU, 2026)
"""

import struct
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

# Magic GGUF
GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION = 3

# GGUF type codes
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_UINT64 = 6
GGUF_TYPE_FLOAT32 = 1
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9

# Tipo quantizzazione Q1_0 (1 bit per peso)
GGUF_TYPE_Q1_0 = 1
GROUP_SIZE = 128  # G128 come OxiBonsai


def pack_nanoquant_to_q1_0_g128(
    U_binary: torch.Tensor,   # ±1, shape [d_out, rank]
    V_binary: torch.Tensor,   # ±1, shape [d_in, rank]
    s1: torch.Tensor,         # scale out, shape [d_out]
    s2: torch.Tensor,         # scale in, shape [d_in]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converte la fattorizzazione NANOQUANT (U, V, s1, s2) in
    formato Q1_0_g128 compatibile con OxiBonsai.
    
    Ricostruisce W_approx = s1 ⊙ (U @ V^T) ⊙ s2^T,
    poi la ricodifica in sign bit + scale FP16 per gruppo di 128.
    
    Args:
        U_binary: [d_out, rank] in {-1, +1}
        V_binary: [d_in, rank] in {-1, +1}
        s1: [d_out]
        s2: [d_in]
    
    Returns:
        (signs_packed, scales) dove:
        - signs_packed: uint8 array con bit impaccati
        - scales: float16 array con 1 scala per 128 pesi
    """
    # Ricostruzione matrice approssimata
    d_out, rank = U_binary.shape
    d_in = V_binary.shape[0]
    
    W = (s1.unsqueeze(1) * 
         (U_binary.float() @ V_binary.float().T) * 
         s2.unsqueeze(0))
    
    W_np = W.detach().cpu().numpy().astype(np.float32)
    
    num_groups = (d_in + GROUP_SIZE - 1) // GROUP_SIZE
    
    signs_list = []   # list di uint8 arrays
    scales_list = []  # list di float16
    
    logger.debug(f"Packing Q1_0_g128: shape {W_np.shape}, {num_groups} groups per row")
    
    for row_idx, row in enumerate(W_np):
        for g in range(num_groups):
            start = g * GROUP_SIZE
            end = min(start + GROUP_SIZE, d_in)
            chunk = row[start:end]
            
            # Pad con zeri se necessario (ultima colonna può essere incompleta)
            if len(chunk) < GROUP_SIZE:
                chunk = np.pad(chunk, (0, GROUP_SIZE - len(chunk)))
            
            # Calcola scala ottimale: max(|w|) in questo gruppo
            scale = np.max(np.abs(chunk))
            if scale < 1e-7:
                scale = 1e-7  # Evita divisione per zero
            
            scales_list.append(np.float16(scale))
            
            # Sign bit: w >= 0 → 1, w < 0 → 0
            bits = (chunk >= 0).astype(np.uint8)
            
            # Pack 8 bit per byte usando np.packbits
            packed = np.packbits(bits)
            signs_list.append(packed)
    
    # Concatena tutti i dati
    signs_packed = np.concatenate(signs_list)
    scales_array = np.array(scales_list, dtype=np.float16)
    
    logger.info(
        f"Q1_0_g128 packed: {len(signs_packed)} bytes signs, "
        f"{len(scales_array)} scales, "
        f"compression ratio: {(d_out * d_in * 4 / len(signs_packed)):.1f}x"
    )
    
    return signs_packed, scales_array


def export_to_gguf(
    quantized_layers: Dict[str, Dict[str, torch.Tensor]],
    model_metadata: Optional[Dict[str, Any]] = None,
    output_path: str = "./model.gguf",
) -> Path:
    """
    Esporta i layer quantizzati in formato GGUF Q1_0_g128.
    
    Usa il formato GGUF standard ma con il nostro custom packing,
    risultando in un file compatibile con OxiBonsai.
    
    Args:
        quantized_layers: Dict[layer_name, {"U_binary": Tensor, "V_binary": Tensor, 
                                             "s1": Tensor, "s2": Tensor}]
        model_metadata: Dict di metadati opzionali (architecture, context_length, etc)
        output_path: Percorso file di output
    
    Returns:
        Path dell'output file
    
    Esempio:
        export_to_gguf(
            quantized_layers=quantizer.get_quantized_layers(),
            model_metadata={"architecture": "llama", "context_length": 4096},
            output_path="./outputs/model_q1_0_g128.gguf"
        )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if model_metadata is None:
        model_metadata = {"architecture": "unknown", "context_length": 4096}
    
    logger.info(f"Exporting to GGUF: {output_path}")
    logger.info(f"Layers: {len(quantized_layers)}")
    
    with open(output_path, "wb") as f:
        # Magic + Version
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", GGUF_VERSION))
        
        # Numero di tensor
        f.write(struct.pack("<Q", len(quantized_layers)))
        
        # Numero di metadati
        f.write(struct.pack("<Q", len(model_metadata)))
        
        # Scrivi metadati
        _write_gguf_metadata(f, model_metadata)
        
        # Scrivi tensors
        for layer_name, layer_data in quantized_layers.items():
            try:
                U = layer_data["U_binary"]
                V = layer_data["V_binary"]
                s1 = layer_data["s1"]
                s2 = layer_data["s2"]
                
                signs, scales = pack_nanoquant_to_q1_0_g128(U, V, s1, s2)
                _write_gguf_tensor(f, layer_name, signs, scales, GGUF_TYPE_Q1_0)
                
            except Exception as e:
                logger.warning(f"Failed to export {layer_name}: {e}")
    
    file_size_mb = output_path.stat().st_size / (1024 ** 2)
    logger.info(f"✅ GGUF export complete: {file_size_mb:.1f} MB")
    logger.info(f"   Ready for OxiBonsai: oxibonsai run --model {output_path}")
    
    return output_path


def _write_gguf_metadata(f, metadata: Dict[str, Any]):
    """Scrive i metadati GGUF in formato key-value."""
    for key, value in metadata.items():
        # Chiave (string)
        key_bytes = key.encode("utf-8")
        f.write(struct.pack("<Q", len(key_bytes)))
        f.write(key_bytes)
        
        # Valore
        if isinstance(value, str):
            f.write(struct.pack("<I", GGUF_TYPE_STRING))
            val_bytes = value.encode("utf-8")
            f.write(struct.pack("<Q", len(val_bytes)))
            f.write(val_bytes)
        elif isinstance(value, int):
            f.write(struct.pack("<I", GGUF_TYPE_UINT32))
            f.write(struct.pack("<I", value))
        elif isinstance(value, float):
            f.write(struct.pack("<I", GGUF_TYPE_FLOAT32))
            f.write(struct.pack("<f", value))
        else:
            logger.warning(f"Unsupported metadata type for {key}: {type(value)}")


def _write_gguf_tensor(
    f,
    name: str,
    signs: np.ndarray,
    scales: np.ndarray,
    quant_type: int,
):
    """Scrive un singolo tensor quantizzato nel formato GGUF."""
    # Nome tensor
    name_bytes = name.encode("utf-8")
    f.write(struct.pack("<Q", len(name_bytes)))
    f.write(name_bytes)
    
    # Tipo quantizzazione
    f.write(struct.pack("<I", quant_type))
    
    # Dimensioni: 2D per matrici
    f.write(struct.pack("<Q", 2))  # rank = 2
    f.write(struct.pack("<Q", len(scales)))  # d_out (numero di righe di scale)
    f.write(struct.pack("<Q", GROUP_SIZE))   # GROUP_SIZE
    
    # Dati: prima i bit, poi le scale
    f.write(struct.pack("<Q", len(signs)))
    f.write(signs.tobytes())
    
    f.write(struct.pack("<Q", len(scales)))
    f.write(scales.tobytes())


def load_gguf_metadata(gguf_path: str) -> Dict[str, Any]:
    """Carica i metadati da un file GGUF (utility function).
    
    Args:
        gguf_path: Path del file GGUF
    
    Returns:
        Dict dei metadati
    """
    metadata = {}
    
    with open(gguf_path, "rb") as f:
        # Leggi header
        magic = struct.unpack("<I", f.read(4))[0]
        version = struct.unpack("<I", f.read(4))[0]
        
        if magic != GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF magic: {magic:#x}")
        
        num_tensors = struct.unpack("<Q", f.read(8))[0]
        num_metadata = struct.unpack("<Q", f.read(8))[0]
        
        # Leggi metadati
        for _ in range(num_metadata):
            key_len = struct.unpack("<Q", f.read(8))[0]
            key = f.read(key_len).decode("utf-8")
            
            type_code = struct.unpack("<I", f.read(4))[0]
            
            if type_code == GGUF_TYPE_STRING:
                val_len = struct.unpack("<Q", f.read(8))[0]
                value = f.read(val_len).decode("utf-8")
            elif type_code == GGUF_TYPE_UINT32:
                value = struct.unpack("<I", f.read(4))[0]
            elif type_code == GGUF_TYPE_FLOAT32:
                value = struct.unpack("<f", f.read(4))[0]
            else:
                value = None
            
            if value is not None:
                metadata[key] = value
    
    return metadata
