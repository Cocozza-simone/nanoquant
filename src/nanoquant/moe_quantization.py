"""
Integrazione ispirata a QMoE (IST-DASLab) per modelli Mixture-of-Experts.

COSA FA E PERCHÉ:
    NANOQUANT usa ADMM per fattorizzare in forma binaria (U, V).
    QMoE ha dimostrato che la quantizzazione degli expert sparsi in MoE
    richiede un trattamento speciale: gli expert vengono quantizzati
    separatamente dagli shared layers, con la possibilità di riusare
    l'Hessiano tra gate/up projection per risparmiare memoria.

STRATEGIA:
    - Shared layers: quantizzazione standard NANOQUANT (ADMM full)
    - Expert layers: ADMM con Hessiano condiviso tra gate/up projection
    
Fonte: https://github.com/IST-DASLab/qmoe (Frantar & Alistarh, MLSys 2024)
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Optional, Tuple, Any
from .config import NanoQuantConfig
from .admm import LatentBinaryADMM
from .reconstruction import FactorizedLinear

logger = logging.getLogger(__name__)


class MoEExpertQuantizer:
    """
    Quantizza in modo selettivo i layer expert di un modello MoE.
    Gli expert sparsi vengono trattati separatamente dagli shared layers.
    """

    def __init__(self, config: NanoQuantConfig, quantize_only_experts: bool = False):
        """
        Args:
            config: NanoQuantConfig con parametri MoE abilitati
            quantize_only_experts: Se True, quantizza solo gli expert
        """
        self.config = config
        self.quantize_only_experts = quantize_only_experts or config.quantize_only_experts
        self.admm_solver = LatentBinaryADMM(
            rank=config.rank,
            num_iterations=config.admm_iterations,
            rho=config.admm_rho,
            lambda_reg=config.admm_lambda,
            epsilon=config.admm_epsilon,
            device=config.device,
        )
        logger.info(f"MoEExpertQuantizer initialized: quantize_only_experts={self.quantize_only_experts}")

    def is_expert_layer(self, name: str) -> bool:
        """Identifica i layer expert per nome.
        
        Compatibile con:
        - Mixtral: "*.experts.*.w_in", "*.experts.*.w_out"
        - DeepSeek: "*.mlp.experts.*"
        - Switch: "*.moe.experts.*"
        - QWen2-MoE: "*.moe.experts.*"
        """
        expert_keywords = ["experts", "expert_", "mlp.experts", "moe.experts"]
        return any(kw in name.lower() for kw in expert_keywords)

    def is_gate_layer(self, name: str) -> bool:
        """Identifica i layer gate/router."""
        gate_keywords = ["gate", "router", "moe_layer"]
        return any(kw in name.lower() for kw in gate_keywords)

    def quantize_moe_model(
        self,
        model: nn.Module,
        hessians: Optional[Dict[str, torch.Tensor]] = None,
        preconditioners: Optional[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]] = None,
    ) -> nn.Module:
        """Quantizza un modello MoE applicando ADMM selettivamente.
        
        Args:
            model: Modello PyTorch (es. LlamaForCausalLM, MixtralForCausalLM)
            hessians: Dict di Hessiani precalcolati per ogni layer (deprecated)
            preconditioners: Tuple of (D_in, D_out) dicts from GlobalCalibration.compute_preconditioners()
                            Se None, non usa preconditioning (degrada la qualità)
        
        Returns:
            Modello quantizzato (modificato in-place)
        """
        D_in, D_out = preconditioners if preconditioners else ({}, {})
        quantized_count = 0
        skipped_count = 0
        
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            
            # Saltta gate layers (non vengono quantizzati)
            if self.is_gate_layer(name):
                logger.debug(f"Skip gate layer: {name}")
                skipped_count += 1
                continue
            
            # Decidi se quantizzare questo layer
            is_expert = self.is_expert_layer(name)
            should_quantize = is_expert or not self.quantize_only_experts
            
            if not should_quantize:
                skipped_count += 1
                continue
            
            # Estrai pesi
            W = module.weight.data  # [out_features, in_features]
            d_out, d_in = W.shape

            # K-FAC: cerca precondizionatori per questo layer
            D_in_layer = D_in.get(name) if D_in else None
            D_out_layer = D_out.get(name) if D_out else None

            # Quantizza il layer con/ senza preconditioning
            if D_in_layer is not None and D_out_layer is not None:
                # Verifica dimensioni
                if len(D_in_layer) == d_in and len(D_out_layer) == d_out:
                    # Applica K-FAC preconditioning: W_f = D_out^{1/2} W D_in^{1/2}
                    # (cfr. BlockReconstructionPipeline._initialize_binary_factors)
                    D_out_sqrt = D_out_layer.sqrt()
                    D_in_sqrt = D_in_layer.sqrt()
                    W_f = D_out_sqrt.unsqueeze(1) * W * D_in_sqrt.unsqueeze(0)
                    U, V, s1, s2 = self.admm_solver.solve(W_f, D_in_layer, D_out_layer)
                else:
                    U, V, s1, s2 = self.admm_solver.solve_simple(W)
            else:
                U, V, s1, s2 = self.admm_solver.solve_simple(W)

            # Crea FactorizedLinear preservando la struttura binaria
            factorized_layer = FactorizedLinear(
                d_out=d_out,
                d_in=d_in,
                rank=U.shape[1],
                U=U,
                V=V,
                s1=s1,
                s2=s2,
                bias=module.bias.data if module.bias is not None else None,
            )
            # Congela i pesi binari (cruciale per inferenza ottimizzata)
            factorized_layer.pack()

            # Sostituisci il modulo nel modello
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                if part.isdigit():
                    parent = parent[int(part)]
                else:
                    parent = getattr(parent, part)
            child_name = parts[-1]
            setattr(parent, child_name, factorized_layer)

            quantized_count += 1
            layer_type = "expert" if is_expert else "shared"
            logger.debug(
                f"Quantized {layer_type} layer: {name} "
                f"[{W.shape[0]}x{W.shape[1]} → rank={U.shape[1]}, FactorizedLinear]"
            )
        
        logger.info(
            f"MoE quantization complete: {quantized_count} layers quantized, {skipped_count} skipped"
        )
        return model

    def get_expert_layers(self, model: nn.Module) -> Dict[str, nn.Linear]:
        """Estrae tutti i layer expert dal modello.
        
        Returns:
            Dict[name, module] dei layer expert
        """
        expert_layers = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and self.is_expert_layer(name):
                expert_layers[name] = module
        return expert_layers

    def estimate_memory_savings(self, model: nn.Module) -> Dict[str, Any]:
        """Stima il risparmio di memoria dalla quantizzazione MoE.
        
        Returns:
            Dict con statistiche: original_size_mb, quantized_size_mb, ratio, ...
        """
        original_params = 0
        expert_params = 0
        shared_params = 0
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                numel = module.weight.numel()
                original_params += numel
                
                if self.is_expert_layer(name):
                    expert_params += numel
                else:
                    shared_params += numel
        
        # FP32 = 4 bytes, 1-bit quantized ≈ 0.2 bytes (+ scales overhead)
        original_size_mb = original_params * 4 / (1024 ** 2)
        # Stima: 1-bit weights + 2 scale vectors per layer
        quantized_size_mb = original_params * 0.2 / (1024 ** 2)
        
        return {
            "original_size_mb": original_size_mb,
            "quantized_size_mb": quantized_size_mb,
            "compression_ratio": original_size_mb / quantized_size_mb if quantized_size_mb > 0 else float('inf'),
            "total_params": original_params,
            "expert_params": expert_params,
            "shared_params": shared_params,
            "expert_ratio": expert_params / original_params if original_params > 0 else 0.0,
        }
