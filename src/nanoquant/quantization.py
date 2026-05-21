"""
Main NANOQUANT quantizer implementing the full pipeline.

Implements Algorithm 1 from the paper:
1. Global Calibration (Phase 1)
2. Block Reconstruction Pipeline (Phase 2)
3. Model Reconstruction (Phase 3)
"""

import torch
import torch.nn as nn
import logging
import copy
import gc
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from .config import NanoQuantConfig
from .calibration import GlobalCalibration
from .reconstruction import (
    BlockReconstructionPipeline,
    ModelReconstruction,
    FactorizedLinear,
)

logger = logging.getLogger(__name__)


def find_linear_layers(module: nn.Module, prefix: str = "") -> List[Tuple[str, nn.Linear]]:
    """Find all linear layers in a module.
    
    Args:
        module: Module to search
        prefix: Name prefix
        
    Returns:
        List of (name, layer) tuples
    """
    linear_layers = []
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            linear_layers.append((full_name, child))
        else:
            linear_layers.extend(find_linear_layers(child, full_name))
    return linear_layers


def get_transformer_blocks(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Get transformer blocks from the model.
    
    Args:
        model: HuggingFace model
        
    Returns:
        List of (block_name, block_module) tuples
    """
    blocks = []
    
    # Try common attribute names for transformer blocks
    for attr_name in ["model", "transformer", "gpt_neox", "gptj"]:
        if hasattr(model, attr_name):
            parent = getattr(model, attr_name)
            for layer_attr in ["layers", "h", "encoder", "decoder"]:
                if hasattr(parent, layer_attr):
                    layer_list = getattr(parent, layer_attr)
                    if isinstance(layer_list, (list, nn.ModuleList)):
                        for i, block in enumerate(layer_list):
                            blocks.append((f"{attr_name}.{layer_attr}.{i}", block))
                        return blocks
    
    # Fallback: search recursively for ModuleList containing transformer blocks
    for name, module in model.named_modules():
        if isinstance(module, (list, nn.ModuleList)):
            for i, block in enumerate(module):
                if isinstance(block, nn.Module) and len(list(block.children())) > 2:
                    blocks.append((f"{name}.{i}", block))
            if blocks:
                return blocks
    
    return blocks


class NanoQuantizer:
    """Main NANOQUANT quantizer.
    
    Implements the full quantization pipeline from Algorithm 1.
    """
    
    def __init__(self, config: NanoQuantConfig):
        self.config = config
        self.config.adapt_for_model_family(config.model_name)
        
        self.model: Optional[nn.Module] = None
        self.tokenizer = None
        self.quantized_model: Optional[nn.Module] = None
        self.D_in: Dict[str, torch.Tensor] = {}
        self.D_out: Dict[str, torch.Tensor] = {}
        
        logger.info(f"Initialized NANOQUANT with config: {config}")
    
    def load_model(self):
        """Load model and tokenizer."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            use_fast=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        logger.info(f"Model loaded. Parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.0f}M")
    
    def quantize(self) -> nn.Module:
        """Run full NANOQUANT quantization pipeline.
        
        Returns:
            Quantized model
        """
        if self.model is None:
            self.load_model()
        
        logger.info("=" * 60)
        logger.info("Starting NANOQUANT Quantization Pipeline")
        logger.info("=" * 60)
        
        # Phase 1: Global Calibration
        logger.info("\n--- Phase 1: Global Calibration ---")
        calibration = GlobalCalibration(self.model, self.tokenizer, self.config)
        self.D_in, self.D_out = calibration.compute_preconditioners()
        
        # Phase 2: Block Reconstruction Pipeline
        logger.info("\n--- Phase 2: Block Reconstruction Pipeline ---")
        self.quantized_model = self._block_reconstruction()
        
        # Phase 3: Model Reconstruction
        logger.info("\n--- Phase 3: Model Reconstruction ---")
        self._model_reconstruction()
        
        logger.info("\n" + "=" * 60)
        logger.info("NANOQUANT Quantization Complete!")
        logger.info("=" * 60)
        
        # Log compression stats
        self._log_compression_stats()
        
        return self.quantized_model
    
    def _block_reconstruction(self) -> nn.Module:
        """Phase 2: Block-wise reconstruction pipeline.
        
        Processes each transformer block sequentially.
        """
        # Create a copy of the model for quantization
        logger.info("Creating model copy for quantization...")
        
        # Move model to specified device
        device = self.config.device
        self.model.to(device)
        
        # Get transformer blocks
        blocks = get_transformer_blocks(self.model)
        
        if not blocks:
            logger.warning("No transformer blocks found! Using fallback approach.")
            blocks = self._find_blocks_fallback()
        
        logger.info(f"Found {len(blocks)} transformer blocks")
        
        # Find all linear layers
        all_linear_layers = find_linear_layers(self.model)
        logger.info(f"Found {len(all_linear_layers)} linear layers")
        
        # Initialize block reconstruction pipeline
        pipeline = BlockReconstructionPipeline(
            self.config,
            self.D_in,
            self.D_out,
        )
        
        # Get calibration inputs
        calib_input_ids, calib_attention_mask = self._get_calibration_inputs()
        calib_input_ids = calib_input_ids.to(device)
        calib_attention_mask = calib_attention_mask.to(device)
        
        # Process each block
        for block_idx, (block_name, block) in enumerate(blocks):
            logger.info(f"\nProcessing block {block_idx + 1}/{len(blocks)}: {block_name}")
            
            # Get inputs for this block
            X, Y_star = self._get_block_inputs_outputs(
                self.model, block_name, calib_input_ids, calib_attention_mask
            )
            
            # Find linear layers in this block
            block_linears = [
                (name, layer) for name, layer in all_linear_layers
                if name.startswith(block_name + ".")
            ]
            
            if not block_linears:
                logger.warning(f"  No linear layers found in {block_name}")
                continue
            
            logger.info(f"  Found {len(block_linears)} linear layers")
            
            # Reconstruct block
            try:
                factorized_layers = pipeline.reconstruct_block(
                    model=self.model,
                    block=block,
                    block_name=block_name,
                    X=X,
                    Y_star=Y_star,
                    linear_layers=block_linears,
                )
                
                # Replace layers in the model
                for (layer_name, _), fl in zip(block_linears, factorized_layers):
                    self._replace_layer(self.model, layer_name, fl)
                
                logger.info(f"  Block {block_idx + 1} reconstructed successfully")

            except Exception as e:
                logger.error(f"  Error reconstructing block {block_idx + 1}: {e}")
                logger.error("  Skipping this block, keeping original weights")
                continue

            finally:
                # Clean up intermediate tensors to avoid memory buildup
                del X, Y_star
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()

        return self.model
    
    def _model_reconstruction(self):
        """Phase 3: Global scale tuning."""
        # Get calibration inputs
        calib_input_ids, calib_attention_mask = self._get_calibration_inputs()
        calib_input_ids = calib_input_ids.to(self.config.device)
        calib_attention_mask = calib_attention_mask.to(self.config.device)

        reconstructor = ModelReconstruction(self.config)

        # Use subset of calibration data for global tuning
        n_samples = min(32, len(calib_input_ids))

        # Compute reference logits from current model without deep copy
        self.model.eval()
        with torch.no_grad():
            ref_outputs = self.model(
                input_ids=calib_input_ids[:n_samples],
                attention_mask=calib_attention_mask[:n_samples],
            )
            ref_logits = ref_outputs.logits.detach().clone()

        reconstructor.reconstruct(
            quantized_model=self.model,
            input_ids=calib_input_ids[:n_samples],
            attention_mask=calib_attention_mask[:n_samples],
            original_logits=ref_logits,
        )
    
    def _get_calibration_inputs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get calibration data.
        
        Returns:
            Tuple of (input_ids, attention_mask)
        """
        from datasets import load_dataset
        
        dataset = load_dataset(
            self.config.calib_dataset,
            self.config.calib_config,
            split="train",
            trust_remote_code=True,
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
    
    def _get_block_inputs_outputs(
        self,
        model: nn.Module,
        block_name: str,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get inputs and target outputs for a specific block.
        
        Args:
            model: Full model
            block_name: Name of the block
            input_ids: Input tokens
            attention_mask: Attention mask
            
        Returns:
            Tuple of (block_inputs, block_targets)
        """
        # Hook to capture block inputs and outputs
        block_inputs = {}
        block_outputs = {}
        
        def make_hook(name, storage, is_input=True):
            def hook(module, input, output):
                if is_input:
                    if isinstance(input, tuple) and input[0] is not None:
                        storage[name] = input[0].detach()
                else:
                    if isinstance(output, tuple):
                        storage[name] = output[0].detach()
                    else:
                        storage[name] = output.detach()
            return hook
        
        # Register hooks
        handles = []
        block_module = model.get_submodule(block_name)
        
        handle_in = block_module.register_forward_pre_hook(
            lambda m, i: block_inputs.update({block_name: i[0].detach() if isinstance(i, tuple) and i[0] is not None else i.detach() if not isinstance(i, tuple) else None})
        )
        
        def output_hook(m, i, o):
            if isinstance(o, tuple):
                block_outputs[block_name] = o[0].detach()
            else:
                block_outputs[block_name] = o.detach()
        
        handle_out = block_module.register_forward_hook(output_hook)
        handles.extend([handle_in, handle_out])
        
        try:
            # Forward pass
            with torch.no_grad():
                model(input_ids=input_ids[:8], attention_mask=attention_mask[:8])
        finally:
            # Remove hooks
            for h in handles:
                h.remove()
        
        X = block_inputs.get(block_name)
        Y_star = block_outputs.get(block_name)
        
        if X is None or Y_star is None:
            logger.warning(f"Could not capture block {block_name} I/O, using defaults")
            # Use model embedding output as default
            with torch.no_grad():
                outputs = model(input_ids=input_ids[:8], attention_mask=attention_mask[:8], output_hidden_states=True)
                X = outputs.hidden_states[0] if hasattr(outputs, "hidden_states") else outputs[0]
                Y_star = X.clone()
        
        return X, Y_star
    
    def _replace_layer(self, model: nn.Module, layer_name: str, new_layer: nn.Module):
        """Replace a layer in the model.
        
        Args:
            model: Model containing the layer
            layer_name: Dot-separated layer name
            new_layer: New layer to insert
        """
        parts = layer_name.split(".")
        parent = model
        
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)
        
        child_name = parts[-1]
        setattr(parent, child_name, new_layer)
    
    def _find_blocks_fallback(self) -> List[Tuple[str, nn.Module]]:
        """Fallback method to find transformer blocks."""
        blocks = []
        
        # Try to find sequential blocks
        for name, module in self.model.named_modules():
            if "layer" in name.lower() or "block" in name.lower():
                # Check if this looks like a transformer block
                children = list(module.named_children())
                if len(children) >= 3:  # Typically has attention + MLP + norms
                    blocks.append((name, module))
        
        # Remove duplicates (keep parent blocks)
        filtered = []
        for name, module in blocks:
            is_child = any(name != other_name and name.startswith(other_name + ".") for other_name, _ in blocks)
            if not is_child:
                filtered.append((name, module))
        
        return filtered
    
    def _log_compression_stats(self):
        """Log compression statistics."""
        if self.quantized_model is None:
            return
        
        total_params = 0
        binary_params = 0
        float_params = 0
        
        for name, module in self.quantized_model.named_modules():
            if isinstance(module, FactorizedLinear):
                binary_params += module.d_out * module.rank + module.d_in * module.rank
                float_params += module.d_out + module.d_in  # scales
            elif isinstance(module, nn.Linear):
                total_params += module.weight.numel()
                if module.bias is not None:
                    float_params += module.bias.numel()
        
        if binary_params > 0:
            total_quantized = binary_params + float_params
            original_bits = total_params * 16  # FP16
            quantized_bits = binary_params * 1 + float_params * 32  # 1-bit + float32
            
            compression_ratio = original_bits / quantized_bits if quantized_bits > 0 else 0
            effective_bits = quantized_bits / total_params if total_params > 0 else 0
            
            logger.info(f"\nCompression Statistics:")
            logger.info(f"  Original size: {original_bits / 8 / 1e9:.2f} GB (FP16)")
            logger.info(f"  Quantized size: {quantized_bits / 8 / 1e9:.2f} GB")
            logger.info(f"  Compression ratio: {compression_ratio:.2f}x")
            logger.info(f"  Effective bits/param: {effective_bits:.2f}")
    
    def save_quantized_model(self, path: str):
        """Save quantized model with full layer structure and metadata.
        
        Saves binary weights, scales, layer dimensions, and config
        for complete model reconstruction.
        
        Args:
            path: Output path
        """
        import os
        import json
        
        os.makedirs(path, exist_ok=True)
        
        # Build detailed state dict with layer metadata
        state_dict = {}
        layer_metadata = {}
        
        for name, module in self.quantized_model.named_modules():
            if isinstance(module, FactorizedLinear):
                prefix = name
                state_dict[f"{prefix}.U_binary"] = module.U_binary
                state_dict[f"{prefix}.V_binary"] = module.V_binary
                state_dict[f"{prefix}.s1"] = module.s1.data
                state_dict[f"{prefix}.s2"] = module.s2.data
                if module.bias is not None:
                    state_dict[f"{prefix}.bias"] = module.bias.data
                
                # Store layer metadata for reconstruction
                layer_metadata[name] = {
                    "d_out": module.d_out,
                    "d_in": module.d_in,
                    "rank": module.rank,
                    "has_bias": module.bias is not None,
                }
        
        # Build config dict with full metadata
        config_dict = {
            "model_name": self.config.model_name,
            "rank": self.config.rank,
            "bits": self.config.bits,
            "version": "0.1.0",
            "num_layers": len(layer_metadata),
            "layer_metadata": layer_metadata,
        }
        
        # Save config
        with open(os.path.join(path, "nanoquant_config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(path)
        
        # Save state dict
        torch.save(state_dict, os.path.join(path, "nanoquant_state.pt"))
        
        logger.info(f"Quantized model saved to {path}")
        logger.info(f"  - {len(layer_metadata)} factorized layers")
    
    @classmethod
    def load_quantized_model(cls, path: str, config: Optional[NanoQuantConfig] = None):
        """Load a previously quantized model with full reconstruction.
        
        Reconstructs all FactorizedLinear layers from saved binary weights
        and scales, then inserts them into the model architecture.
        
        Args:
            path: Path to saved model
            config: Optional config (will use saved config if not provided)
            
        Returns:
            NanoQuantizer instance with fully loaded model
        """
        import json
        import os
        
        # Load saved config
        config_path = os.path.join(path, "nanoquant_config.json")
        with open(config_path, "r") as f:
            saved_config = json.load(f)
        
        if config is None:
            config = NanoQuantConfig(
                model_name=saved_config["model_name"],
                rank=saved_config["rank"],
                bits=saved_config["bits"],
            )
        
        quantizer = cls(config)
        
        logger.info("Loading base model...")
        quantizer.load_model()
        
        # Load state dict
        state_path = os.path.join(path, "nanoquant_state.pt")
        state_dict = torch.load(state_path, map_location=config.device)
        
        layer_metadata = saved_config.get("layer_metadata", {})
        logger.info(f"Restoring {len(layer_metadata)} factorized layers...")
        
        # Reconstruct and replace each factorized layer
        for layer_name, metadata in layer_metadata.items():
            d_out = metadata["d_out"]
            d_in = metadata["d_in"]
            rank = metadata["rank"]
            has_bias = metadata["has_bias"]
            
            prefix = layer_name
            # Load saved tensors
            U_binary = state_dict[f"{prefix}.U_binary"]
            V_binary = state_dict[f"{prefix}.V_binary"]
            s1 = state_dict[f"{prefix}.s1"]
            s2 = state_dict[f"{prefix}.s2"]
            bias = state_dict.get(f"{prefix}.bias") if has_bias else None
            
            # Create FactorizedLinear layer
            factorized = FactorizedLinear(
                d_out=d_out,
                d_in=d_in,
                rank=rank,
                U=U_binary,
                V=V_binary,
                s1=s1,
                s2=s2,
                bias=bias,
            )
            # Set as packed (frozen binary)
            factorized.pack()
            factorized = factorized.to(config.device)
            
            # Replace the original layer in the model
            quantizer._replace_layer(quantizer.model, layer_name, factorized)
            logger.debug(f"  Restored: {layer_name}")
        
        quantizer.quantized_model = quantizer.model
        logger.info("Model loaded successfully!")
        
        return quantizer
    
    def _get_layer_rank_for_bits(self, layer_name: str, d_out: int, d_in: int) -> int:
        """Calculate effective rank for target bit-width.
        
        For sub-1-bit compression, we may need to adjust the rank
        to achieve the target effective bit rate.
        
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
        
        # For sub-1-bit: calculate rank needed for target bits
        # effective_bits = (2 * rank * (d_out + d_in) + (d_out + d_in) * 32) / (d_out * d_in * 16)
        # Solve for rank to achieve target bits
        # Simplified: rank ≈ (target_bits * d_out * d_in) / (2 * (d_out + d_in))
        
        rank_for_bits = int(target_bits * d_out * d_in / (2 * (d_out + d_in)))
        rank = max(1, min(rank_for_bits, self.config.rank))
        
        logger.debug(f"  {layer_name}: rank adjusted to {rank} for {target_bits:.2f} bits")
        return rank
