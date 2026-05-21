"""
Binary weight packing for efficient storage and inference.

Implements the packing scheme from Figure 2(c) of the paper:
- Maps {-1, +1} values to bits {0, 1}
- Packs into 8-bit integer blocks for memory efficiency
- Supports efficient unpacking for inference
"""

import torch
import numpy as np
import logging
from typing import Tuple, Optional
from .device_utils import get_optimal_device

logger = logging.getLogger(__name__)


def torch_packbits(bits: torch.Tensor) -> torch.Tensor:
    """Pack a uint8 tensor (values 0 or 1) into uint8 bytes using PyTorch-native ops.

    Uses big-endian bit ordering (MSB first) to match np.packbits output.

    Args:
        bits: uint8 tensor with values in {0, 1}, any shape

    Returns:
        Packed uint8 tensor
    """
    # Flatten tensor
    bits = bits.reshape(-1)
    num_elements = bits.numel()

    # Pad to multiple of 8
    if num_elements % 8 != 0:
        padding = 8 - (num_elements % 8)
        bits = torch.cat([bits, torch.zeros(padding, dtype=torch.uint8, device=bits.device)])

    # Reshape to [-1, 8]
    bits = bits.reshape(-1, 8)

    # Big-endian weights: MSB first (128, 64, 32, 16, 8, 4, 2, 1)
    weights = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=bits.device)

    # Hadamard product and sum to get packed bytes
    packed = (bits.to(torch.int16) * weights.to(torch.int16)).sum(dim=1).to(torch.uint8)

    return packed


def torch_unpackbits(packed: torch.Tensor, num_elements: int) -> torch.Tensor:
    """Unpack a uint8 tensor into bits using PyTorch-native ops.

    Uses big-endian bit ordering (MSB first) to match np.unpackbits input.

    Args:
        packed: uint8 tensor with packed bits
        num_elements: Number of elements to return

    Returns:
        uint8 tensor with values in {0, 1}
    """
    # Create shift positions (MSB first: 7 down to 0)
    shifts = torch.arange(7, -1, -1, dtype=torch.int64, device=packed.device)

    # Unpack each byte: ((packed >> shift) & 1) for each shift position
    bits = ((packed.unsqueeze(1).long() >> shifts) & 1).to(torch.uint8).reshape(-1)

    # Return only the requested number of elements
    return bits[:num_elements]


def pack_binary_tensor(tensor: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
    """Pack a binary tensor {-1, +1} into packed bits.

    Maps -1 -> 0, +1 -> 1 and packs into uint8 blocks.

    Args:
        tensor: Binary tensor with values in {-1, +1}, any shape

    Returns:
        Tuple of (packed_tensor, original_shape)
    """
    # Flatten tensor
    original_shape = tensor.shape
    flat = tensor.reshape(-1)

    # Map {-1, +1} -> {0, 1}
    bits = ((flat + 1) / 2).clamp(0, 1).to(torch.uint8)

    # Pack 8 bits per uint8 using native PyTorch ops
    try:
        packed_tensor = torch_packbits(bits)
    except Exception:
        # Fallback to NumPy path for compatibility
        num_elements = bits.numel()
        if num_elements % 8 != 0:
            padding = 8 - (num_elements % 8)
            bits = torch.cat([bits, torch.zeros(padding, dtype=torch.uint8, device=bits.device)])
        bits_cpu = bits.cpu().numpy()
        packed = np.packbits(bits_cpu, bitorder='big')
        packed_tensor = torch.from_numpy(packed).to(tensor.device)

    return packed_tensor, original_shape


def unpack_binary_tensor(packed: torch.Tensor, original_shape: Tuple[int, ...]) -> torch.Tensor:
    """Unpack bits back to binary tensor {-1, +1}.

    Args:
        packed: Packed uint8 tensor
        original_shape: Original shape of the binary tensor

    Returns:
        Unpacked binary tensor with values in {-1, +1}
    """
    num_elements = int(np.prod(original_shape))

    # Use native PyTorch path first
    try:
        bits = torch_unpackbits(packed, num_elements)
    except Exception:
        # Fallback to NumPy path for compatibility
        packed_cpu = packed.cpu().numpy()
        bits = torch.from_numpy(np.unpackbits(packed_cpu, bitorder='big')).to(packed.device)
        bits = bits[:num_elements]

    # Map {0, 1} -> {-1, +1}
    binary = bits.float() * 2 - 1

    # Reshape to original shape
    return binary.reshape(original_shape)


def pack_binary_matrix(matrix: torch.Tensor) -> dict:
    """Pack a binary matrix for efficient storage.
    
    Args:
        matrix: Binary matrix with values in {-1, +1}
        
    Returns:
        Dictionary with packed data and metadata
    """
    packed, shape = pack_binary_tensor(matrix)
    return {
        "packed": packed,
        "shape": shape,
        "num_elements": int(np.prod(shape)),
    }


def unpack_binary_matrix(packed_dict: dict, device: Optional[str] = None) -> torch.Tensor:
    """Unpack a binary matrix from packed storage.
    
    Args:
        packed_dict: Dictionary from pack_binary_matrix
        device: Target device (defaults to CPU)
        
    Returns:
        Binary matrix with values in {-1, +1}
    """
    if device is None:
        device = "cpu"
    tensor = unpack_binary_tensor(packed_dict["packed"].to(device), packed_dict["shape"])
    return tensor


class PackedBinaryStorage:
    """Efficient storage for packed binary weights.
    
    Packs multiple binary matrices into a single contiguous buffer
    for maximum memory efficiency.
    """
    
    def __init__(self):
        self.layers = {}
        self._total_bits = 0
        self._total_params = 0
    
    def add_layer(self, name: str, U_binary: torch.Tensor, V_binary: torch.Tensor):
        """Add a quantized layer's binary weights.
        
        Args:
            name: Layer name
            U_binary: Binary U matrix [d_out, rank] in {-1, +1}
            V_binary: Binary V matrix [d_in, rank] in {-1, +1}
        """
        packed_u = pack_binary_matrix(U_binary)
        packed_v = pack_binary_matrix(V_binary)
        
        self.layers[name] = {
            "U": packed_u,
            "V": packed_v,
        }
        
        self._total_bits += packed_u["num_elements"] + packed_v["num_elements"]
        self._total_params += U_binary.numel() + V_binary.numel()
    
    def get_layer(self, name: str, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve unpacked binary weights for a layer.
        
        Args:
            name: Layer name
            device: Target device (defaults to CPU)
            
        Returns:
            Tuple of (U_binary, V_binary) in {-1, +1}
        """
        if name not in self.layers:
            raise ValueError(f"Layer {name} not found in packed storage")
        
        layer = self.layers[name]
        U = unpack_binary_matrix(layer["U"], device)
        V = unpack_binary_matrix(layer["V"], device)
        return U, V
    
    def get_compression_stats(self) -> dict:
        """Get compression statistics.
        
        Returns:
            Dictionary with compression metrics
        """
        original_bits = self._total_params * 16  # FP16
        quantized_bits = self._total_bits  # 1 bit per param
        
        return {
            "original_bits": original_bits,
            "quantized_bits": quantized_bits,
            "compression_ratio": original_bits / quantized_bits if quantized_bits > 0 else 0,
            "effective_bits": quantized_bits / self._total_params if self._total_params > 0 else 0,
            "total_params": self._total_params,
            "space_savings": (1 - quantized_bits / original_bits) * 100 if original_bits > 0 else 0,
        }
    
    def state_dict(self) -> dict:
        """Get serializable state dict.
        
        Returns:
            Dictionary for saving/loading
        """
        state = {}
        for name, layer in self.layers.items():
            state[name] = {
                "U_packed": layer["U"]["packed"],
                "U_shape": layer["U"]["shape"],
                "V_packed": layer["V"]["packed"],
                "V_shape": layer["V"]["shape"],
            }
        return state
    
    @classmethod
    def from_state_dict(cls, state: dict, device: str = "auto"):
        """Load from state dict.
        
        Args:
            state: State dictionary
            device: Target device
            
        Returns:
            PackedBinaryStorage instance
        """
        storage = cls()
        for name, layer_state in state.items():
            storage.layers[name] = {
                "U": {
                    "packed": layer_state["U_packed"].to(device) if isinstance(layer_state["U_packed"], torch.Tensor) else torch.tensor(layer_state["U_packed"], device=device),
                    "shape": tuple(layer_state["U_shape"]),
                    "num_elements": int(np.prod(layer_state["U_shape"])),
                },
                "V": {
                    "packed": layer_state["V_packed"].to(device) if isinstance(layer_state["V_packed"], torch.Tensor) else torch.tensor(layer_state["V_packed"], device=device),
                    "shape": tuple(layer_state["V_shape"]),
                    "num_elements": int(np.prod(layer_state["V_shape"])),
                },
            }
        return storage
