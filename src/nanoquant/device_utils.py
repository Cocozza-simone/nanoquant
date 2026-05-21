"""Device detection and optimization utilities for cross-platform support."""

import torch
import platform


def get_optimal_device(device: str = "auto") -> str:
    """Auto-detect optimal device for the current hardware.
    
    Priority:
    1. If device != "auto", return as-is
    2. CUDA (NVIDIA GPU)
    3. MPS (Apple Silicon GPU)
    4. CPU (fallback)
    
    Args:
        device: Device preference ("auto", "cuda", "cpu", "mps")
        
    Returns:
        Optimal device string: "cuda", "mps", or "cpu"
    """
    if device != "auto":
        # User specified a device, validate it
        if device == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif device == "mps" and torch.backends.mps.is_available():
            return "mps"
        elif device == "cpu":
            return "cpu"
        else:
            # Fallback if specified device not available
            return get_optimal_device("auto")
    
    # Auto-detect mode
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_device_info() -> dict:
    """Get information about available devices.
    
    Returns:
        Dictionary with device availability and capability info
    """
    return {
        "platform": platform.system(),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "optimal_device": get_optimal_device("auto"),
        "pytorch_version": torch.__version__,
    }


def move_to_device(tensor: torch.Tensor, device: str) -> torch.Tensor:
    """Safely move tensor to device with fallback.
    
    Args:
        tensor: Tensor to move
        device: Target device
        
    Returns:
        Tensor on the specified device, or CPU if device unavailable
    """
    optimal = get_optimal_device(device)
    return tensor.to(optimal)
