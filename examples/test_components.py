#!/usr/bin/env python3
"""
Component test example - demonstrates NANOQUANT without full LLM.

This example shows how each component works with synthetic data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from nanoquant.admm import LatentBinaryADMM
from nanoquant.reconstruction import FactorizedLinear, straight_through_sign


def demo_ste():
    """Demonstrate Straight-Through Estimator."""
    print("\n" + "=" * 50)
    print("1. Straight-Through Estimator (STE)")
    print("=" * 50)
    
    x = torch.tensor([1.5, -0.5, 0.1, -2.0], requires_grad=True)
    y = straight_through_sign(x)
    
    print(f"Input:  {x.data.numpy()}")
    print(f"Output: {y.data.numpy()}")
    print(f"Expected: [1, -1, 1, -1]")
    
    # Backward
    loss = y.sum()
    loss.backward()
    print(f"Gradient: {x.grad.numpy()}")
    print("(Gradient flows through unchanged)")


def demo_admm():
    """Demonstrate ADMM factorization."""
    print("\n" + "=" * 50)
    print("2. Latent Binary ADMM Factorization")
    print("=" * 50)
    
    # Create a synthetic low-rank weight matrix
    d_out, d_in, rank = 64, 32, 4
    
    print(f"Target matrix: {d_out} x {d_in}")
    print(f"Target rank: {rank}")
    
    # Create near-binary low-rank matrix
    U_true = torch.sign(torch.randn(d_out, rank))
    V_true = torch.sign(torch.randn(d_in, rank))
    s1 = torch.ones(d_out) * 0.5
    s2 = torch.ones(d_in) * 0.5
    
    W = s1.unsqueeze(1) * (U_true @ V_true.T) * s2.unsqueeze(0)
    
    print(f"Original matrix norm: {torch.norm(W).item():.4f}")
    
    # Run ADMM
    admm = LatentBinaryADMM(
        rank=rank,
        num_iterations=100,
        rho=1.0,
        lambda_reg=0.01,
        epsilon=1e-5,
        device="cpu",
    )
    
    U, V, s1_est, s2_est = admm.solve_simple(W)
    
    # Reconstruct
    W_recon = s1_est.unsqueeze(1) * (torch.sign(U) @ torch.sign(V.T)) * s2_est.unsqueeze(0)
    
    rel_error = torch.norm(W - W_recon) / torch.norm(W)
    
    print(f"\nADMM Results:")
    print(f"  U shape: {U.shape}")
    print(f"  V shape: {V.shape}")
    print(f"  s1 shape: {s1_est.shape}, mean: {s1_est.mean().item():.4f}")
    print(f"  s2 shape: {s2_est.shape}, mean: {s2_est.mean().item():.4f}")
    print(f"\nReconstruction relative error: {rel_error.item():.4f}")
    
    # Compression stats
    original_bits = d_out * d_in * 16  # FP16
    compressed_bits = (d_out * rank + d_in * rank) * 1 + (d_out + d_in) * 32
    compression = original_bits / compressed_bits
    
    print(f"\nCompression:")
    print(f"  Original: {original_bits / 8:.0f} bytes (FP16)")
    print(f"  Compressed: {compressed_bits / 8:.0f} bytes")
    print(f"  Ratio: {compression:.2f}x")


def demo_factorized_linear():
    """Demonstrate FactorizedLinear layer."""
    print("\n" + "=" * 50)
    print("3. Factorized Linear Layer")
    print("=" * 50)
    
    d_out, d_in, rank = 256, 128, 8
    batch, seq_len = 2, 10
    
    # Create layer
    U = torch.randn(d_out, rank)
    V = torch.randn(d_in, rank)
    s1 = torch.ones(d_out)
    s2 = torch.ones(d_in)
    
    layer = FactorizedLinear(d_out, d_in, rank, U, V, s1, s2)
    
    # Forward pass
    x = torch.randn(batch, seq_len, d_in)
    y = layer(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Expected:     ({batch}, {seq_len}, {d_out})")
    
    # Verify against standard linear
    W = layer.get_binary_weight()
    y_expected = F.linear(x, W)
    
    error = torch.norm(y - y_expected).item()
    print(f"\nConsistency with weight matrix: error={error:.6f}")
    
    # Compression
    compressed_bits = layer.get_packed_size()
    original_bits = d_out * d_in * 16
    compression = original_bits / compressed_bits
    
    print(f"\nCompression ratio: {compression:.2f}x")
    print(f"Effective bits: {compressed_bits / (d_out * d_in):.2f}")


def demo_full_layer_quantization():
    """Demonstrate quantizing a single linear layer."""
    print("\n" + "=" * 50)
    print("4. Single Layer Quantization Demo")
    print("=" * 50)
    
    d_out, d_in = 512, 256
    rank = 8
    
    # Create random linear layer
    linear = nn.Linear(d_in, d_out, bias=True)
    W = linear.weight.data
    
    print(f"Original weight: {W.shape}")
    print(f"Original memory: {W.numel() * 2 / 1024:.2f} KB (FP16)")
    
    # Quantize with ADMM
    admm = LatentBinaryADMM(rank=rank, num_iterations=100, device="cpu")
    U, V, s1, s2 = admm.solve_simple(W)
    
    # Measure error
    W_quant = s1.unsqueeze(1) * (torch.sign(U) @ torch.sign(V.T)) * s2.unsqueeze(0)
    
    rel_error = torch.norm(W - W_quant) / torch.norm(W)
    snr = 20 * torch.log10(torch.norm(W) / torch.norm(W - W_quant))
    
    print(f"\nQuantization Results:")
    print(f"  Rank: {rank}")
    print(f"  Relative error: {rel_error.item():.4f}")
    print(f"  SNR: {snr.item():.2f} dB")
    
    # Memory
    binary_params = d_out * rank + d_in * rank
    scale_params = d_out + d_in
    total_binary_bits = binary_params  # 1 bit each
    total_scale_bits = scale_params * 32  # float32
    total_bits = total_binary_bits + total_scale_bits
    original_bits = W.numel() * 16
    
    print(f"\nMemory:")
    print(f"  Original: {original_bits / 8 / 1024:.2f} KB")
    print(f"  Quantized: {total_bits / 8 / 1024:.4f} KB")
    print(f"  Compression: {original_bits / total_bits:.2f}x")
    print(f"  Effective bits/weight: {total_bits / W.numel():.2f}")


def main():
    """Run all demos."""
    print("=" * 60)
    print("NANOQUANT Component Demonstration")
    print("=" * 60)
    
    demo_ste()
    demo_admm()
    demo_factorized_linear()
    demo_full_layer_quantization()
    
    print("\n" + "=" * 60)
    print("All demonstrations complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
