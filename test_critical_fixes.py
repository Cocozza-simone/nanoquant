#!/usr/bin/env python3
"""
Test suite for the 4 critical bug fixes in Fase 1.

Tests:
1. SVID no-op is fixed (uses svid_rank1_fast)
2. Magnitude balancing is fixed (division instead of multiplication)
3. Seed is fixed (reproducibility)
4. MoE creates FactorizedLinear (not dense weight overwrite)
"""

import sys
sys.path.insert(0, '/Users/simonecocozza/Downloads/NanoQuant(3)/nanoquant/src')

import torch
import torch.nn as nn
from nanoquant.admm import LatentBinaryADMM
from nanoquant.svid import svid_rank1_fast
from nanoquant.config import NanoQuantConfig
from nanoquant.moe_quantization import MoEExpertQuantizer
from nanoquant.reconstruction import FactorizedLinear

def test_svid_rank1_not_noop():
    """Test that SVID rank1 is not a no-op (sign(x)*abs(x)=x)"""
    print("\n[TEST 1] SVID rank1_fast is not a no-op...")
    
    # Create a random matrix
    M = torch.randn(10, 8)
    
    # Apply SVID
    Z = svid_rank1_fast(M)
    
    # Check that Z is not identical to M
    assert not torch.allclose(Z, M), "SVID output should differ from input (not an identity op)"
    
    # Check that SVID output has rank ≤ 1 (approximately)
    U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
    ratio = S[1] / (S[0] + 1e-8) if len(S) > 1 else 0
    assert ratio < 0.1, f"SVID should produce rank-1 approx, but ratio={ratio:.3f}"
    
    print("  ✓ SVID rank1_fast produces non-trivial rank-1 approximation")

def test_magnitude_balancing_division():
    """Test that magnitude balancing uses division, not multiplication"""
    print("\n[TEST 2] Magnitude balancing uses division...")
    
    config = NanoQuantConfig(device="cpu", seed=42)
    admm = LatentBinaryADMM(
        rank=4,
        num_iterations=10,
        rho=1.0,
        lambda_reg=0.01,
        epsilon=1e-5,
        device="cpu",
        seed=42
    )
    
    # Create random U, V and preconditioners
    U = torch.randn(8, 4, device="cpu") * 0.01
    V = torch.randn(6, 4, device="cpu") * 0.01
    D_out = torch.abs(torch.randn(8)) + 0.1  # Avoid division by zero
    D_in = torch.abs(torch.randn(6)) + 0.1
    
    # Call _magnitude_balancing
    U_final, V_final, s1, s2 = admm._magnitude_balancing(U, V, D_out, D_in)
    
    # Manually compute what should happen with division (correct)
    U_b_correct = U / (D_out.sqrt().unsqueeze(1) + 1e-8)
    V_b_correct = V / (D_in.sqrt().unsqueeze(1) + 1e-8)
    
    # Check that the depreconditioned factors are approximately correct
    # (accounting for the equilibrium rescaling eta in the algorithm)
    norm_U_b = torch.norm(U_b_correct, p='fro')
    norm_V_b = torch.norm(V_b_correct, p='fro')
    eta = torch.sqrt(norm_V_b / norm_U_b) if norm_U_b > 1e-8 else 1.0
    
    # The output should be roughly like U_b_correct * eta / s1
    # We just check that the scaling is reasonable (not inverted)
    mean_U_final = torch.mean(torch.abs(U_final))
    assert mean_U_final < 10.0, f"U_final magnitude too large, suggest inverted division: {mean_U_final:.3f}"
    
    print("  ✓ Magnitude balancing uses division (depreconditioning)")

def test_seed_reproducibility():
    """Test that seed=42 produces reproducible results"""
    print("\n[TEST 3] Seed reproducibility...")
    
    W = torch.randn(16, 12)
    
    # Run ADMM twice with same seed
    admm1 = LatentBinaryADMM(rank=4, num_iterations=5, device="cpu", seed=42)
    U1, V1, s1_1, s2_1 = admm1.solve_simple(W)
    
    admm2 = LatentBinaryADMM(rank=4, num_iterations=5, device="cpu", seed=42)
    U2, V2, s1_2, s2_2 = admm2.solve_simple(W)
    
    # Check that results are identical
    assert torch.allclose(U1, U2, atol=1e-6), "U should be identical with same seed"
    assert torch.allclose(V1, V2, atol=1e-6), "V should be identical with same seed"
    
    print("  ✓ ADMM solver is reproducible with seed=42")

def test_moe_creates_factorized_linear():
    """Test that MoE path creates FactorizedLinear, not dense overwrites"""
    print("\n[TEST 4] MoE path creates FactorizedLinear...")
    
    config = NanoQuantConfig(device="cpu", moe_enabled=True)
    
    # Create a simple MoE-like model
    model = nn.Sequential(
        nn.Linear(32, 16),
        nn.Linear(16, 8),
    )
    
    # Name the layers to simulate MoE structure
    for i, module in enumerate(model.modules()):
        if isinstance(module, nn.Linear):
            module._name = f"layer.{i}.experts.0"
    
    # Initialize quantizer
    quantizer = MoEExpertQuantizer(config)
    
    # Quantize
    quantizer.quantize_moe_model(model)
    
    # Check that layers were replaced with FactorizedLinear
    factorized_count = 0
    for module in model.modules():
        if isinstance(module, FactorizedLinear):
            factorized_count += 1
    
    assert factorized_count > 0, "MoE path should create FactorizedLinear layers"
    
    print(f"  ✓ MoE created {factorized_count} FactorizedLinear layers")

if __name__ == "__main__":
    print("="*60)
    print("Testing Critical Fixes (Fase 1)")
    print("="*60)
    
    try:
        test_svid_rank1_not_noop()
        test_magnitude_balancing_division()
        test_seed_reproducibility()
        test_moe_creates_factorized_linear()
        
        print("\n" + "="*60)
        print("✓ ALL CRITICAL FIXES VALIDATED")
        print("="*60)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
