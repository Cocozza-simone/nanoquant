#!/usr/bin/env python3
"""
Quick test script to verify MPS memory fixes without running full quantization.
Tests configuration auto-adjustment and device detection.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from nanoquant.config import NanoQuantConfig
from nanoquant.device_utils import get_optimal_device, get_device_info

print("=" * 80)
print("🔍 NANOQUANT MPS Memory Fix - Verification Test")
print("=" * 80)

# Test 1: Device detection
print("\n✅ Test 1: Device Detection")
print("-" * 80)
device_info = get_device_info()
print(f"Platform: {device_info['platform']}")
print(f"PyTorch Version: {device_info['pytorch_version']}")
print(f"MPS Available: {device_info['mps_available']}")
print(f"CUDA Available: {device_info['cuda_available']}")

optimal = get_optimal_device("auto")
print(f"Optimal Device (auto): {optimal}")

# Test 2: MPS-specific configuration
print("\n✅ Test 2: MPS Configuration Auto-Adjustment")
print("-" * 80)

if device_info['mps_available']:
    # Test with large parameters (should be reduced)
    config = NanoQuantConfig(
        model_name="gpt2",
        calib_samples=128,  # Will be reduced
        calib_seq_len=2048,  # Will be reduced
        device="mps",
    )
    
    print(f"Original defaults:")
    print(f"  - calib_samples: 128")
    print(f"  - calib_seq_len: 2048")
    print(f"  - tune_fp_batch_size: 4")
    print(f"\nAfter MPS auto-adjustment:")
    print(f"  - calib_samples: {config.calib_samples}")
    print(f"  - calib_seq_len: {config.calib_seq_len}")
    print(f"  - tune_fp_batch_size: {config.tune_fp_batch_size}")
    print(f"  - device: {config.device}")
    
    if config.calib_samples < 128 and config.calib_seq_len < 2048:
        print("\n✅ MPS auto-adjustment working correctly!")
    else:
        print("\n❌ MPS auto-adjustment NOT working!")
else:
    print("MPS not available on this system. Skipping MPS-specific test.")

# Test 3: CPU configuration
print("\n✅ Test 3: CPU Configuration Auto-Adjustment")
print("-" * 80)
config = NanoQuantConfig(
    model_name="gpt2",
    calib_samples=128,  # Will be reduced for CPU
    calib_seq_len=2048,  # Will be reduced for CPU
    device="cpu",
)

print(f"After CPU auto-adjustment:")
print(f"  - calib_samples: {config.calib_samples}")
print(f"  - calib_seq_len: {config.calib_seq_len}")
print(f"  - device: {config.device}")

if config.calib_samples <= 32 and config.calib_seq_len <= 512:
    print("\n✅ CPU auto-adjustment working correctly!")
else:
    print("\n❌ CPU auto-adjustment NOT working!")

# Test 4: CUDA configuration (no reduction if available)
print("\n✅ Test 4: CUDA Configuration (No Reduction)")
print("-" * 80)
if device_info['cuda_available']:
    config = NanoQuantConfig(
        model_name="gpt2",
        calib_samples=128,
        calib_seq_len=2048,
        device="cuda",
    )
    
    print(f"CUDA configuration (should not be reduced):")
    print(f"  - calib_samples: {config.calib_samples}")
    print(f"  - calib_seq_len: {config.calib_seq_len}")
    
    if config.calib_samples == 128 and config.calib_seq_len == 2048:
        print("\n✅ CUDA configuration preserved (as expected)!")
    else:
        print("\n⚠️  CUDA was adjusted (might be intentional if device not available)")
else:
    print("CUDA not available on this system.")

# Test 5: Model loading test (MPS optimized script)
print("\n✅ Test 5: MPS-Optimized Script Availability")
print("-" * 80)

script_path = os.path.join(os.path.dirname(__file__), "quantize_mps_optimized.py")
if os.path.exists(script_path):
    print(f"✅ MPS-optimized script found: {script_path}")
    
    with open(script_path, 'r') as f:
        content = f.read()
        if 'MPS_MODELS' in content and 'gpt2' in content:
            print("✅ Pre-configured models found in script")
        else:
            print("❌ Pre-configured models NOT found")
else:
    print(f"❌ MPS-optimized script NOT found at: {script_path}")

print("\n" + "=" * 80)
print("✅ Verification Complete!")
print("=" * 80)
print("\nNext steps:")
print("  1. Test with MPS-optimized script:")
print("     python scripts/quantize_mps_optimized.py --model gpt2 --rank 4")
print("  2. Monitor memory in Activity Monitor (Memoria tab)")
print("  3. Check that MPS memory doesn't exceed 30GB")
print("\n" + "=" * 80)
