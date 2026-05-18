#!/usr/bin/env python3
"""
Quick test script to verify memory_optimized quantization works.
Tests basic functionality without long quantization runs.

Usage:
    python test_memory_optimized.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test all imports work."""
    print("\n✅ Testing imports...")
    try:
        from nanoquant import NanoQuantizer
        from nanoquant.evaluation import evaluate_perplexity
        from memory_configs import get_optimal_config, estimate_memory_usage
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_hardware_detection():
    """Test hardware auto-detection."""
    print("\n✅ Testing hardware detection...")
    try:
        from memory_configs import get_hardware_profile
        profile = get_hardware_profile()
        print(f"✅ Detected: {profile}")
        return True
    except Exception as e:
        print(f"❌ Hardware detection failed: {e}")
        return False


def test_config_generation():
    """Test configuration generation."""
    print("\n✅ Testing config generation...")
    try:
        from memory_configs import get_optimal_config
        
        configs_to_test = [
            ("gpt2", "auto"),
            ("opt-125m", "auto"),
        ]
        
        for model_name, gpu_type in configs_to_test:
            config = get_optimal_config(model_name, gpu_type)
            print(f"  ✅ {model_name}: rank={config.rank}, calib_samples={config.calib_samples}")
        
        return True
    except Exception as e:
        print(f"❌ Config generation failed: {e}")
        return False


def test_memory_estimation():
    """Test memory estimation."""
    print("\n✅ Testing memory estimation...")
    try:
        from memory_configs import get_optimal_config, estimate_memory_usage
        
        config = get_optimal_config("gpt2", "auto")
        mem_est = estimate_memory_usage(config)
        print(f"✅ GPT-2 estimated memory: {mem_est:.1f} GB")
        
        config = get_optimal_config("llama2-7b", "auto")
        mem_est = estimate_memory_usage(config)
        print(f"✅ Llama-2-7B estimated memory: {mem_est:.1f} GB")
        
        return True
    except Exception as e:
        print(f"❌ Memory estimation failed: {e}")
        return False


def test_model_loading():
    """Test actual model loading."""
    print("\n✅ Testing model loading (GPT-2 only)...")
    try:
        from memory_configs import get_optimal_config
        from nanoquant import NanoQuantizer
        
        config = get_optimal_config("gpt2", "auto")
        config.device = "cpu"  # Force CPU for testing
        
        quantizer = NanoQuantizer(config)
        quantizer.load_model()
        
        params = sum(p.numel() for p in quantizer.model.parameters())
        print(f"✅ Model loaded successfully!")
        print(f"  Model: {config.model_name}")
        print(f"  Parameters: {params / 1e6:.1f}M")
        print(f"  Device: {config.device}")
        
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_summary():
    """Print test summary."""
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("\n📋 Next steps:")
    print("  1. Run with dry-run to check config:")
    print("     python quantize_memory_optimized.py --model gpt2 --dry-run")
    print()
    print("  2. Run actual quantization:")
    print("     python quantize_memory_optimized.py --model gpt2 --output ./outputs/gpt2")
    print()
    print("  3. For larger models:")
    print("     python quantize_memory_optimized.py --model llama2-7b --quality conservative")
    print()


def main():
    print("\n" + "="*70)
    print("🧪 NANOQUANT Memory-Optimized - Test Suite")
    print("="*70)
    
    tests = [
        ("Imports", test_imports),
        ("Hardware Detection", test_hardware_detection),
        ("Config Generation", test_config_generation),
        ("Memory Estimation", test_memory_estimation),
        ("Model Loading", test_model_loading),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Print results
    print("\n" + "="*70)
    print("📊 Test Results:")
    print("="*70)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:30} {status}")
        if not result:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print_summary()
        return 0
    else:
        print("\n⚠️  Some tests failed. Check errors above.")
        return 1


if __name__ == "__main__":
    exit(main())
