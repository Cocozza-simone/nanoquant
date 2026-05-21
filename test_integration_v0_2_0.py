"""
Test per verificare l'integrazione di QMoE e OxiBonsai in NANOQUANT v0.2.0

Questo script verifica che:
1. I nuovi moduli si importano correttamente
2. La configurazione MoE funziona
3. L'esportazione GGUF funziona
4. La backward compatibility è mantenuta
"""

import sys
import torch
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test: Importa tutti i nuovi moduli"""
    logger.info("=" * 60)
    logger.info("TEST 1: Import dei nuovi moduli")
    logger.info("=" * 60)
    
    try:
        from nanoquant import (
            NanoQuantConfig,
            NanoQuantizer,
            MoEExpertQuantizer,
            export_to_gguf,
            pack_nanoquant_to_q1_0_g128,
            ternary_svd_init,
            apply_group_scaling,
        )
        logger.info("✅ Tutti i moduli importati con successo")
        return True
    except ImportError as e:
        logger.error(f"❌ Errore di import: {e}")
        return False


def test_config_moe():
    """Test: Configurazione MoE funziona"""
    logger.info("=" * 60)
    logger.info("TEST 2: Configurazione MoE")
    logger.info("=" * 60)
    
    from nanoquant import NanoQuantConfig
    
    # Test 1: Auto-detection per Mixtral
    config_mixtral = NanoQuantConfig(
        model_name="mistralai/Mixtral-8x7B-v0.1",
        rank=8,
    )
    
    if config_mixtral.moe_enabled:
        logger.info("✅ Auto-detection MoE per Mixtral: OK")
    else:
        logger.error("❌ Auto-detection MoE fallito")
        return False
    
    # Test 2: Parametri MoE
    config_custom = NanoQuantConfig(
        model_name="custom-model",
        rank=8,
        moe_enabled=True,
        quantize_only_experts=True,
        tie_hessians=True,
        expert_parallelism=False,
    )
    
    if (config_custom.moe_enabled and 
        config_custom.quantize_only_experts and 
        config_custom.tie_hessians):
        logger.info("✅ Parametri MoE configurati correttamente")
        return True
    else:
        logger.error("❌ Parametri MoE non impostati")
        return False


def test_moe_expert_quantizer():
    """Test: MoEExpertQuantizer funziona"""
    logger.info("=" * 60)
    logger.info("TEST 3: MoEExpertQuantizer")
    logger.info("=" * 60)
    
    from nanoquant import NanoQuantConfig, MoEExpertQuantizer
    import torch.nn as nn
    
    config = NanoQuantConfig(
        model_name="test",
        rank=4,
        moe_enabled=True,
        device="cpu",
    )
    
    quantizer = MoEExpertQuantizer(config)
    
    # Test: Identificazione layer expert
    test_names = [
        "layer.0.mlp.experts.0.w_in",
        "layer.0.mlp.experts.1.w_out",
        "layer.0.moe.experts.0.dense",
        "layer.0.attention.self.query",
        "layer.0.gate.w",
    ]
    
    results = []
    for name in test_names:
        is_expert = quantizer.is_expert_layer(name)
        is_gate = quantizer.is_gate_layer(name)
        results.append((name, is_expert, is_gate))
        status = "expert" if is_expert else ("gate" if is_gate else "other")
        logger.info(f"  {name}: {status}")
    
    # Verifica risultati attesi
    expected = [
        ("layer.0.mlp.experts.0.w_in", True, False),
        ("layer.0.mlp.experts.1.w_out", True, False),
        ("layer.0.moe.experts.0.dense", True, False),
        ("layer.0.attention.self.query", False, False),
        ("layer.0.gate.w", False, True),
    ]
    
    if results == expected:
        logger.info("✅ Identificazione layer expert: OK")
        return True
    else:
        logger.error("❌ Identificazione layer expert fallita")
        return False


def test_ternary_init():
    """Test: Ternary initialization funziona"""
    logger.info("=" * 60)
    logger.info("TEST 4: Ternary Initialization (QMoE)")
    logger.info("=" * 60)
    
    from nanoquant import ternary_svd_init, estimate_init_quality
    
    # Crea una matrice test
    W = torch.randn(64, 128)
    
    try:
        # Testa ternary_svd_init
        U_init, V_init = ternary_svd_init(W, rank=4, sparsity=0.9)
        
        # Verifica shape
        assert U_init.shape == (64, 4), f"Shape U: {U_init.shape}"
        assert V_init.shape == (128, 4), f"Shape V: {V_init.shape}"
        
        # Calcola qualità init
        error = estimate_init_quality(W, U_init, V_init)
        logger.info(f"  Initial reconstruction error: {error:.4f}")
        
        logger.info("✅ Ternary initialization: OK")
        return True
    except Exception as e:
        logger.error(f"❌ Ternary init fallita: {e}")
        return False


def test_gguf_export():
    """Test: GGUF export funziona"""
    logger.info("=" * 60)
    logger.info("TEST 5: GGUF Export (OxiBonsai)")
    logger.info("=" * 60)
    
    from nanoquant import pack_nanoquant_to_q1_0_g128, export_to_gguf
    
    # Crea dati test
    U_binary = torch.sign(torch.randn(32, 4))
    V_binary = torch.sign(torch.randn(64, 4))
    s1 = torch.abs(torch.randn(32)) + 0.1
    s2 = torch.abs(torch.randn(64)) + 0.1
    
    try:
        # Testa pack_nanoquant_to_q1_0_g128
        signs, scales = pack_nanoquant_to_q1_0_g128(U_binary, V_binary, s1, s2)
        
        logger.info(f"  Signs packed: {len(signs)} bytes")
        logger.info(f"  Scales: {len(scales)} entries (FP16)")
        
        # Testa export_to_gguf (no actual file write, just format)
        quantized_layers = {
            "test.layer.0": {
                "U_binary": U_binary,
                "V_binary": V_binary,
                "s1": s1,
                "s2": s2,
            }
        }
        
        output_path = export_to_gguf(
            quantized_layers=quantized_layers,
            model_metadata={"architecture": "test", "context_length": 4096},
            output_path="/tmp/test_gguf.gguf"
        )
        
        if output_path.exists():
            file_size = output_path.stat().st_size
            logger.info(f"  GGUF file created: {file_size} bytes")
            output_path.unlink()  # Cleanup
            logger.info("✅ GGUF export: OK")
            return True
        else:
            logger.error("❌ GGUF file not created")
            return False
            
    except Exception as e:
        logger.error(f"❌ GGUF export fallita: {e}")
        return False


def test_backward_compatibility():
    """Test: Backward compatibility"""
    logger.info("=" * 60)
    logger.info("TEST 6: Backward Compatibility")
    logger.info("=" * 60)
    
    from nanoquant import (
        NanoQuantConfig,
        LatentBinaryADMM,
        pack_binary_tensor,
        unpack_binary_tensor,
    )
    
    try:
        # Test: Config senza parametri MoE funziona
        config = NanoQuantConfig(model_name="test", rank=4)
        assert not config.moe_enabled
        logger.info("  Config senza MoE: OK")
        
        # Test: ADMM solver funziona
        admm = LatentBinaryADMM(rank=4, num_iterations=10)
        W_test = torch.randn(16, 32)
        U, V, s1, s2 = admm.solve_simple(W_test)
        assert U.shape == (16, 4)
        assert V.shape == (32, 4)
        logger.info("  ADMM solver: OK")
        
        # Test: Packing funziona
        binary_tensor = torch.ones(8, 8).sign()
        packed, shape = pack_binary_tensor(binary_tensor)
        unpacked = unpack_binary_tensor(packed, shape)
        assert torch.allclose(binary_tensor, unpacked)
        logger.info("  Packing/unpacking: OK")
        
        logger.info("✅ Backward compatibility: OK")
        return True
        
    except Exception as e:
        logger.error(f"❌ Backward compatibility fallita: {e}")
        return False


def main():
    """Esegui tutti i test"""
    logger.info("\n" + "=" * 60)
    logger.info("NANOQUANT v0.2.0 - Test Suite")
    logger.info("=" * 60 + "\n")
    
    tests = [
        ("Imports", test_imports),
        ("Config MoE", test_config_moe),
        ("MoE Expert Quantizer", test_moe_expert_quantizer),
        ("Ternary Initialization", test_ternary_init),
        ("GGUF Export", test_gguf_export),
        ("Backward Compatibility", test_backward_compatibility),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ Test {test_name} crashed: {e}")
            results.append((test_name, False))
        logger.info("")  # Empty line
    
    # Riepilogo
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info("=" * 60)
    logger.info(f"Result: {passed}/{total} tests passed")
    logger.info("=" * 60 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
