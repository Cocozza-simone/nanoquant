from src.nanoquant import NanoQuantConfig, NanoQuantizer

# Configura la quantizzazione
config = NanoQuantConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    rank=8,                    # Fattorizzazione low-rank
    bits=1.0,                  # 1-bit compression
    calib_samples=128,         # Campioni calibrazione
    # Device auto-detect: usa MPS su M4 Pro! 🎯
)

# Crea il quantizzatore
quantizer = NanoQuantizer(config)

# Carica il modello
quantizer.load_model()

# Quantizza
quantizer.quantize()

# Valuta
results = quantizer.evaluate()
print(f"Perplexity: {results['perplexity']}")

# Salva
quantizer.save_quantized_model("./outputs/quantized")