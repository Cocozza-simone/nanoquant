# 🚀 Guida Veloce - NANOQUANT su macOS M4 Pro

## ✅ Problemi Risolti

1. **Device Auto-Detection** ✅
   - Usa automaticamente **MPS** (GPU accelerata) su macOS M4 Pro
   - Fallback a CPU se MPS non disponibile
   - Non richiede configurazione manuale

2. **Modelli Gated HuggingFace** ✅
   - Script di demo usa modelli pubblici (GPT2, etc.)
   - Istruzioni per modelli protetti (Llama-2)

## 🎯 Come Iniziare (3 minuti)

### Opzione 1: Demo (Consigliato - Funziona Subito!)

```bash
cd /Users/simonecocozza/Downloads/NanoQuant\(3\)/nanoquant
source .venv/bin/activate

# GPT2 (124M, no auth needed)
python scripts/quantize_demo.py --model gpt2 --rank 4

# OPT (125M, no auth needed)
python scripts/quantize_demo.py --model opt --rank 4
```

**Output atteso:**
```
Platform: Darwin
Optimal Device: mps  ✅
✅ Device resolved to: mps
✅ Config created successfully
✅ Model loaded successfully!
Parameters: 124,439,808
Device: mps
```

### Opzione 2: Llama-2 (Richiede Login HuggingFace)

```bash
# 1. Fai login a HuggingFace
huggingface-cli login
# Incolla il token da https://huggingface.co/settings/tokens

# 2. Accetta i termini di Llama-2
# Vai a: https://huggingface.co/meta-llama/Llama-2-7b-hf
# Clicca "Access repository"

# 3. Riesegui
python scripts/quantize.py --model meta-llama/Llama-2-7b-hf --rank 8 --bits 1.0
```

## 📊 Device Detection Report

Per verificare che il device sia stato risolto correttamente:

```python
from src.nanoquant import get_device_info

info = get_device_info()
print(f"Platform: {info['platform']}")              # Darwin
print(f"Optimal Device: {info['optimal_device']}")  # mps
print(f"MPS Available: {info['mps_available']}")    # True
print(f"CUDA Available: {info['cuda_available']}")  # False
```

**Output su macOS M4 Pro:**
```
Platform: Darwin
Optimal Device: mps       ← GPU accelerata! 🚀
MPS Available: True       ← Metal Performance Shaders
CUDA Available: False     ← Non disponibile su macOS
PyTorch Version: 2.12.0
```

## 🔧 Configurazione Manuale (Se Necessario)

```python
from src.nanoquant import NanoQuantConfig

# Auto (CONSIGLIATO)
config = NanoQuantConfig(device="auto")  # Usa MPS su M4 Pro

# Forzare dispositivo specifico
config = NanoQuantConfig(device="mps")   # Metal Performance Shaders
config = NanoQuantConfig(device="cpu")   # CPU puro
config = NanoQuantConfig(device="cuda")  # CUDA (non disponibile su macOS)
```

## 📚 Modelli Consigliati (No Auth)

| Nome | Parametri | Memory | Comando |
|------|-----------|--------|---------|
| GPT2 | 124M | ~500MB | `python scripts/quantize_demo.py --model gpt2` |
| OPT-125M | 125M | ~500MB | `python scripts/quantize_demo.py --model opt` |
| DistilBERT | 66M | ~300MB | `python scripts/quantize_demo.py --model distilbert` |

## 🔑 Come Fare Login a HuggingFace

### Metodo 1: CLI (Consigliato)
```bash
huggingface-cli login
# Incolla il token e premi Enter
# File salvato in: ~/.cache/huggingface/token
```

### Metodo 2: Environment Variable
```bash
export HF_TOKEN="hf_xxxxxxxxxxxx"
python scripts/quantize.py --model meta-llama/Llama-2-7b-hf
```

### Metodo 3: Programmaticamente
```python
from huggingface_hub import login
login(token="hf_xxxxxxxxxxxx")

# Ora puoi usare modelli gated
```

### Dove Trovare il Token:
1. Vai su https://huggingface.co/settings/tokens
2. Clicca "New token"
3. Dagli un nome: "NANOQUANT"
4. Seleziona "read" (o migliore se vuoi fare modifiche)
5. Copia il token

## ⚡ Performance su M4 Pro

Rispetto a CPU puro:
- **MPS (GPU)**: 5-10x più veloce ⚡
- **CPU**: 1x (baseline)

La quantizzazione di GPT2 dovrebbe prendere:
- Su MPS: ~5-10 minuti
- Su CPU: ~30-60 minuti

## 🧪 Verifica Installation

```bash
cd /Users/simonecocozza/Downloads/NanoQuant\(3\)/nanoquant
source .venv/bin/activate

# Test device
python -c "
from src.nanoquant import get_device_info
info = get_device_info()
print(f'✅ Device: {info[\"optimal_device\"]}')"

# Run tests
python -m pytest tests/test_nanoquant.py -v  # Dovrebbe passare 26/26
```

## 🆘 Troubleshooting

### Errore: "401 Unauthorized"
```
❌ Cannot access gated repo for url https://huggingface.co/...
```
**Soluzione:**
```bash
huggingface-cli login
# Oppure usa modello pubblico: python scripts/quantize_demo.py
```

### Errore: "Torch not compiled with CUDA"
```
❌ AssertionError: Torch not compiled with CUDA enabled
```
**Soluzione:** ✅ Risolto! Il codice ora usa auto-detection
```python
from src.nanoquant import get_device_info
info = get_device_info()  # Mostra device disponibili
```

### Device rimane su "cuda"
```
❌ Device: cuda (ma CUDA non disponibile!)
```
**Soluzione:** Usa `--device auto`
```bash
python scripts/quantize.py --device auto  # Ora usa MPS
```

## 📖 Ulteriori Letture

- [PyTorch Metal Performance Shaders](https://pytorch.org/docs/stable/notes/mps.html)
- [HuggingFace Authentication](https://huggingface.co/docs/hub/security)
- [NANOQUANT Paper](https://arxiv.org/abs/2602.06694)
- [NanoQuant README Completo](./README.md)

## ✨ Prossimi Passi

1. ✅ Esegui demo: `python scripts/quantize_demo.py`
2. ✅ Verifica device: `python -c "from src.nanoquant import get_device_info; print(get_device_info())"`
3. ✅ Esegui test: `python -m pytest tests/test_nanoquant.py -v`
4. 🚀 Avvia quantizzazione completa su modello di scelta

**Buon lavoro! 🎉**
