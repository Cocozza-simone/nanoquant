# NANOQUANT - Report di Analisi Dettagliata (Revisione v2)

> **Nota revisione**: Questo documento incorpora correzioni derivate da un confronto sistematico
> con i paper di riferimento: *NANOQUANT: Efficient Sub-1-Bit Quantization of Large Language
> Models* (arXiv:2602.06694v2) e *Optimizing Neural Networks with Kronecker-factored Approximate
> Curvature* — K-FAC (arXiv:1503.05671v7). Le sezioni modificate sono marcate con `[CORRETTO]`
> o `[AGGIUNTO]`; quelle invariate sono mantenute intatte.

## Indice

1. [Panoramica del Progetto](#panoramica-del-progetto)
2. [Architettura Generale](#architettura-generale)
3. [Analisi per Modulo](#analisi-per-modulo)
4. [Valutazione della Qualità del Codice](#valutazione-della-qualità-del-codice)
5. [Analisi delle Performance](#analisi-delle-performance)
6. [Problemi di Robustezza](#problemi-di-robustezza)
7. [Suggerimenti di Miglioramento](#suggerimenti-di-miglioramento)
8. [Considerazioni Architetturali](#considerazioni-architetturali)
9. [Feature Mancanti o Incompleti](#feature-mancanti-o-incompleti)
10. [Conclusioni](#conclusioni)

---

## Panoramica del Progetto

NANOQUANT è un framework di post-training quantization (PTQ) per modelli LLM che raggiunge
compressione sub-1-bit tramite fattorizzazione binaria a basso rango. Il progetto è attualmente
nella versione 0.2.0, con recenti integrazioni da QMoE (IST-DASLab) e OxiBonsai (COOLJAPAN).

### Dati Chiave

| Aspetto | Dettaglio |
|---------|-----------|
| **Versione** | 0.2.0 |
| **Linguaggio** | Python 3.9+ |
| **Framework** | PyTorch 2.0+, HuggingFace Transformers |
| **Moduli Core** | 18+ file Python |
| **Test** | ~670 righe di test unitari |
| **Script** | 5+ script di utilità |
| **Integrazioni** | QMoE (MoE), OxiBonsai (GGUF) |

---

## Architettura Generale

### Pipeline a Tre Fasi

```
┌─────────────────────────────────────────────────────────────┐
│                    NANOQUANT Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│  FASE 1: Global Calibration (calibration.py)                │
│  ├── Raccolta statistiche K-FAC con regularizzazione        │
│  └── Calcolo preconditioner D_in, D_out per ogni layer     │
│                                                               │
│  FASE 2: Block Reconstruction (reconstruction.py)             │
│  ├── Step 1: TUNEFP - Error Propagation Mitigation           │
│  ├── Step 2: LB-ADMM - Low-Rank Binary Factorization         │
│  └── Step 3: TUNELATENTSTE - Component Refinement          │
│                                                               │
│  FASE 3: Model Reconstruction (model_reconstruction.py)        │
│  └── TUNESCALESKD - Global Scale Tuning via KL Divergence  │
└─────────────────────────────────────────────────────────────┘
```

---

## Analisi per Modulo

### 1. `config.py` - Configurazione

**Punti di Forza:**
- Dataclass ben strutturata con parametri chiari
- Auto-adattamento per famiglie di modelli (Llama, Gemma)
- Correzioni automatiche per dispositivi MPS e CPU
- Supporto MoE con parametri dedicati

**Problemi Identificati:**

```python
# 1. Parametri di ottimizzazione duplicati e potenzialmente in conflitto
tune_fp_epochs = 8
tune_fp_lr = 1e-4
tune_fp_batch_size = 4
pre_tune_steps = 20      # Legacy - ridefinisce il sistema epochs/steps
pre_tune_lr = 1e-5       # Conflitto con tune_fp_lr
post_tune_lr = 1e-4      # Diverso da tune_latent_lr=1e-5

# 2. Metodo adapt_for_model_family usa solamente string matching
# Questo e' fragile e non gestisce varianti come "meta-llama/Llama-2-7b-chat-hf"
```

> **[AGGIUNTO]** I valori di default documentati nel paper (Appendice C) sono:
> learning rate 1e-4 per pre-factorized tuning, 1e-5 per factorized tuning, 1e-6 per
> global scale reconstruction, 8 epoche per tutte le fasi con cosine LR scheduler,
> batch size 4 / 1 / 1 rispettivamente. Il config dovrebbe allinearsi esplicitamente
> a questi valori canonici ed eliminare i parametri legacy che li contraddicono.

**Valutazione**: 6.5/10 - Buona struttura ma con incoerenze tra parametri legacy e nuovi.

---

### 2. `quantization.py` - Quantizzatore Principale

**Punti di Forza:**
- Orchestrazione pulita delle tre fasi
- Gestione degli errori per block (try/except)
- Logging informativo

**Problemi Critici:**

```python
# 1. Block I/O Capture - Potenziale Memory Leak
# Il metodo _get_block_inputs_outputs registra hook che potrebbero 
# non essere rimossi in caso di eccezione

def _get_block_inputs_outputs(...):
    handles = []
    # ... registrazione hook ...
    # Se questa riga solleva un'eccezione, gli hook NON vengono rimossi
    with torch.no_grad():
        model(input_ids=input_ids[:8], ...)  # Potrebbe fallire
    # Rimozione hook dopo - ma potrebbe non essere raggiunta

# 2. Il metodo _get_block_inputs_outputs usa solo 8 campioni
# ma non c'e' motivo chiaramente documentato per questo valore fisso
X, Y_star = self._get_block_inputs_outputs(...)
# In _get_calibration_inputs vengono usati tutti i calib_samples
# ma per I/O dei blocchi solo 8? Possibile perdita di accuratezza

# 3. Sostituzione layer: metodi ad hoc senza gestione tipo
# Il check 'if part.isdigit()' e' fragile per nomi come "layer_0"
```

> **[AGGIUNTO]** Il paper (Algorithm 1, riga 9) usa esplicitamente `Xcal` completo per
> calcolare gli input di ogni blocco: `Xb ← Mc<b(Xcal)`. Limitarsi a 8 campioni fissi
> è una discrepanza rispetto alla specifica. Il valore dovrebbe almeno essere configurabile
> tramite un parametro dedicato (es. `block_io_samples`), o allineato a `calib_samples`.

**Valutazione**: 6/10 - Logica core corretta ma con potenziali leak e inconsistenze.

---

### 3. `calibration.py` - Calibrazione Globale

**Punti di Forza:**
- Classe `_OnlineSecondMoment` per accumulo online con footprint minimo
- Separazione chiara tra phase forward (attivazioni) e backward (gradienti)
- Supporto esplicito per MPS con CPU offload

**Problemi Identificati:**

```python
# 1. Accumulo su CPU ma poi trasferisce su device config
# Questo crea overhead di trasferimento dati
self.D_in[name] = D_in.to(self.device)  # Sposta su GPU/MPS ad ogni layer

# 2. Il metodo _compute_preconditioners_with_hooks ha una struttura monolitica
# con ripetizione di codice tra forward e backward. Dovrebbe essere decomposto.

# 3. Il parametro 'trust_remote_code=True' e' sempre True senza possibilità
# di override. Questo e' un potenziale rischio di sicurezza.

# 4. Nessun check che il modello sia in eval mode durante la calibrazione
# Il forward delle linear puo' avere comportamenti diversi (es. Dropout)
```

> **[AGGIUNTO — Shrinkage Regularization non documentata]**
> Il paper (eq. 3) specifica un passo di shrinkage obbligatorio sui precondizionatori
> diagonali, assente dalla documentazione interna:
> ```
> [D̃_(·)]_ii ← (1 − γ)[D_(·)]_ii + γ mean(D_(·))
> ```
> Il coefficiente γ ∈ [0,1] ha valori ottimali **diversi per famiglia di modello**:
> γ = 0.2 per Llama e Qwen, γ = 0.6 per Gemma e Rnj. Questo passo è critico per la
> stabilità con calibration set piccoli (128 campioni) e deve essere esplicitamente
> documentato nel modulo e nei docstring. Se assente dall'implementazione, è una
> lacuna algoritmicamente rilevante.

**Valutazione**: 7/10 - Implementazione solida per la raccolta K-FAC ma con ridondanza.

---

### 4. `admm.py` - Solver ADMM

**Punti di Forza:**
- Implementazione corretta delle equazioni paper (Cholesky, SVID)
- Integrazione con ternary_init per inizializzazione migliorata
- Fallback robusto a pinv se Cholesky fallisce

**Problemi Identificati:**

```python
# 1. Unit test debole: "errore relativo < 0.95" e' una soglia molto bassa
# per una tecnica di compression che dovrebbe essere molto accurata
assert error < 0.95

# 2. La funzione svd_sign_value_decomposition non e' usata in modo critico
# e fa sign().abs() che annulla l'effetto: M_approx.sign() * M_approx.abs() = M_approx

# 3. Il parametro device e' hardcoded a "cpu" per default:
if device is None:
    self.device = "cpu"  # Perche' default CPU anche con CUDA disponibile?

# 4. Creazione temporanea di solver ADMM per rank diversi:
# In reconstruction.py, per ogni layer con rank effettivo diverso,
# viene creato un nuovo LatentBinaryADMM. Questo e' inefficiente.
```

> **[CORRETTO — Cholesky non è overhead, è ottimizzazione deliberata]**
> Il punto 4 originale classificava Cholesky come inefficienza. Il paper (Step 2-2)
> chiarisce il contrario: la scelta di Cholesky riduce la complessità da O(2r³/3)
> della fattorizzazione LU generale a **O(r³/3)**, ed è descritta esplicitamente come
> ottimizzazione che "enables NANOQUANT to scale efficiently to massive architectures
> (e.g., Llama-2-70B) within limited computational budgets." Il problema reale è la
> creazione di un nuovo istanza solver per ogni rank diverso, non il Cholesky in sé.
>
> **[AGGIUNTO — Garanzia di convergenza]**
> L'Appendice B del paper dimostra formalmente la proprietà di **Monotonic Descent**
> (Theorem 3): se il parametro di penalità ρ > L_f (costante di Lipschitz di ∇f),
> la sequenza di iterate non aumenta il Lagrangiano aumentato. I sub-problemi per U e V
> sono Symmetric Positive Definite per qualsiasi ρ > 0 (Lemma 2). Il test con soglia
> 0.95 è quindi troppo lasso rispetto alle garanzie teoriche: andrebbe tarato su ρ e
> sulla dimensione del layer.

**Valutazione**: 7/10 - Implementazione matematicamente fondata con garanzie teoriche;
inefficienza reale è la re-istanziazione del solver, non il Cholesky.

---

### 5. `reconstruction.py` - Ricostruzione a Blocchi

**Punti di Forza:**
- Implementazione completa dei tre step della pipeline
- Integrazione diretta tra i componenti

**Problemi Critici:**

```python
# 1. Il metodo _error_propagation_mitigation usa MSE loss diretta
# Ma non ci sono collegamenti con i dati di calibrazione effettivi
# Calcola Y_star come output del blocco originale ma poi lo usa
# come target per l'ottimizzazione del blocco stesso (circolare!)

# 2. Il metodo _factorized_component_refinement sostituisce i moduli
# con setattr ma non implementa correttamente la forward con factorized layers.
# La classe StraightThroughEstimator esiste in refinement.py ma non e' usata
# direttamente in FactorizedLinear.forward

# 3. Il metodo _factorized_component_refinement non e' distaccato dalla
# classe BlockReconstructionPipeline in modo pulito per il testing

# 4. Riferimento circolare tra reconstruction.py e i moduli di tuning:
# BlockReconstructionPipeline chiama i propri metodi _factorized_* invece
# di delegare a classi specializzate come StraightThroughEstimator ecc.
```

> **[AGGIUNTO — Allineamento con la specifica del paper]**
> Il paper (eq. 10) definisce l'obiettivo del refinement come:
> ```
> min_{U,V,s1,s2} ‖B(Xin) − B̂(Xin; sign(U), sign(V), s1, s2)‖²_F
> ```
> dove B è il blocco full-precision e B̂ il blocco quantizzato con tutti i blocchi
> precedenti congelati. Il pattern non è circolare: Y_star viene calcolato dal
> teacher frozen, non dall'istanza in ottimizzazione. Se l'implementazione usa
> lo stesso oggetto per entrambi, è un bug di riferimento, non un problema di design.
> L'Appendice D.3 mostra che il flip ratio post-refinement è 0.47%–6.82%: la
> convergenza STE è empiricamente solida nonostante la non-convessità.

**Valutazione**: 5/10 - Il cuore della pipeline è presente ma con errori logici importanti.

---

### 6. `packing.py` - Packing Binario

**Punti di Forza:**
- Implementazione nativa PyTorch di pack/unpack
- Fallback a NumPy per compatibilità
- Classe PackedBinaryStorage per gestione multi-layer

**Problemi Identificati:**

```python
# 1. La funzione torch_packbits usa moltiplicazione e somma invece di shift/or
# Causa overhead su grandi matrici
# (bits * weights).sum() invece di bitwise ops

# 2. La funzione torch_unpackbits rigenera 'shifts' ad ogni chiamata
# Dovrebbe essere bufferizzato o memoizzato

# 3. Il fallback a NumPy scarica su CPU e poi rimanda su device
# Questo e' molto inefficiente per tensori su GPU

# 4. Il modulo usa import condizionale che causa circular import risk:
from .packing import pack_binary_matrix  # dentro gguf_export.py
```

> **[AGGIUNTO — Contesto kernel produzione]**
> L'Appendice E.2 del paper descrive il kernel GEMV produzione: i bit sono decompressi
> on-the-fly con una lightweight mask operation, seguita da FMA su float16/bfloat16.
> Il kernel non usa Tensor Core WMMA ed è intenzionalmente matmul-free per compatibilità
> su hardware senza Tensor Cores (es. NVIDIA Jetson TX2). L'inefficienza del packing
> Python è quindi reale, ma va contestualizzata: è separata dai kernel CUDA produzione
> che gestiscono il caso critico in modo ottimizzato.

**Valutazione**: 6.5/10 - Funzionalità corretta ma implementazione non ottimizzata.

---

### 7. `kernels.py` - Kernel Ottimizzati

**Punti di Forza:**
- `@torch.compile(mode="reduce-overhead")` per accelerazione
- Due modalità: packed e unpacked

**Problemi Identificati:**

```python
# 1. La forward di OptimizedFactorizedLinear fa Y = s1 * (x @ V @ U^T) * s2
# Questo NON corrisponde alla formula del paper:
# W ≈ s1 ⊙ (U V^T) ⊙ s2^T dove ⊙ e' Hadamard product
# In realta' viene calcolato: output = s1 * (x @ V @ U^T) dove x e' scalato da s2 prima
# Il calcolo dovrebbe essere: x @ (V * s2) poi (x @ V) @ (U * s1).T oppure
# conviene pre-computare le matrici effettive per ogni layer

# 2. Il metodo binary_gemv_simple ha un commento errato:
# "s2 is already absorbed in the input" ma in realta' viene applicato dopo

# 3. Unpacking ad ogni forward pass per packed weights
# Questo e' estremamente inefficiente - dovrebbe avvenire una sola volta

# 4. Il metodo get_weight_matrix() ricostruisce W completa ad ogni chiamata
# In un contesto di debug/testing questo va bene ma non per inferenza
```

> **[CORRETTO — Unpack on-the-fly è design intenzionale, non bug]**
> Il punto 3 classifica l'unpack ad ogni forward come "estremamente inefficiente".
> Il paper (Appendice E.2) descrive questo pattern come una scelta progettuale
> deliberata del kernel GEMV produzione: "the bits are unpacked on-the-fly with a
> lightweight mask operation." Il motivo è evitare l'overhead di memoria di mantenere
> i pesi decompressi, con risparmio di bandwidth particolarmente rilevante su hardware
> memory-bound. Il problema reale in `kernels.py` è che l'unpack avviene in Python
> PyTorch invece che in CUDA, rendendo il codice Python non competitivo con il kernel
> CUDA produzione documentato nel paper. La raccomandazione corretta è: allineare
> l'implementazione Python al pattern del kernel CUDA, non eliminare l'unpack on-the-fly.
>
> **[CORRETTO — Formula forward]**
> Il punto 1 è valido: la formula del paper (eq. 1) è
> `Ŵ = s1 ⊙ (U±1 V±1ᵀ) ⊙ s2ᵀ` con prodotto Hadamard e broadcasting espliciti.
> La forward corretta in termini computazionali è:
> `output = (x · s2) @ V @ Uᵀ · s1ᵀ` (scala input canale per canale, poi GEMM,
> poi scala output canale per canale), oppure equivalentemente le scale vengono
> assorbite in U e V prima della moltiplicazione per evitare operazioni scalari
> intermedie — strategia documentata nel paper come ottimizzazione per hardware
> accelerator (Appendice C: "the core linear transformation proceeds sequentially
> without intervening scalar operations").

**Valutazione**: 5/10 - I kernel Python non sono competitivi con l'implementazione
CUDA del paper; la formula ha una discrepanza reale rispetto alla specifica.

---

### 8. `inference.py` - Motore di Inferenza

**Punti di Forza:**
- Wrapper conveniente per utilizzo endpoint
- Benchmark automatico con warmup

**Problemi Identificati:**

```python
# 1. Il metodo generate() usa .to(self.device) ma self.device potrebbe essere "auto"
# Il controllo e' in create_inference_engine ma non e' robusto

# 2. Il metodo _optimize_for_inference usa __class__.__name__ per tipo check
# Questo e' fragile - dovrebbe usare isinstance o hasattr
if module.__class__.__name__ == "FactorizedLinear":  # Fragile!

# 3. Nessun caching dei tokenizer o del modello tra chiamate

# 4. Il metodo benchmark usa sempre lunghezza 50 token - poco configurabile
```

**Valutazione**: 5.5/10 - Funzionale ma molto basico per un motore di inferenza.

---

### 9. `evaluation.py` - Valutazione

**Punti di Forza:**
- Supporto per task di benchmarking standard (perplessità, zero-shot)
- Sliding window per testi lunghi

**Problemi Identificati:**

```python
# 1. Le metriche zero-shot usano loss come proxy per accuracy
# Questo e' un approccio semplificato che non corrisponde alle metriche ufficiali LM-eval
# Per esempio, evaluazione di boolq usa negative log-likelihood come scoring
# ma le metriche ufficiali richiedono prompt formatting specifico

# 2. Il metodo evaluate_perplexity non gestisce correttamente
# il modello con i nuovi layer FactorizedLinear (usa nn.Module generico)
# Perciò potrebbe mischiare layer quantizzati e non

# 3. I task zero-shot hanno formattazione prompt molto semplificata
# Per un benchmark accurato servirebbe un formato standardizzato (es. few-shot prompt)
```

> **[AGGIUNTO — Benchmark ufficiali usati nel paper]**
> Il paper valuta su sei task zero-shot con lm-evaluation-harness: WinoGrande,
> HellaSwag, BoolQ, ARC-Easy, ARC-Challenge, PIQA. I risultati sono riportati come
> accuracy, non come loss proxy. L'attuale `evaluation.py` non è conforme a questi
> standard. L'integrazione con lm-evaluation-harness è quindi una priorità alta,
> non una feature opzionale, se si vuole confrontabilità con i risultati del paper.

**Valutazione**: 5/10 - Metriche di base ma non conformi agli standard della comunità.

---

### 10. Moduli di Integrazione (v0.2.0)

#### `ternary_init.py`
**Punti di Forza:**
- Implementazione solida della proiezione ternaria ispirata a QMoE
- Gestione fallback sicura

**Problemi:**
```python
# Il parametro sparsity di default (0.9) potrebbe essere troppo aggressivo 
# per modelli non-MoE. QMoE e' specifico per modelli sparsi

# La funzione ternary_project usa torch.quantile con dim che puo' essere lento
# per matrici molto grandi. Considerare un approccio approssimato per grandi scale.
```

#### `group_scale.py`
**Punti di Forza:**
- Integrazione pulita con OxiBonsai
- Calcolo scale di gruppo in forma chiusa

**Problemi:**
```python
# Il calcolo group-by-group in un loop Python e' inefficiente per matrici grandi
# Potrebbe essere vettorializzato con reshape + mean su dim dedicata

# Il GroupScaledWeights ha parametri ridondanti (d_out, d_in, rank 
# sono anche nella forma dei tensori)
```

#### `gguf_export.py`
**Punti di Forza:**
- Implementazione completa del formato GGUF Q1_0_g128
- Validazione del magic header

**Problemi:**
```python
# Il formato scrive manualmente invece di usare codec binari standard
# La scrittura del magic "GGUF" e' codificata come 0x46554747 duplicata
# nell'init ma correttamente usata nei metodi

# Il formato e' compatibile con OxiBonsai ma non verificato/testato
# con un'implementazione reale del backend

# _write_gguf_tensor scrive una dimensione fissa rank=2 per tutti i tensori
# Questo potrebbe non essere corretto per strutture diverse
```

#### `moe_quantization.py`
**Punti di Forza:**
- Riconoscimento layer expert per keyword
- Supporto Hessiano condiviso con tie_hessians

**Problemi:**
```python
# La logica di identificazione layer e' basata solo su string matching
# "experts", "expert_", ecc. Questo non e' sufficiente per tutte le varianti MoE

# Il quantizzatore MoE non ha tra i parametri D_in/D_out per preconditioning,
# quindi la quantizzazione MoE non usa K-FAC per ora

# Il metodo quantize_moe_model crea un'approssimazione W_approx e poi
# la copia su module.weight.data. Questo PERDE tutta la struttura della factorization
# perche' sovrascrive i pesi originali con una approssimazione dense
```

---

## Valutazione della Qualità del Codice

### Stili e Convenzioni

| Aspetto | Stato | Note |
|---------|-------|------|
| **Type Hints** | Parziale | Buono nei moduli core, assente in alcuni moduli nuovi |
| **Docstrings** | Bene | Docstrings dettagliati ma in italiano/inglese misto |
| **Commenti** | Migliorabile | Commenti in italiano nei moduli integrazione, codice standard in inglese |
| **Lunghezza funzioni** | Moderata | Alcune funzioni >100 righe (calibration.py, model_reconstruction.py) |
| **Cicli Complessi** | Presente | Nested loops con molte chiavi di dict |

### Problemi di Stile Italiani/Inglese

Nei moduli di integrazione (`ternary_init.py`, `group_scale.py`, `gguf_export.py`,
`moe_quantization.py`) i commenti sono in italiano mentre tutto il resto del progetto è in
inglese.

**Impatto**: Inconsistenza per una libreria che ha potenziale uso internazionale. È
consigliabile uniformizzare ad un solo linguaggio (preferibilmente inglese).

---

## Analisi delle Performance

### Hotspots Identificati

1. **Calibration - Memory Transfer**:
   - Spostamento dati CPU→MPS/CPU→GPU in fase di accumulo statistici
   - Ogni sample causa un trasferimento nella grafo di backprop

2. **ADMM - Creazione Temporanea Solver** `[CORRETTO]`:
   - Per ogni layer con rank effettivo diverso, creazione nuovo istanza solver
   - **Nota**: il Cholesky in sé NON è un hotspot — è un'ottimizzazione documentata
     nel paper (O(r³/3) vs O(2r³/3) LU). Il problema è l'allocazione/deallocazione
     ripetuta dell'oggetto solver, non il metodo di fattorizzazione scelto.

3. **Kernels - Unpack per Forward** `[CORRETTO]`:
   - L'unpack on-the-fly è un pattern intenzionale del kernel produzione (Appendice E.2).
   - Il vero problema è che l'implementazione Python non sfrutta le ottimizzazioni
     bitwise del kernel CUDA (mask operation + FMA vectorizzato su float16).
   - Riformulazione corretta dell'hotspot: *il kernel Python replica il pattern
     CUDA corretto ma senza le ottimizzazioni di basso livello.*

4. **Block I/O - Calcolo per ogni blocco**:
   - `_get_block_inputs_outputs` richiede un forward pass del modello per ogni blocco
   - Per N blocchi = N forward passes completi

---

## Problemi di Robustezza

### Potenziali Bug

1. **Memory Leak nei Hooks**:
   - Hooks non rimossi se eccezione durante `_get_block_inputs_outputs`
   - Fix: usare `try/finally` per rimozione hook

2. **Numerical Stability in Divisioni**:
   ```python
   s1 = s1 / (D_out_sqrt + 1e-8)
   # Problema: se D_out e' molto piccolo, divisione instabile
   # Meglio: s1 * D_out_sqrt.rsqrt()
   ```

3. **Gestione di Division per Zero**:
   ```python
   eta = torch.sqrt(norm_V / norm_U)
   # Se norm_U e' esattamente zero, restituisce inf/nan
   ```

   > **[AGGIUNTO — Analisi teorica del paper]**
   > La Proposition 1 e il Remark 1 dell'Appendice A del paper dimostrano che il
   > bilanciamento η = √(‖V̂‖_F / ‖Û‖_F) è necessario per stabilità numerica: senza
   > normalizzazione il condition number κ(H) = (λ_max+ρ)/(λ_min+ρ) può divergere
   > sia verso zero (scale vanishing, gradiente dominato dal regularizer) sia verso
   > infinito (scale exploding, Cholesky numericamente instabile). Il check
   > `norm_U > 1e-8` è insufficiente: va gestito il caso con clamp o con una soglia
   > adattativa basata sul valore di ρ corrente.

4. **Assenza di Seed Fissi**:
   - `torch.randn` per init ADMM non usa il seed `config.seed`
   - Risultati non riproducibili tra esecuzioni
   - Il paper usa `random seed value of 0` (Appendice C) — questo va documentato
     e applicato in modo consistente

5. **No Checkpoint/Resume**:
   - Se la quantizzazione fallisce a metà blocco N, bisogna ricominciare dall'inizio
   - Nessun checkpointing intermedio

---

## Suggerimenti di Miglioramento

### Alto Impatto

1. **Vettorializzare il Group Scaling**:
   ```python
   # INVECE DI:
   for g in range(num_groups):
       W_g = W[:, g*gs:(g+1)*gs]
       
   # USARE:
   W_grouped = W.reshape(d_out, num_groups, group_size)
   group_scales = (W_grouped * torch.sign(W_grouped)).mean(dim=2)
   ```

2. **Implementare Checkpoint Intermedio**:
   ```python
   def quantize(self):
       for i, block in enumerate(blocks):
           if self._checkpoint_exists(i):
               continue  # Skip al checkpoint successivo
           # ... processa e salva checkpoint ...
   ```

3. **Kernel Python: allineare al pattern CUDA documentato** `[CORRETTO]`:
   - Rimpiazzare l'unpack Python con operazioni bitwise native (mask + FMA)
   - La raccomandazione precedente "caching dei tensori unpacked" va contro il
     design del paper, che privilegia il risparmio di memoria sull'unpack on-the-fly
   - Il target corretto: implementare in Triton o C++ il pattern documentato
     nell'Appendice E.2 (uint32 packed, lightweight mask, FMA su float16)

4. **Logging Strutturato**:
   - Sostituire i logger.info fitti con un progress_callback configurabile
   - Integrare supporto per wandb/tensorboard per monitoraggio remoto

### Medio Impatto

5. **`bits` come Property Calcolato** `[CORRETTO]`:
   ```python
   @property
   def effective_bits(self):
       # Formula corretta dal paper (eq. F.5 Appendice F):
       # BPW = [r(n+m) + 16(n+m)] / (n*m)
       # Dove n=d_out, m=d_in, r=rank
       # NB: include overhead FP16 delle scale s1 (n valori) e s2 (m valori)
   ```

6. **Parallelizzazione dei Blocchi**:
   - Alcuni blocchi NON dipendenti potrebbero essere processati in parallelo
   - Esplorare `torch.distributed` per modelli molto grandi

7. **Supporto a nuove Architetture** `[CORRETTO]`:
   - Il paper testa già su **cinque famiglie**: Llama-2, Llama-3, Gemma-3, Qwen-3,
     Rnj-1 (da 0.6B a 70B). L'implementazione open-source potrebbe non coprirle
     tutte. La raccomandazione corretta è: estendere la copertura **oltre** le cinque
     famiglie documentate (es. Phi, Mistral, OLMo), non colmare una mancanza di base.

### Basso Impatto

8. **Deduplicare i Parametri Legacy**:
   - Rimuovere `pre_tune_steps`/`post_tune_steps`/`glob_tune_steps` dal config
   - Mantenere solo: `tune_fp_epochs`/`tune_latent_epochs`/`tune_scales_epochs`

9. **Standardizzare le Lingue**:
   - Tutti i commenti in inglese
   - Preferibilmente inglese per ricerca internazionale

10. **Aggiungere Metriche di Qualità Dettagliate**:
    - Errore di ricostruzione per ogni blocco
    - BPW effettivo per layer (formula paper Appendice F)
    - Flip ratio delle variabili latenti pre/post refinement (già monitorato nel paper,
      Appendice D.3: range 0.47%–6.82%)

---

## Considerazioni Architetturali

### Separazione dei Concetti

La pipeline ha 3 fasi ben distinte ma la loro implementazione ha coupling indesiderato:

```
┌───────────────────────────────────────┐
│          Potenziale Separazione        │
├───────────────────────────────────────┤
│ Attuale:                                │
│   NanoQuantizer                         │
│   ├── load_model()                      │
│   ├── quantize()                        │
│   │   ├── Phase 1: calibration          │
│   │   ├── Phase 2: block reconstruction │
│   │   └─- Phase 3: model reconstruction │
│   └── save()                            │
│                                          │
│ Migliore:                               │
│   NanoQuantPipeline                     │
│   ├── calibrator: GlobalCalibration     │
│   ├── block_processor: BlockWorker      │
│   ├── global_tuner: ScaleTuner          │
│   └── checkpoint_manager: Checkpoint    │
└───────────────────────────────────────┘
```

### Dipendenze Preventive `[CORRETTO]`

> Le versioni minime sono aggiornate in base all'Appendice C del paper, che specifica
> l'ambiente di test usato per tutti gli esperimenti.

| Dipendenza | Versione Minima Suggerita | Motivazione |
|-----------|---------------------------|-------------|
| torch | >=2.6.0 | Versione usata nel paper (Appendice C); torch.compile stabile |
| transformers | >=4.51.3 | Versione usata nel paper (non >=4.38 come indicato in precedenza) |
| datasets | >=4.0.0 | Versione usata nel paper |
| lm_eval | >=0.4.9 | Per valutazione zero-shot conforme agli standard del paper |
| einops | >=0.7 | Per operazioni tensoriali leggibili |
| tqdm | >=4.66 | Progress bar per long-running ops |
| wandb | opzionale | Tracking esperimenti |
| pytest | >=7.0 | OK |
| black | >=24 | Per formattazione stabile |

---

## Feature Mancanti o Incompleti

### Incomplete

1. **MoE Quantization**: Il modulo esiste ma la creazione di W_approx dense invece
   della struttura fattorizzata vuol dire che non c'è beneficio di compressione MoE

2. **GGUF Export**: Implementazione del formato esiste ma non collegata al main
   pipeline; dovrebbe essere una target di `save_quantized_model` o un export a
   posteriori

3. **Ternary Init**: Migliora ADMM ma non è sperimentato con benchmark variando lo
   sparsity per diversi modelli

4. **Group Scaling**: Implementazione presente ma non chiamata nel main flow
   (verifica richiesta)

### Da Aggiungere

1. **Adaptive Rank Allocation**: Il paper (sezione 4.6) identifica esplicitamente
   come future work "investigating adaptive rank allocation across layers to further
   optimize the accuracy-per-bit Pareto frontier." Rank uniforme è una semplificazione.

2. **Mixed Precision**: Alcuni layer potrebbero beneficiare di più precisione
   (es. embeddings, LM head)

3. **~~Activation-Aware Quantization~~** `[RIMOSSO — già presente]`:
   Questa feature NON manca. Il paper implementa esattamente questo nel Step 2-1
   (Hessian-Aware Preconditioning) tramite l'approssimazione K-FAC:
   `L(Ŵ) ≈ ‖D̃_out(W − Ŵ)D̃_in‖²_F`
   dove D̃_in e D̃_out sono costruiti da statistiche di attivazione e gradiente.
   Questo è il meccanismo core dell'intera pipeline di inizializzazione.

4. **A/B Testing Framework**: Confronto quantitativo tra diverse configurazioni
   di quantizzazione

5. **Export Multi-Format**: ONNX, safetensors, oltre a GGUF

6. **Integrazione lm-evaluation-harness** `[CORRETTO — priorità alta, non opzionale]`:
   Il paper usa questo framework per tutti i benchmark zero-shot. Non si tratta di
   "Benchmark OpenML" generico ma di una dipendenza specifica necessaria per
   riprodurre i risultati pubblicati (WinoGrande, HellaSwag, BoolQ, ARC-E/C, PIQA).

---

## Conclusioni

NANOQUANT è un progetto con potenziale eccellente, con una base teorica forte
(fattorizzazione binaria a basso rango) e integrazioni con ricerche recenti
(QMoE, OxiBonsai). Il codice è strutturato in modo pulito con i moduli core ben separati.

### Punteggio Complessivo: 6.5/10 `[PUNTEGGI RIVISTI]`

| Categoria | Punteggio Orig. | Punteggio Rivisto | Note |
|-----------|:-:|:-:|------|
| Correttezza Algoritmica | 7/10 | **8/10** | L'Appendice B dimostra Monotonic Descent (Theorem 3) e SPD structure (Lemma 2); garanzie teoriche solide non considerate in precedenza. La discrepanza nella formula forward dei kernel rimane valida. |
| Qualità del Codice | 6/10 | 6/10 | Invariato: incoerenze stilistiche e parametri duplicati confermati. |
| Performance | 5/10 | **6/10** | I kernel CUDA produzione (GEMV/GEMM, Appendice E) esistono e mostrano 3.6×–12.2× speedup rispetto a BF16. Il punteggio basso era basato solo sull'implementazione Python. |
| Robustezza | 5/10 | 5/10 | Invariato: gestione errori basica, assenza checkpoint, seed non fissi. |
| Estensibilità | 6/10 | 6/10 | Invariato: buona struttura modulare ma MoE incompleto. |
| Documentazione | 7/10 | 7/10 | Invariato: README completo ma docstring miste. |

### Raccomandazioni Prioritarie (Revisione Finale)

1. **Unificare il linguaggio dei commenti** (preferibilmente inglese per progetto aperto)

2. **Completare l'integrazione MoE** con K-FAC preconditioning e struttura fattorizzata
   preservata (non sovrascrivere con dense approximation)

3. **Allineare i kernel Python al pattern CUDA documentato** (unpack on-the-fly con
   bitwise mask + FMA, non caching dei pesi decompressi)

4. **Documentare e verificare la shrinkage regularization** (eq. 3 del paper):
   γ=0.2 per Llama/Qwen, γ=0.6 per Gemma/Rnj; critica per stabilità con 128 campioni

5. **Integrare lm-evaluation-harness** per benchmark zero-shot conformi al paper
   (WinoGrande, HellaSwag, BoolQ, ARC-E/C, PIQA)

6. **Implementare checkpoint intermedio e resilienza per esecuzioni lunghe su GPU**

7. **Fissare il seed a 0** (come nel paper, Appendice C) per riproducibilità

8. **Correggere la formula forward in `kernels.py`** per allineamento con eq. 1 del paper
   e gestione efficiente delle scale senza operazioni scalari intermedie
