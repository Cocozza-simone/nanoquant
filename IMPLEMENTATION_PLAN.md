# Piano di Implementazione: Miglioramenti NANOQUANT (Punti 2-5)

## Contesto
Il punto 1 (Welford su `calibration.py`) è già completato e i test esistenti passano. Questo piano copre i rimanenti 4 miglioramenti strutturali identificati nell'analisi iniziale.

---

## Punto 2: Rimozione `deepcopy(self.model)` in `quantization.py`

**Problema**: `_model_reconstruction` crea `copy.deepcopy(self.model)` per ottenere un modello di riferimento per il KL divergence (Phase 3). Questo duplica l'allocazione di memoria (~14GB per un 7B FP16), portando il picco da ~14GB a ~28GB.

**Soluzione**: Invece di fare deep copy dell'intero modello, calcolare i logits di riferimento dal modello originale batch per batch durante l'ottimizzazione dei soli parametri scala, senza mai tenere due modelli in memoria.

**Cambio specifico in `quantization.py` (`_model_reconstruction`, riga ~248-268)**:
- Rimuovere `ref_model = copy.deepcopy(self.model)`
- Sostituire con: prima della creazione di `ModelReconstruction`, creare un secondo modello caricato da zero in stile `AutoModelForCausalLM.from_pretrained(..., device_map="auto", torch_dtype=torch.float16)` o, meglio ancora, ristrutturare `ModelReconstruction.reconstruct()` per accettare un callable `get_orig_logits()` che emetta i logits on-demand dal modello principale tenuto in memoria dopo aver disattivato i gradienti sui parametri fattorizzati.
- Esplorare l'aggiunta di un flag `reference_logits_only: bool` in modo da non materializzare mai il secondo modello.

**Rischio**: `_model_reconstruction` dipende da `reconstructor.reconstruct()` che richiede due modelli separati. Bisogna verificare l'interfaccia esatta di `ModelReconstruction` in `model_reconstruction.py` prima di procedere.

---

## Punto 3: Cleanup Esplicito tra i Blocchi (PRIORITA: ALTA - Iniziare da qui)

**Problema**: I tensori `X` e `Y_star` restano in memoria dopo l'elaborazione di ogni blocco. Su un modello con 32 layer questo accumula 4-8GB di leak.

**Soluzione in `quantization.py` (`_block_reconstruction` ciclo for)**:
- Dopo `pipeline.reconstruct_block(...)`, aggiungere:
  ```python
  del X, Y_star
  gc.collect()
  if torch.cuda.is_available():
      torch.cuda.empty_cache()
  elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
      torch.mps.empty_cache()
  ```
- Questo è il cambio più sicuro e immediato.

---

## Punto 4: `@torch.compile` su `kernels.py`

**Problema**: `OptimizedFactorizedLinear.forward()` in `kernels.py` può essere compilato con PyTorch 2.0+ ma non lo è.

**Soluzione in `kernels.py`**:
- Aggiungere `@torch.compile(mode="reduce-overhead")` sopra al metodo `forward` di `OptimizedFactorizedLinear`
- Verificare che la classe funzioni ancora come `nn.Module` quando la funzione compilata è un metodo
- Testare che non ci siano regressioni di performance nella prima chiamata (overhead di compilazione)
- Oppure applicare su `binary_gemv_simple` se `OptimizedFactorizedLinear.forward()` contiene logica condizionale che confonde il compiler

---

## Punto 5: Bit-packing Nativo PyTorch in `packing.py`

**Problema**: `pack_binary_tensor()` usa il percorso NumPy (`bits.cpu().numpy()`, `np.packbits()`) che forza sincronizzazione CPU-GPU per ogni layer (~hundreds di sincronizzazioni per un modello intero).

**Soluzione in `packing.py`**:
- Implementare `torch_packbits` e `torch_unpackbits` usando `torch.bitwise_or` e shift nativi
- Mantenere il percorso NumPy come fallback per compatibilità
- Assicurare che l'output sia byte-identico alla versione NumPy per non rompere i test
- La funzione deve gestire device "cpu", "cuda" e "mps"

---

## Ordine di Esecuzione Consigliato
1. **Punto 3** (cleanup, sicuro, immediato)
2. **Punto 5** (bit-packing, buono per calibrazione)
3. **Punto 2** (deepcopy, cambio più impattante – verificare prima interfaccia `ModelReconstruction`)
4. **Punto 4** (`@torch.compile`, riabilitativo, testare a fondo)

## Verifica
- Eseguire tutti i test in `tests/test_nanoquant.py`
- Verificare import dopo ogni modifica con `python -c "import src.nanoquant..."`
