# Stacking Experiment Run Log

## Run 1: patient_01, 512 tokens (2026-04-10)

**Goal**: Train a single-patient cartridge on patient_01 and evaluate accuracy. Gate: abort if < 80%.

### Config
- Model: Qwen/Qwen3-4b (`FlexQwen3ForCausalLM`)
- KV cache tokens: 512 (initialized from patient_01 medical records via `KVFromText`)
- Training data: 12,672 filtered conversations from HF (`hazyresearch/m07d11_longhealth_synthesize_qwen3-4b_p10_n65536-{0,1}`)
- lr: 2e-2, epochs: 2, global_batch_size: 32, packed_seq_length: 2048
- Eval: 20 multiple-choice questions for patient_01, temperature=0.3
- Eval every 64 optimizer steps, save every 256 steps

### Results

Best run: `9ovgs3z9` (40 min, 293 optimizer steps across 2 epochs)

| Step | Epoch | Train Loss | Eval Accuracy |
|------|-------|-----------|---------------|
| 0    | 1     | 0.5640    | 30%           |
| 64   | 1     | 0.1790    | **55%** (peak)|
| 128  | 1     | 0.1517    | 50%           |
| 192  | 2     | 0.1405    | 40%           |
| 256  | 2     | 0.1267    | 55%           |
| 293  | 2     | 0.1259    | 40% (final)   |

Two other runs with same config showed similar patterns (peak 40-50%, final 40-45%).

### Observations
- **Gate FAILED**: 40-55% accuracy, well below 80% threshold
- Severe overfitting: train loss drops steadily (0.56 → 0.13) but eval accuracy peaks early (step 64) then oscillates/declines
- Epoch 2 did not improve over epoch 1
- Untrained cache baseline (step 0): 30% — above random (20% for 5-choice MC), meaning `KVFromText` initialization provides some value even before training

### Bug encountered
- `_quick_eval()` crashed with `FileNotFoundError: /data/stacking/caches/patient_01/cache_last.pt`
- Cause: `run_dir` was set to `.../patient_01/stacking_patient_01_toks512/` (subdirectory) but `patient_cache_path()` expected `.../patient_01/cache_last.pt`
- Fix: committed in `ff02a30` — set `run_dir = patient_cache_dir(patient_id)` directly
- The orchestrator never reached Phase 4 (stacking evaluation) due to this crash

### Where to find results

| Artifact | Location |
|----------|----------|
| Wandb runs | [jakub-smekal/cartridges](https://wandb.ai/jakub-smekal/cartridges) — runs `9ovgs3z9`, `vj38t50u`, `r7xoqcud` |
| Per-question eval tables | Wandb run `9ovgs3z9` → `generate_longhealth_patient_01/table` |
| Cache checkpoints | Modal volume `cartridge-stacking-data` at `/data/stacking/caches/patient_01/stacking_patient_01_toks512/cache-step{256,293}.pt` |
| Cache on wandb | Run `9ovgs3z9` → Files → `cache-step256.pt`, `cache-step293.pt` |
| Filtered training data | Modal volume at `/data/stacking/data/patient_01.parquet` (12,672 conversations) |
| Patient text file | Modal volume at `/data/stacking/data/patient_01_text.txt` (44,877 chars) |

### Files used
- `examples/benchmarks/longhealth/stacking/modal_app.py` — orchestration
- `examples/benchmarks/longhealth/stacking/config.py` — constants
- `examples/benchmarks/longhealth/stacking/prepare_data.py` — HF data filtering + text export
- `examples/benchmarks/longhealth/stacking/per_patient_train.py` — training config
- `cartridges/train.py` — training loop (existing, unmodified)
- `cartridges/cache.py` — `TrainableCache` (added `stack_caches()`, not yet exercised)

---

## Run 2: patients 2-5, 512 tokens (2026-04-13)

**Goal**: Train per-patient cartridges for the remaining 4 patients using the same config as patient_01.

### Config
Same as Run 1: Qwen/Qwen3-4b, 512 KV tokens, lr=2e-2, 2 epochs, global_batch_size=32.
4 H100 GPUs in parallel (within the 10-GPU limit).

### Training data per patient
| Patient | Conversations | Text (chars) |
|---------|--------------|-------------|
| patient_02 | 13,728 | 50,301 |
| patient_03 | 12,160 | 47,900 |
| patient_04 | 13,568 | 45,885 |
| patient_05 | 12,960 | 37,677 |

### Eval trajectory (all patients, 20 MC questions each)

| Patient | Step 0 | Step 64 | Step 128 | Step 192 | Step 256 | Final |
|---------|--------|---------|----------|----------|----------|-------|
| patient_02 | 25% | 50% | **60%** | 55% | 60% | 55% (step 302) |
| patient_03 | 20% | 30% | 35% | 35% | 35% | 30% (step 268) |
| patient_04 | 5% | **65%** | 55% | — | — | 50% (wandb) |
| patient_05 | 15% | 30% | 15% | 40% | 25% | 35% (step 276) |

### All patients summary (including patient_01 from Run 1)

| Patient | Baseline | Peak Accuracy | Peak Step | Final | Wandb Run |
|---------|----------|--------------|-----------|-------|-----------|
| patient_01 | 30% | **55%** | 64 | 40% | `9ovgs3z9` |
| patient_02 | 25% | **60%** | 128 | 55% | `pfw7r934` |
| patient_03 | 20% | **35%** | 128 | 30% | `mfzlrpml` |
| patient_04 | 5% | **65%** | 64 | 50% | `c3s5lg8j` |
| patient_05 | 15% | **40%** | 192 | 35% | `ylqbf4rv` |

### Observations
- All 5 patients show the same pattern: accuracy peaks in epoch 1 (step 64-128) then oscillates/declines in epoch 2
- Patients with lowest baselines showed the biggest training improvement (patient_04: 5% → 65%)
- High eval variance (patient_05 fluctuated 15-40%) likely due to small eval set (20 questions)
- Average peak accuracy across all 5 patients: **51%**
- Average final accuracy: **42%**
- No patient reached the 80% gate — this is a property of the 512-token compression, not a bug

### Bug encountered
- `_quick_eval()` crashed with `RuntimeError: Expected all tensors to be on the same device` (CPU vs CUDA)
- Cause: `TrainableCache.from_pretrained()` creates `_seq_ids` buffer on CPU; `.to(torch.bfloat16)` changes dtype but doesn't move device
- Fix: added `.to(device)` before `.to(torch.bfloat16)` — committed in `67e08af`
- Non-blocking: all training completed and caches saved before the crash

### Where to find results

| Artifact | Location |
|----------|----------|
| Wandb runs | `pfw7r934` (p02), `mfzlrpml` (p03), `c3s5lg8j` (p04), `ylqbf4rv` (p05) |
| Per-question eval tables | Each wandb run → `generate_longhealth_patient_XX/table` |
| Cache checkpoints | Modal volume `cartridge-stacking-data` at `/data/stacking/caches/patient_XX/cache_last.pt` |
| Cache on wandb | Each run → Files → `cache-step*.pt` |
| Filtered training data | Modal volume at `/data/stacking/data/patient_XX.parquet` |
| Patient text files | Modal volume at `/data/stacking/data/patient_XX_text.txt` |
