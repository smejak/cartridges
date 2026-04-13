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

### Changes for next run
- Tokens: 512 → 2048 (committed in `ff02a30`)
- Epochs: 2 → 1 (to reduce overfitting)
- Eval/save every 32 steps (finer granularity)
- Keep last 20 checkpoints
- Fixed cache path mismatch
