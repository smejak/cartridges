"""Modal orchestration for the per-patient cartridge stacking experiment.

Uses a remote orchestrator function so `modal run --detach` works correctly.

Usage:
    modal run --detach modal_app.py
    modal run --detach modal_app.py --phase 2   # training only
    modal run --detach modal_app.py --phase 4   # eval only (needs trained caches)
"""
import json
import os
import sys
from itertools import permutations
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal configuration
# ---------------------------------------------------------------------------

if modal.is_local():
    REPO_ROOT = Path(__file__).resolve().parents[4]
else:
    REPO_ROOT = Path("/root/cartridges_repo")

app = modal.App("cartridge-stacking")

volume = modal.Volume.from_name("cartridge-stacking-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch",
        "transformers>=4.49.0,<=4.55",
        "wandb",
        "pydrantic",
        "datasets",
        "numpy",
        "einops",
        "tqdm",
        "peft",
        "evaluate",
        "matplotlib",
        "openai",
        "tiktoken",
        "markdown",
        "pandas",
        "pyarrow",
    )
    .env({
        "CARTRIDGES_OUTPUT_DIR": "/data",
        "CARTRIDGES_DIR": "/root/cartridges_repo",
    })
    .add_local_dir(
        str(REPO_ROOT),
        remote_path="/root/cartridges_repo",
        ignore=[
            "**/__pycache__",
            "**/.git",
            "**/node_modules",
            "**/viz",
            "**/*.egg-info",
            "**/.mypy_cache",
        ],
    )
)

secrets = [modal.Secret.from_name("jakub-api-keys")]

PATIENT_IDS = [f"patient_{i:02d}" for i in range(1, 6)]
EARLY_STOP_ACCURACY = 0.80

COMMON_KWARGS = dict(
    image=image,
    volumes={"/data": volume, "/root/.cache/huggingface": hf_cache},
    secrets=secrets,
    memory=32768,
)


def _setup_env():
    repo = "/root/cartridges_repo"
    stacking = os.path.join(repo, "examples/benchmarks/longhealth/stacking")
    for p in [repo, stacking]:
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("CARTRIDGES_DIR", repo)
    os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/data")


# ---------------------------------------------------------------------------
# Phase 1+2: Train a single patient's cartridge
# ---------------------------------------------------------------------------

@app.function(gpu="H100", timeout=7200, **COMMON_KWARGS)
def train_patient(patient_idx: int) -> dict:
    _setup_env()

    from prepare_data import export_patient_text_files, filter_and_write_per_patient_parquets
    from config import patient_cache_path, patient_data_path, patient_text_path, patient_cache_dir

    patient_id = f"patient_{patient_idx:02d}"

    # Phase 1: Prepare data (idempotent)
    if not os.path.exists(patient_data_path(patient_id)):
        filter_and_write_per_patient_parquets()
    if not os.path.exists(patient_text_path(patient_id)):
        export_patient_text_files()
    volume.commit()

    # Phase 2: Train
    from per_patient_train import make_config
    config = make_config(patient_id)
    # Save cache directly to patient_cache_dir (no extra subdir) so
    # patient_cache_path() points to the actual saved file.
    config.run_dir = patient_cache_dir(patient_id)
    os.makedirs(config.run_dir, exist_ok=True)
    config.run()
    volume.commit()

    # Get accuracy — the training run logs eval to wandb, but also return it
    accuracy = _quick_eval(patient_id)
    print(f"[train_patient] {patient_id} final accuracy: {accuracy:.2%}")
    return {
        "patient_id": patient_id,
        "patient_idx": patient_idx,
        "accuracy": accuracy,
        "cache_path": patient_cache_path(patient_id),
    }


def _quick_eval(patient_id: str) -> float:
    import torch
    from transformers import AutoTokenizer
    from cartridges.cache import TrainableCache
    from cartridges.data.longhealth.evals import LongHealthMultipleChoiceGenerateDataset
    from cartridges.generation import flex_generate
    from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
    from config import GENERATE_MAX_NEW_TOKENS, MODEL_NAME, TEMPERATURE, patient_cache_path

    device = "cuda"
    model = FlexQwen3ForCausalLM.from_pretrained(MODEL_NAME).to(device).to(torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model.eval()

    cache = TrainableCache.from_pretrained(patient_cache_path(patient_id), device=device).to(device).to(torch.bfloat16)
    dataset = LongHealthMultipleChoiceGenerateDataset(
        config=LongHealthMultipleChoiceGenerateDataset.Config(patient_ids=[patient_id]),
        tokenizer=tokenizer, seed=42,
    )

    correct = 0
    for i in range(len(dataset)):
        elem = dataset[i]
        input_ids = elem.input_ids[0].to(device)
        seq_ids = torch.zeros(input_ids.shape[0], dtype=torch.long, device=device)
        pos_ids = torch.arange(input_ids.shape[0], device=device)
        with torch.no_grad():
            pred_ids = flex_generate(
                model=model, tokenizer=tokenizer, input_ids=input_ids,
                seq_ids=seq_ids, position_ids=pos_ids, cache=cache,
                max_new_tokens=GENERATE_MAX_NEW_TOKENS, temperature=TEMPERATURE,
            )
        pred = tokenizer.decode(pred_ids.get(0, []), skip_special_tokens=True)
        is_correct, _ = dataset.score(pred=pred, answer=elem.answer, convo_id=elem.convo_id)
        correct += int(is_correct)

    del model, cache
    torch.cuda.empty_cache()
    return correct / len(dataset) if len(dataset) > 0 else 0


# ---------------------------------------------------------------------------
# Phase 4a: Stacked evaluation
# ---------------------------------------------------------------------------

@app.function(gpu="H100", timeout=3600, **COMMON_KWARGS)
def evaluate_stack(patient_ids: list[str], capture_attention: bool = True) -> dict:
    _setup_env()
    volume.reload()

    from evaluate_stacked import evaluate_stacked
    df = evaluate_stacked(
        patient_ids=patient_ids,
        capture_attention=capture_attention,
        log_to_wandb=True,
    )
    overall_acc = df["is_correct"].mean()
    per_patient = df.groupby("question_patient")["is_correct"].mean().to_dict()
    return {
        "patient_ids": patient_ids,
        "ordering": "_".join(patient_ids),
        "overall_accuracy": float(overall_acc),
        "per_patient_accuracy": {k: float(v) for k, v in per_patient.items()},
    }


# ---------------------------------------------------------------------------
# Phase 4a-rope: Stacked evaluation with RoPE adjustment
# ---------------------------------------------------------------------------

@app.function(gpu="H100", timeout=7200, **COMMON_KWARGS)
def evaluate_stack_rope(patient_ids: list[str], capture_attention: bool = True) -> dict:
    """Evaluate a stacked cartridge with RoPE-adjusted position re-indexing.

    Same as evaluate_stack but uses stack_caches_rope_adjusted so that each
    cache's keys get sequential (non-overlapping) position encodings.
    """
    _setup_env()
    volume.reload()

    from evaluate_stacked import evaluate_stacked
    df = evaluate_stacked(
        patient_ids=patient_ids,
        capture_attention=capture_attention,
        log_to_wandb=True,
        rope_adjust=True,
    )
    overall_acc = df["is_correct"].mean()
    per_patient = df.groupby("question_patient")["is_correct"].mean().to_dict()
    return {
        "patient_ids": patient_ids,
        "ordering": "_".join(patient_ids),
        "rope_adjust": True,
        "overall_accuracy": float(overall_acc),
        "per_patient_accuracy": {k: float(v) for k, v in per_patient.items()},
    }


# ---------------------------------------------------------------------------
# Phase 4b: Permutation evaluation batch
# ---------------------------------------------------------------------------

@app.function(gpu="H100", timeout=10800, **COMMON_KWARGS)
def evaluate_permutation_batch(tasks: list[dict]) -> list[dict]:
    _setup_env()
    volume.reload()
    from permutation_eval import run_worker
    return run_worker(tasks, capture_attention=False, log_to_wandb=True)


# ---------------------------------------------------------------------------
# Remote orchestrator — this is what --detach keeps alive
# ---------------------------------------------------------------------------

@app.function(gpu="H100", timeout=600, **COMMON_KWARGS)
def fixup_patient01_cache() -> bool:
    """One-time fix: patient_01's cache was saved to a subdirectory.
    Copy it to the expected location so evaluate_stacked can find it."""
    _setup_env()
    volume.reload()
    from config import patient_cache_path, patient_cache_dir
    import shutil

    expected = patient_cache_path("patient_01")
    old_path = os.path.join(
        patient_cache_dir("patient_01"),
        "stacking_patient_01_toks512",
        "cache_last.pt",
    )

    if os.path.exists(expected):
        print(f"  patient_01 cache already at {expected}")
        return True

    if os.path.exists(old_path):
        # old_path is a symlink to cache-stepNNN.pt; resolve and copy
        real_old = os.path.realpath(old_path)
        print(f"  Copying {real_old} → {expected}")
        shutil.copy2(real_old, expected)
        volume.commit()
        return True

    # Try downloading from wandb as last resort
    try:
        import wandb
        print("  Downloading patient_01 cache from wandb run 9ovgs3z9...")
        os.makedirs(patient_cache_dir("patient_01"), exist_ok=True)
        out = wandb.restore(
            "cache-step293.pt",
            run_path="jakub-smekal/cartridges/9ovgs3z9",
            root=patient_cache_dir("patient_01"),
        )
        shutil.copy2(out.name, expected)
        volume.commit()
        return True
    except Exception as e:
        print(f"  Failed to download from wandb: {e}")
        return False


@app.function(timeout=86400, **COMMON_KWARGS)  # no GPU, 24h timeout
def orchestrate(phase: int = 0, patient_idxs: list[int] = None):
    """Run the full experiment from a remote container (detach-safe).

    Args:
        phase: 0=all, 2=training only, 4=eval only
        patient_idxs: which patients to train (default: all 5)
    """
    print("=" * 60)
    print("CARTRIDGE STACKING EXPERIMENT — ORCHESTRATOR")
    print("=" * 60)

    if patient_idxs is None:
        patient_idxs = list(range(1, 6))

    if phase in (0, 2):
        print(f"\n[Phase 2] Training patients {patient_idxs} in parallel...")
        print(f"  GPUs requested: {len(patient_idxs)}")
        handles = [train_patient.spawn(i) for i in patient_idxs]
        for h in handles:
            r = h.get()
            print(f"  {r['patient_id']}: accuracy={r['accuracy']:.2%}")

    if phase in (0, 4):
        # --- Fixup: ensure patient_01 cache is at the expected path ---
        print("\n[Fixup] Checking patient_01 cache path...")
        fixup_patient01_cache.remote()

        # --- Phase 4a: Canonical stacked evaluations ---
        # Single-patient evals (5 GPUs) + canonical stacks (4 GPUs) = 9 GPUs
        print("\n[Phase 4a] Canonical stacked evaluations (9 GPUs)...")
        all_handles = []

        # Single-patient evals (for k=1 data in figures)
        for pid in PATIENT_IDS:
            print(f"  Submitting single: [{pid}]")
            all_handles.append(("single", pid, evaluate_stack.spawn([pid], capture_attention=False)))

        # Canonical stack evals (k=2..5)
        for k in range(2, len(PATIENT_IDS) + 1):
            pids = PATIENT_IDS[:k]
            print(f"  Submitting k={k}: {pids}")
            all_handles.append(("canonical", k, evaluate_stack.spawn(pids, capture_attention=True)))

        # Wait for all Phase 4a to complete before starting 4b
        for label, key, h in all_handles:
            r = h.get()
            print(f"  [{label} {key}] {r['ordering']}: overall={r['overall_accuracy']:.2%}")

        # --- Phase 4b: Permutation evaluations ---
        print(f"\n[Phase 4b] Permutation evaluations...")
        all_tasks = []
        for k in [2, 3, 4, 5]:
            for perm in permutations(range(len(PATIENT_IDS)), k):
                pids = [PATIENT_IDS[i] for i in perm]
                all_tasks.append({
                    "stack_size": k,
                    "perm_indices": list(perm),
                    "patient_ids": pids,
                    "ordering_str": "_".join(pids),
                })
        print(f"  Total permutation tasks: {len(all_tasks)}")

        # Distribute across 10 workers
        num_workers = 10
        worker_batches = [[] for _ in range(num_workers)]
        for i, task in enumerate(all_tasks):
            worker_batches[i % num_workers].append(task)

        all_results = []
        batch_handles = []
        for w in range(num_workers):
            if worker_batches[w]:
                batch_handles.append(evaluate_permutation_batch.spawn(worker_batches[w]))

        for h in batch_handles:
            results = h.get()
            all_results.extend(results)
            print(f"  Completed batch: {len(results)} evaluations")

        # Save to volume
        results_dir = "/data/stacking/results"
        os.makedirs(results_dir, exist_ok=True)
        out_path = os.path.join(results_dir, "permutation_results_all.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        volume.commit()
        print(f"  Saved {len(all_results)} results to {out_path}")

    if phase in (0, 5):
        # --- Phase 5: RoPE-adjusted canonical stacked evaluations ---
        print("\n[Fixup] Checking patient_01 cache path...")
        fixup_patient01_cache.remote()

        print("\n[Phase 5] RoPE-adjusted stacked evaluations (4 GPUs)...")
        rope_handles = []
        for k in range(2, len(PATIENT_IDS) + 1):
            pids = PATIENT_IDS[:k]
            print(f"  Submitting rope k={k}: {pids}")
            rope_handles.append(("rope", k, evaluate_stack_rope.spawn(pids, capture_attention=False)))

        for label, key, h in rope_handles:
            r = h.get()
            print(f"  [{label} k={key}] {r['ordering']}: overall={r['overall_accuracy']:.2%}")

    if phase == 6:
        # --- Retry: just k=5 RoPE-adjusted (timed out at 3600s) ---
        print("\n[Fixup] Checking patient_01 cache path...")
        fixup_patient01_cache.remote()

        pids = PATIENT_IDS[:5]
        print(f"\n[Phase 6] Retrying rope k=5: {pids}")
        r = evaluate_stack_rope.remote(pids, capture_attention=False)
        print(f"  [rope k=5] {r['ordering']}: overall={r['overall_accuracy']:.2%}")

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    return {"status": "complete"}


# ---------------------------------------------------------------------------
# Local entrypoint — just kicks off the remote orchestrator
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(phase: int = 0, patients: str = ""):
    """Kick off the remote orchestrator (safe to --detach).

    Args:
        phase: 0=all, 2=training only, 4=eval only
        patients: comma-separated patient indices, e.g. "2,3,4,5". Empty = all 5.
    """
    patient_idxs = [int(x) for x in patients.split(",") if x.strip()] or None
    result = orchestrate.remote(phase, patient_idxs=patient_idxs)
    print(f"Final result: {result}")
