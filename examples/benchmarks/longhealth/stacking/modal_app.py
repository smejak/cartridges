"""Modal orchestration for the per-patient cartridge stacking experiment.

Phases:
  1. Data preparation (runs inside GPU functions to avoid separate step)
  2. Per-patient training (1 GPU each, patient_01 first as early-stop gate)
  3. Stacked evaluation with attention capture (canonical orderings)
  4. Permutation evaluation (320 tasks, batched at max 10 GPUs)
  5. Figure generation (local)

Usage:
    modal run modal_app.py          # Run the full experiment
    modal run modal_app.py --phase 2  # Run only Phase 2
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

REPO_ROOT = Path(__file__).resolve().parents[4]  # cartridges repo root

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
)

# Mount the repo code so it's available inside the container
code_mount = modal.Mount.from_local_dir(
    str(REPO_ROOT),
    remote_path="/root/cartridges_repo",
    condition=lambda path: (
        not path.startswith(".")
        and "__pycache__" not in path
        and ".egg-info" not in path
        and "node_modules" not in path
        and "viz" not in path
    ),
)

# Secrets for wandb and HF
secrets = [modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("hf-secret")]

# Shared constants (mirrored from config.py to avoid import issues)
PATIENT_IDS = [f"patient_{i:02d}" for i in range(1, 6)]
NUM_TOKENS = 512
EARLY_STOP_ACCURACY = 0.80

# ---------------------------------------------------------------------------
# Helper: set up the Python path inside the container
# ---------------------------------------------------------------------------

def _setup_env():
    """Set up sys.path and env vars inside a Modal container."""
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

@app.function(
    image=image,
    gpu="H100",
    timeout=7200,
    volumes={"/data": volume, "/root/.cache/huggingface": hf_cache},
    mounts=[code_mount],
    secrets=secrets,
    memory=32768,
)
def train_patient(patient_idx: int) -> dict:
    """Train one patient's cartridge. Returns accuracy dict."""
    _setup_env()

    from prepare_data import export_patient_text_files, filter_and_write_per_patient_parquets
    from config import patient_cache_path, patient_data_path, patient_text_path

    patient_id = f"patient_{patient_idx:02d}"

    # Phase 1: Prepare data for this patient (idempotent)
    if not os.path.exists(patient_data_path(patient_id)):
        filter_and_write_per_patient_parquets()
    if not os.path.exists(patient_text_path(patient_id)):
        export_patient_text_files()

    volume.commit()

    # Phase 2: Train
    from per_patient_train import make_config
    config = make_config(patient_id)
    config.run()

    volume.commit()

    # Read final eval results from wandb or compute accuracy from the results
    # The training script logs to wandb, but we also return accuracy directly
    import glob
    import pandas as pd

    results_pattern = os.path.join(config.run_dir, "**", "*.json")
    result_files = glob.glob(results_pattern, recursive=True)

    # Parse accuracy from the training logs if available
    accuracy = None
    try:
        import wandb
        api = wandb.Api()
        runs = api.runs(
            f"jakub-smekal/cartridges",
            filters={"tags": {"$in": [patient_id]}, "state": "finished"},
            order="-created_at",
        )
        if runs:
            run = runs[0]
            for key in run.summary.keys():
                if "score" in key and patient_id in key:
                    accuracy = run.summary[key]
                    break
    except Exception:
        pass

    # Fallback: run a quick eval
    if accuracy is None:
        accuracy = _quick_eval(patient_id)

    return {
        "patient_id": patient_id,
        "patient_idx": patient_idx,
        "accuracy": accuracy,
        "cache_path": patient_cache_path(patient_id),
    }


def _quick_eval(patient_id: str) -> float:
    """Run a quick eval of the trained cartridge on its patient's questions."""
    import torch
    from transformers import AutoTokenizer
    from cartridges.cache import TrainableCache
    from cartridges.data.longhealth.evals import LongHealthMultipleChoiceGenerateDataset
    from cartridges.generation import flex_generate
    from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
    from config import (
        GENERATE_MAX_NEW_TOKENS,
        MODEL_NAME,
        TEMPERATURE,
        patient_cache_path,
    )

    device = "cuda"
    model = FlexQwen3ForCausalLM.from_pretrained(MODEL_NAME).to(device).to(torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model.eval()

    cache_path = patient_cache_path(patient_id)
    cache = TrainableCache.from_pretrained(cache_path, device=device).to(torch.bfloat16)

    dataset = LongHealthMultipleChoiceGenerateDataset(
        config=LongHealthMultipleChoiceGenerateDataset.Config(patient_ids=[patient_id]),
        tokenizer=tokenizer,
        seed=42,
    )

    correct = 0
    for i in range(len(dataset)):
        element = dataset[i]
        input_ids = element.input_ids[0].to(device)
        seq_ids = torch.zeros(input_ids.shape[0], dtype=torch.long, device=device)
        position_ids = torch.arange(input_ids.shape[0], device=device)

        with torch.no_grad():
            pred_ids = flex_generate(
                model=model, tokenizer=tokenizer,
                input_ids=input_ids, seq_ids=seq_ids, position_ids=position_ids,
                cache=cache, max_new_tokens=GENERATE_MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
            )

        pred_text = tokenizer.decode(pred_ids.get(0, []), skip_special_tokens=True)
        is_correct, _ = dataset.score(pred=pred_text, answer=element.answer, convo_id=element.convo_id)
        correct += int(is_correct)

    del model, cache
    torch.cuda.empty_cache()
    return correct / len(dataset) if len(dataset) > 0 else 0


# ---------------------------------------------------------------------------
# Phase 4a: Stacked evaluation with attention capture
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
    volumes={"/data": volume, "/root/.cache/huggingface": hf_cache},
    mounts=[code_mount],
    secrets=secrets,
    memory=32768,
)
def evaluate_stack(patient_ids: list[str], capture_attention: bool = True) -> dict:
    """Evaluate a single stacked configuration."""
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
# Phase 4b: Permutation evaluation batch
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="H100",
    timeout=10800,
    volumes={"/data": volume, "/root/.cache/huggingface": hf_cache},
    mounts=[code_mount],
    secrets=secrets,
    memory=32768,
)
def evaluate_permutation_batch(tasks: list[dict]) -> list[dict]:
    """Evaluate a batch of permutation tasks (model loaded once)."""
    _setup_env()
    volume.reload()

    from permutation_eval import run_worker
    results = run_worker(tasks, capture_attention=False, log_to_wandb=True)
    return results


# ---------------------------------------------------------------------------
# Local entrypoint: orchestrate everything
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(phase: int = 0):
    """
    Run the full experiment or a specific phase.
    phase=0 (default): run all phases
    phase=2: training only
    phase=4: evaluation only (requires trained caches)
    """
    import time

    if phase in (0, 2):
        print("=" * 60)
        print("PHASE 2a: Training patient_01 (early stop gate)")
        print("=" * 60)

        result = train_patient.remote(1)
        acc = result["accuracy"]
        print(f"patient_01 accuracy: {acc:.2%}")

        if acc < EARLY_STOP_ACCURACY:
            print(f"GATE FAILED: {acc:.2%} < {EARLY_STOP_ACCURACY:.0%}")
            print("Increase NUM_TOKENS (set CARTRIDGES_NUM_TOKENS env var) and retry.")
            return

        print("Gate passed! Training patients 2-5 in parallel...")
        handles = [train_patient.spawn(i) for i in range(2, 6)]
        for h in handles:
            r = h.get()
            print(f"  {r['patient_id']}: accuracy={r['accuracy']:.2%}")

    if phase in (0, 4):
        print()
        print("=" * 60)
        print("PHASE 4a: Canonical stacked evaluations")
        print("=" * 60)

        # Evaluate canonical orderings k=2..5 (with attention capture)
        canonical_handles = []
        for k in range(2, len(PATIENT_IDS) + 1):
            pids = PATIENT_IDS[:k]
            print(f"  Submitting k={k}: {pids}")
            canonical_handles.append(evaluate_stack.spawn(pids, capture_attention=True))

        for h in canonical_handles:
            r = h.get()
            print(f"  {r['ordering']}: overall={r['overall_accuracy']:.2%}")

        # Also evaluate each single patient (for the figures, k=1 data)
        single_handles = []
        for pid in PATIENT_IDS:
            single_handles.append(evaluate_stack.spawn([pid], capture_attention=False))
        for h in single_handles:
            r = h.get()
            print(f"  {r['ordering']}: {r['overall_accuracy']:.2%}")

        print()
        print("=" * 60)
        print("PHASE 4b: Permutation evaluations (320 tasks)")
        print("=" * 60)

        # Generate all permutation tasks
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

        print(f"Total permutation tasks: {len(all_tasks)}")

        # Distribute across 10 workers (max 10 concurrent GPUs)
        num_workers = 10
        worker_batches = [[] for _ in range(num_workers)]
        for i, task in enumerate(all_tasks):
            worker_batches[i % num_workers].append(task)

        all_results = []
        for batch_start in range(0, num_workers, 10):
            batch_end = min(batch_start + 10, num_workers)
            batch_handles = []
            for w in range(batch_start, batch_end):
                if worker_batches[w]:
                    batch_handles.append(
                        evaluate_permutation_batch.spawn(worker_batches[w])
                    )

            for h in batch_handles:
                results = h.get()
                all_results.extend(results)
                print(f"  Completed batch: {len(results)} evaluations")

        # Save aggregated results locally
        out_path = "permutation_results_all.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved {len(all_results)} results to {out_path}")

    if phase in (0, 5):
        print()
        print("=" * 60)
        print("PHASE 5: Generating figures (local)")
        print("=" * 60)
        print("Run: python plot_results.py")
