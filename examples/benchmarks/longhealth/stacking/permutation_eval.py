"""Phase 4b: Evaluate all permutations of patient stacks.

Generates all P(5,k) orderings for k in {2,3,4,5}, assigns them to workers,
and runs evaluations. Each worker loads the model once and iterates over its
assigned tasks.

Usage:
    # Run all permutations sequentially (for debugging)
    python permutation_eval.py

    # Run a specific worker's batch (for Modal distribution)
    python permutation_eval.py --worker_id 0 --num_workers 10

    # Run a specific stack size only
    python permutation_eval.py --stack_size 3
"""
import argparse
import json
import os
from itertools import permutations
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import wandb
from transformers import AutoTokenizer
from tqdm import tqdm

from cartridges.cache import TrainableCache
from cartridges.data.longhealth.evals import LongHealthMultipleChoiceGenerateDataset
from cartridges.generation import flex_generate
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
from cartridges.utils import get_logger, seed_everything

from config import (
    GENERATE_MAX_NEW_TOKENS,
    MODEL_NAME,
    NUM_TOKENS,
    PATIENT_IDS,
    RESULTS_DIR,
    TEMPERATURE,
    WANDB_ENTITY,
    WANDB_PROJECT,
    patient_cache_path,
)
from evaluate_stacked import load_and_stack_caches

logger = get_logger(__name__)


def generate_all_tasks(
    stack_sizes: list[int] = None,
) -> list[dict]:
    """Generate all (stack_size, permutation) tasks."""
    if stack_sizes is None:
        stack_sizes = [2, 3, 4, 5]

    tasks = []
    for k in stack_sizes:
        for perm in permutations(range(len(PATIENT_IDS)), k):
            pids = [PATIENT_IDS[i] for i in perm]
            tasks.append({
                "stack_size": k,
                "perm_indices": list(perm),
                "patient_ids": pids,
                "ordering_str": "_".join(pids),
            })

    return tasks


def run_worker(
    tasks: list[dict],
    capture_attention: bool = False,
    log_to_wandb: bool = True,
    seed: int = 42,
) -> list[dict]:
    """Run a batch of evaluation tasks. Loads the model once."""
    seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading model {MODEL_NAME}")
    model = FlexQwen3ForCausalLM.from_pretrained(MODEL_NAME).to(device).to(torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model.eval()

    all_results = []

    # Group tasks by stack_size to minimize FlexAttention recompilation
    tasks_by_k = {}
    for task in tasks:
        k = task["stack_size"]
        tasks_by_k.setdefault(k, []).append(task)

    for k in sorted(tasks_by_k.keys()):
        k_tasks = tasks_by_k[k]
        logger.info(f"Evaluating {len(k_tasks)} permutations for k={k}")

        # Load eval dataset for all patients that appear in any permutation
        # (for a given k, the questions are always for the patients in that perm)
        for task in tqdm(k_tasks, desc=f"k={k} permutations"):
            patient_ids = task["patient_ids"]
            ordering_str = task["ordering_str"]

            # Load and stack caches in the specified order
            stacked_cache = load_and_stack_caches(patient_ids, device=device)
            stacked_cache = stacked_cache.to(device).to(torch.bfloat16)

            # Load eval dataset for this specific set of patients
            dataset = LongHealthMultipleChoiceGenerateDataset(
                config=LongHealthMultipleChoiceGenerateDataset.Config(
                    patient_ids=patient_ids,
                ),
                tokenizer=tokenizer,
                seed=seed,
            )

            num_correct = 0
            per_patient_correct = {}
            per_patient_total = {}

            for i in range(len(dataset)):
                element = dataset[i]
                input_ids = element.input_ids[0].to(device)
                seq_ids = torch.zeros(input_ids.shape[0], dtype=torch.long, device=device)
                position_ids = torch.arange(input_ids.shape[0], device=device)

                with torch.no_grad():
                    pred_ids = flex_generate(
                        model=model,
                        tokenizer=tokenizer,
                        input_ids=input_ids,
                        seq_ids=seq_ids,
                        position_ids=position_ids,
                        cache=stacked_cache,
                        max_new_tokens=GENERATE_MAX_NEW_TOKENS,
                        temperature=TEMPERATURE,
                    )

                pred_text = tokenizer.decode(
                    pred_ids.get(0, []), skip_special_tokens=True
                )
                is_correct, _ = dataset.score(
                    pred=pred_text, answer=element.answer, convo_id=element.convo_id
                )

                parts = element.convo_id.split("_")
                qpat = f"{parts[0]}_{parts[1]}"

                per_patient_correct[qpat] = per_patient_correct.get(qpat, 0) + int(is_correct)
                per_patient_total[qpat] = per_patient_total.get(qpat, 0) + 1
                num_correct += int(is_correct)

            overall_acc = num_correct / len(dataset) if len(dataset) > 0 else 0
            per_patient_acc = {
                pid: per_patient_correct.get(pid, 0) / per_patient_total.get(pid, 1)
                for pid in patient_ids
            }

            result = {
                "stack_size": k,
                "ordering": ordering_str,
                "patient_ids": patient_ids,
                "overall_accuracy": overall_acc,
                "num_questions": len(dataset),
                "num_correct": num_correct,
                **{f"accuracy_{pid}": acc for pid, acc in per_patient_acc.items()},
            }
            all_results.append(result)
            logger.info(f"  {ordering_str}: {overall_acc:.2%}")

    # Save results
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    if log_to_wandb:
        wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name=f"permutation_eval_worker",
            tags=["eval", "permutations", "longhealth", "stacking"],
            config={
                "num_tasks": len(tasks),
                "stack_sizes": list(tasks_by_k.keys()),
                "num_tokens_per_cartridge": NUM_TOKENS,
            },
        )
        wandb.log({
            "permutation/results_table": wandb.Table(
                dataframe=pd.DataFrame(all_results)
            ),
        })
        wandb.finish()

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_id", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--stack_size", type=int, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    stack_sizes = [args.stack_size] if args.stack_size else [2, 3, 4, 5]
    all_tasks = generate_all_tasks(stack_sizes=stack_sizes)
    logger.info(f"Total tasks: {len(all_tasks)}")

    if args.worker_id is not None:
        # Distribute tasks across workers
        worker_tasks = all_tasks[args.worker_id :: args.num_workers]
        logger.info(
            f"Worker {args.worker_id}/{args.num_workers}: "
            f"{len(worker_tasks)} tasks"
        )
    else:
        worker_tasks = all_tasks

    results = run_worker(
        worker_tasks,
        log_to_wandb=not args.no_wandb,
        seed=args.seed,
    )

    # Save worker results
    out_path = os.path.join(
        RESULTS_DIR,
        f"permutation_results"
        + (f"_worker{args.worker_id}" if args.worker_id is not None else "")
        + ".json",
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved {len(results)} results to {out_path}")
