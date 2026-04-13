"""Phase 4a: Evaluate stacked cartridges with attention distribution capture.

Loads multiple trained cartridge caches, stacks them, then evaluates the model
on the questions for all patients in the stack. Logs per-question results and
attention distributions to wandb.

Usage:
    python evaluate_stacked.py --patient_ids patient_01 patient_02 patient_03
    python evaluate_stacked.py --patient_ids patient_01 patient_02 --no_attention
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
import wandb
from transformers import AutoTokenizer
from tqdm import tqdm

from cartridges.cache import TrainableCache, AttnConfig
from cartridges.data.longhealth.evals import LongHealthMultipleChoiceGenerateDataset
from cartridges.generation import flex_generate
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
from cartridges.utils import get_logger, seed_everything

from attention_capture import AttentionCapture
from config import (
    EVAL_BATCH_SIZE,
    GENERATE_MAX_NEW_TOKENS,
    MODEL_NAME,
    NUM_TOKENS,
    RESULTS_DIR,
    TEMPERATURE,
    WANDB_ENTITY,
    WANDB_PROJECT,
    patient_cache_path,
)

logger = get_logger(__name__)


def load_and_stack_caches(
    patient_ids: list[str],
    device: str = "cuda",
    rope_adjust: bool = False,
    rope_theta: float = 10000.0,
) -> TrainableCache:
    """Load individual caches and stack them in the given order.

    Args:
        patient_ids: ordered list of patient IDs.
        device: torch device string.
        rope_adjust: if True, strip and re-apply RoPE with sequential positions
            so that stacked caches have non-overlapping position encodings.
        rope_theta: RoPE base frequency (must match the model's config).
    """
    caches = []
    for pid in patient_ids:
        path = patient_cache_path(pid)
        logger.info(f"Loading cache for {pid} from {path}")
        cache = TrainableCache.from_pretrained(path, device=device)
        caches.append(cache)

    if rope_adjust:
        stacked = TrainableCache.stack_caches_rope_adjusted(caches, rope_theta=rope_theta)
        logger.info(
            f"Stacked {len(caches)} caches with RoPE adjustment: "
            f"{stacked.num_cartridge_tokens()} total cartridge tokens"
        )
    else:
        stacked = TrainableCache.stack_caches(caches)
        logger.info(
            f"Stacked {len(caches)} caches: "
            f"{stacked.num_cartridge_tokens()} total cartridge tokens"
        )
    return stacked


def evaluate_stacked(
    patient_ids: list[str],
    capture_attention: bool = True,
    log_to_wandb: bool = True,
    seed: int = 42,
    rope_adjust: bool = False,
) -> pd.DataFrame:
    """Run stacked evaluation and return per-question results DataFrame.

    Args:
        patient_ids: ordered list of patient IDs for the stack.
        capture_attention: whether to capture attention distributions.
        log_to_wandb: whether to log results to Weights & Biases.
        seed: random seed.
        rope_adjust: if True, use RoPE-adjusted stacking to fix position
            conflicts across stacked caches.
    """
    seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    logger.info(f"Loading model {MODEL_NAME}")
    model = FlexQwen3ForCausalLM.from_pretrained(MODEL_NAME).to(device).to(torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model.eval()

    # Stack caches
    rope_theta = getattr(model.config, "rope_theta", 10000.0)
    stacked_cache = load_and_stack_caches(
        patient_ids, device=device, rope_adjust=rope_adjust, rope_theta=rope_theta
    )
    stacked_cache = stacked_cache.to(device).to(torch.bfloat16)

    # Attention hook
    attn_capture = None
    if capture_attention:
        attn_capture = AttentionCapture(
            num_tokens_per_cartridge=NUM_TOKENS,
            patient_ids=patient_ids,
        )
        attn_capture.register(model)

    # Load eval dataset for the patients in the stack
    dataset = LongHealthMultipleChoiceGenerateDataset(
        config=LongHealthMultipleChoiceGenerateDataset.Config(
            patient_ids=patient_ids,
        ),
        tokenizer=tokenizer,
        seed=seed,
    )
    logger.info(f"Eval dataset: {len(dataset)} questions for {patient_ids}")

    # Evaluate
    ordering_str = "_".join(patient_ids)
    results = []

    for i in tqdm(range(len(dataset)), desc=f"Evaluating stack [{ordering_str}]"):
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

        # Decode prediction
        pred_token_ids = pred_ids.get(0, [])
        pred_text = tokenizer.decode(pred_token_ids, skip_special_tokens=True)

        # Score
        is_correct, extras = dataset.score(
            pred=pred_text, answer=element.answer, convo_id=element.convo_id
        )

        # Extract patient from convo_id (format: patient_XX_YY)
        parts = element.convo_id.split("_")
        question_patient = f"{parts[0]}_{parts[1]}"

        row = {
            "question_idx": i,
            "convo_id": element.convo_id,
            "question_patient": question_patient,
            "question": element.prompt,
            "correct_answer": element.answer,
            "model_answer": pred_text,
            "extracted_pred": extras.get("extracted_pred"),
            "is_correct": bool(is_correct),
            "stack_ordering": ordering_str,
            "stack_size": len(patient_ids),
        }

        # Capture attention distribution
        if attn_capture is not None:
            attn_dist = attn_capture.compute_attention_distribution()
            if attn_dist is not None:
                for key, val in attn_dist.items():
                    row[f"attn_to_{key}"] = val

        results.append(row)

    # Clean up hook
    if attn_capture is not None:
        attn_capture.remove()

    df = pd.DataFrame(results)

    # Compute per-patient accuracy
    per_patient_acc = df.groupby("question_patient")["is_correct"].mean().to_dict()
    overall_acc = df["is_correct"].mean()
    logger.info(f"Overall accuracy: {overall_acc:.2%}")
    for pid, acc in sorted(per_patient_acc.items()):
        logger.info(f"  {pid}: {acc:.2%}")

    # Log to wandb
    if log_to_wandb:
        rope_suffix = "_rope" if rope_adjust else ""
        run_name = f"stacking_eval_{ordering_str}_toks{NUM_TOKENS}{rope_suffix}"
        tags = ["eval", "stacking", "longhealth", f"k{len(patient_ids)}"]
        if rope_adjust:
            tags.append("rope_adjust")
        wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name=run_name,
            tags=tags,
            config={
                "patient_ids": patient_ids,
                "stack_size": len(patient_ids),
                "num_tokens_per_cartridge": NUM_TOKENS,
                "model": MODEL_NAME,
                "temperature": TEMPERATURE,
                "rope_adjust": rope_adjust,
            },
        )

        # Per-question table
        wandb.log({"eval/results_table": wandb.Table(dataframe=df)})

        # Per-patient accuracy
        for pid, acc in per_patient_acc.items():
            wandb.log({f"eval/accuracy_{pid}": acc})
        wandb.log({"eval/accuracy_overall": overall_acc})

        # Attention distribution summary (if captured)
        attn_cols = [c for c in df.columns if c.startswith("attn_to_")]
        if attn_cols:
            # Attention by correctness
            correct_df = df[df["is_correct"]]
            incorrect_df = df[~df["is_correct"]]

            for col in attn_cols:
                region = col.replace("attn_to_", "")
                wandb.log({
                    f"attention/correct_mean_{region}": correct_df[col].mean()
                    if len(correct_df) > 0 else 0,
                    f"attention/incorrect_mean_{region}": incorrect_df[col].mean()
                    if len(incorrect_df) > 0 else 0,
                    f"attention/overall_mean_{region}": df[col].mean(),
                })

            # Log attention heatmap data as a table
            attn_df = df[["convo_id", "question_patient", "is_correct"] + attn_cols]
            wandb.log({"attention/heatmap_data": wandb.Table(dataframe=attn_df)})

        wandb.finish()

    # Save to disk
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    result_path = os.path.join(RESULTS_DIR, f"eval_{ordering_str}.json")
    df.to_json(result_path, orient="records", indent=2)
    logger.info(f"Results saved to {result_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--patient_ids", nargs="+", required=True,
        help="Ordered list of patient IDs for the stack",
    )
    parser.add_argument("--no_attention", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rope_adjust", action="store_true",
        help="Re-index RoPE positions when stacking so caches have sequential "
             "non-overlapping positions instead of all sharing 0..N-1.",
    )
    args = parser.parse_args()

    evaluate_stacked(
        patient_ids=args.patient_ids,
        capture_attention=not args.no_attention,
        log_to_wandb=not args.no_wandb,
        seed=args.seed,
        rope_adjust=args.rope_adjust,
    )
