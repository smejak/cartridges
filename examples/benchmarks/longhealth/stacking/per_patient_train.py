"""Phase 2: Train a single cartridge per patient.

Usage:
    python per_patient_train.py                          # trains all 5 patients sequentially
    python per_patient_train.py patient_idx=1            # trains patient_01 only
    CARTRIDGES_NUM_TOKENS=1024 python per_patient_train.py  # override token count
"""
import os
import sys

import pydrantic

from cartridges.initialization import KVFromText
from cartridges.models.config import HFModelConfig
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
from cartridges.train import GenerationEvalConfig, TrainConfig
from cartridges.datasets import TrainDataset, DataSource
from cartridges.data.longhealth.evals import LongHealthMultipleChoiceGenerateDataset
from cartridges.utils.wandb import WandBConfig

from config import (
    CACHES_DIR,
    EPOCHS,
    EVAL_BATCH_SIZE,
    GENERATE_MAX_NEW_TOKENS,
    GLOBAL_BATCH_SIZE,
    LR,
    MODEL_NAME,
    NUM_TOKENS,
    PACKED_SEQ_LENGTH,
    PATIENT_IDS,
    TEMPERATURE,
    TOP_K_LOGITS,
    WANDB_ENTITY,
    WANDB_PROJECT,
    patient_cache_dir,
    patient_data_path,
    patient_text_path,
)


def make_config(patient_id: str) -> TrainConfig:
    """Build a TrainConfig for a single patient."""
    return TrainConfig(
        model=HFModelConfig(
            pretrained_model_name_or_path=MODEL_NAME,
            model_cls=FlexQwen3ForCausalLM,
        ),
        kv_cache_initializer=KVFromText.Config(
            max_tokens=NUM_TOKENS,
            text_source=patient_text_path(patient_id),
        ),
        lr=LR,
        epochs=EPOCHS,
        global_batch_size=GLOBAL_BATCH_SIZE,

        dataset=TrainDataset.Config(
            data_sources=[
                DataSource(path=patient_data_path(patient_id), type="local"),
            ],
            top_k_logits=TOP_K_LOGITS,
            packed_seq_length=PACKED_SEQ_LENGTH,
            packing_mode="truncate",
        ),

        save_every_n_steps=32,
        generate_eval_every_n_steps=32,
        keep_last_n_saved=20,  # keep enough checkpoints to pick best
        generate_evals=[
            GenerationEvalConfig(
                dataset=LongHealthMultipleChoiceGenerateDataset.Config(
                    patient_ids=[patient_id],
                ),
                name_for_wandb=f"longhealth_{patient_id}",
                generate_max_new_tokens=GENERATE_MAX_NEW_TOKENS,
                batch_size=EVAL_BATCH_SIZE,
                temperature=TEMPERATURE,
            ),
        ],
        distributed_backend="gloo",

        wandb=WandBConfig(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            tags=["train", "longhealth", "stacking", "per-patient", patient_id],
        ),
        output_dir=patient_cache_dir(patient_id),
        name=f"stacking_{patient_id}_toks{NUM_TOKENS}",
    )


if __name__ == "__main__":
    # Support running a single patient via CLI: python per_patient_train.py patient_idx=1
    patient_idx = int(os.environ.get("PATIENT_IDX", "0"))
    if patient_idx > 0:
        patient_id = f"patient_{patient_idx:02d}"
        config = make_config(patient_id)
        pydrantic.main(config)
    else:
        # Default: build configs for all 5 patients (run sequentially)
        configs = [make_config(pid) for pid in PATIENT_IDS]
        pydrantic.main(configs)
