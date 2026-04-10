"""Phase 1: Prepare per-patient data.

1a. Download HF synthesis datasets, filter by patient_id in system_prompt,
    write per-patient parquet files.
1b. Export each patient's full medical records to a text file for KVFromText
    cache initialization.
"""
import os
from pathlib import Path

from cartridges.data.longhealth.resources import FULL_STRING_TEMPLATE
from cartridges.data.longhealth.utils import load_longhealth_dataset
from cartridges.structs import write_conversations
from cartridges.utils.hf import read_conversations_from_hf
from cartridges.utils import get_logger

from config import (
    DATA_DIR,
    HF_DATA_SOURCES,
    PATIENT_IDS,
    patient_data_path,
    patient_text_path,
)

logger = get_logger(__name__)


def filter_and_write_per_patient_parquets():
    """Download all-patient HF data and split into per-patient parquet files."""
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

    logger.info("Loading conversations from HuggingFace...")
    all_convos = []
    for source in HF_DATA_SOURCES:
        logger.info(f"  Loading {source}")
        convos = read_conversations_from_hf(source)
        logger.info(f"  Loaded {len(convos)} conversations")
        all_convos.extend(convos)
    logger.info(f"Total conversations loaded: {len(all_convos)}")

    for patient_id in PATIENT_IDS:
        patient_convos = [
            c for c in all_convos if patient_id in (c.system_prompt or "")
        ]
        out_path = patient_data_path(patient_id)
        write_conversations(patient_convos, out_path)
        logger.info(f"{patient_id}: {len(patient_convos)} conversations → {out_path}")

    logger.info("Done filtering per-patient data.")


def export_patient_text_files():
    """Write each patient's medical records to a text file for KVFromText."""
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

    patients = load_longhealth_dataset(PATIENT_IDS)
    for patient in patients:
        notes = "\n".join(
            f"<{nid}>\n{text}\n</{nid}>"
            for nid, text in patient.texts.items()
        )
        text = FULL_STRING_TEMPLATE.format(
            name=patient.name,
            patient_id=patient.patient_id,
            birthday=patient.birthday,
            diagnosis=patient.diagnosis,
            num_notes=len(patient.texts),
            notes=notes,
        )
        out_path = patient_text_path(patient.patient_id)
        with open(out_path, "w") as f:
            f.write(text)
        logger.info(f"{patient.patient_id}: {len(text)} chars → {out_path}")

    logger.info("Done exporting patient text files.")


if __name__ == "__main__":
    filter_and_write_per_patient_parquets()
    export_patient_text_files()
