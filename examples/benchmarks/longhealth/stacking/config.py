import os

# --- Patients ---
NUM_PATIENTS = 5
PATIENT_IDXS = list(range(1, NUM_PATIENTS + 1))
PATIENT_IDS = [f"patient_{idx:02d}" for idx in PATIENT_IDXS]

# --- Model ---
MODEL_NAME = "Qwen/Qwen3-4b"

# --- Cartridge ---
NUM_TOKENS = int(os.environ.get("CARTRIDGES_NUM_TOKENS", "512"))

# --- Training hyperparameters ---
LR = 2e-2
EPOCHS = 2
GLOBAL_BATCH_SIZE = 32
PACKED_SEQ_LENGTH = 2048
TOP_K_LOGITS = 20

# --- Evaluation ---
TEMPERATURE = 0.3
GENERATE_MAX_NEW_TOKENS = 512
EVAL_BATCH_SIZE = 32
EARLY_STOP_ACCURACY = 0.80

# --- Data sources (existing HF synthesis data for all 10 patients) ---
HF_DATA_SOURCES = [
    "hazyresearch/m07d11_longhealth_synthesize_qwen3-4b_p10_n65536-0",
    "hazyresearch/m07d11_longhealth_synthesize_qwen3-4b_p10_n65536-1",
]

# --- Paths ---
OUTPUT_DIR = os.environ.get("CARTRIDGES_OUTPUT_DIR", ".")
STACKING_DIR = os.path.join(OUTPUT_DIR, "stacking")
DATA_DIR = os.path.join(STACKING_DIR, "data")
CACHES_DIR = os.path.join(STACKING_DIR, "caches")
RESULTS_DIR = os.path.join(STACKING_DIR, "results")
FIGURES_DIR = os.path.join(STACKING_DIR, "figures")

# --- Wandb ---
WANDB_ENTITY = "jakub-smekal"
WANDB_PROJECT = "cartridges"


def patient_data_path(patient_id: str) -> str:
    return os.path.join(DATA_DIR, f"{patient_id}.parquet")


def patient_text_path(patient_id: str) -> str:
    return os.path.join(DATA_DIR, f"{patient_id}_text.txt")


def patient_cache_dir(patient_id: str) -> str:
    return os.path.join(CACHES_DIR, patient_id)


def patient_cache_path(patient_id: str) -> str:
    return os.path.join(patient_cache_dir(patient_id), "cache_last.pt")
