import os
from pathlib import Path

# PROJECT PATHS 
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"

# Output path
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
LOGS_DIR = OUTPUT_DIR / "logs"

# Ensure the directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


ANSWER_MAP = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
REVERSE_MAP = {idx: letter for letter, idx in ANSWER_MAP.items()}
OPTION_COLS = ["A", "B", "C", "D", "E"]
TEXT_COLS = ["prompt", *OPTION_COLS]

SEED = 42

BASELINE_CFG = {
    "text_strategy": "v1_simple",
    "max_features": 5000,
    "min_df": 2,
    "max_df": 0.9,
    "ngram_range": (1, 2),
    "lr_C": 1.0,
}

LSTM_CFG = {
    "vocab_size": 10000,
    "max_len": 256,
    "embed_dim": 128,
    "hidden_dim": 256,
    "num_layers": 2,
    "dropout": 0.4,
    "num_classes": 5,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "epochs": 50,
    "patience": 7,
    "weight_decay": 1e-4,
    "random_state": SEED,
}

DEBERTA_CFG = {
    "model_name": "microsoft/deberta-v3-small",
    "max_len": 256,
    "batch_size": 16,
    "lr": 2e-5,
    "weight_decay": 0.01,
    "epochs": 5,
    "warmup_ratio": 0.1,
    "patience": 3,
    "dropout": 0.3,
    "pos_class_weight": 4.0,
    "seed": SEED,
}

# WANDB SETTINGS
WANDB_PROJECT = "23f2003236-t22026"
WANDB_ENTITY = "23f2003236"  

def print_config():
    """Print all configuration settings"""

    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Train CSV: {TRAIN_CSV}")
    print(f"Test CSV: {TEST_CSV}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"WANDB Project: {WANDB_PROJECT}")
    print(f"WANDB Entity: {WANDB_ENTITY}")

if __name__ == "__main__":
    print_config()
