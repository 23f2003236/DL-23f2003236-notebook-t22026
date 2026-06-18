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

# DATA SETTINGS 
#1. Train/Val split
TRAIN_TEST_SPLIT = 0.8  # 80% train, 20% val
RANDOM_SEED = 42

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
