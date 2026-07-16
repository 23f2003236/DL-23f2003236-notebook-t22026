"""
config.py
=========
Single source of truth for every hyperparameter, path, and constant used
across the Smart MCQ Solver project. Every notebook (01_eda -> 06_ensemble)
used these exact values during training, so inference MUST use the same
ones or predictions will not match the trained checkpoints.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR       = Path(__file__).resolve().parent.parent
DATA_DIR       = BASE_DIR / "data"
OUTPUT_DIR     = BASE_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
LOGS_DIR       = OUTPUT_DIR / "logs"

for d in (CHECKPOINT_DIR, PREDICTIONS_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Answer label maps (used identically in all 4 model notebooks)
# ---------------------------------------------------------------------------
ANSWER_MAP  = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
REVERSE_MAP = {v: k for k, v in ANSWER_MAP.items()}
OPTION_COLS = list("ABCDE")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. TF-IDF + Logistic Regression baseline  (02_baseline.ipynb)
#    Best strategy in the notebook was picked dynamically (max val MAP@3);
#    all three text-builder strategies are implemented in preprocessing.py.
#    Standalone leaderboard score: 0.751
# ---------------------------------------------------------------------------
TFIDF_CFG = {
    "text_strategy": "v3_labeled",   # change if a different variant was your best
    "max_features": 15000,
    "min_df": 2,
    "max_df": 0.9,
    "ngram_range": (1, 3),
    "lr_C": 3.0,
    "model_path": CHECKPOINT_DIR / "tfidf_best_lr_model.pkl",
    "vectorizer_path": CHECKPOINT_DIR / "tfidf_best_vectorizer.pkl",
}

# ---------------------------------------------------------------------------
# 2. LSTM from scratch (03_lstm.ipynb)
#    Standalone leaderboard score: 0.7543
# ---------------------------------------------------------------------------
LSTM_CFG = {
    "vocab_size": 10000,
    "max_len": 256,
    "embed_dim": 128,
    "hidden_dim": 256,
    "num_layers": 2,
    "dropout": 0.4,
    "num_classes": 5,
    "checkpoint_path": CHECKPOINT_DIR / "lstm_best.pt",
    "tokenizer_path": CHECKPOINT_DIR / "lstm_tokenizer.pkl",
}

# ---------------------------------------------------------------------------
# 3. DeBERTa-v3-small option scorer (04_DeBERTa.ipynb)
#    Standalone leaderboard score: 0.7547 (best single model)
# ---------------------------------------------------------------------------
DEBERTA_CFG = {
    "model_name": "microsoft/deberta-v3-small",
    "max_len": 256,
    "dropout": 0.3,
    "checkpoint_path": CHECKPOINT_DIR / "deberta_best.pt",
}

# ---------------------------------------------------------------------------
# 4. RoBERTa multiple-choice (05_RoBERTa.ipynb)
#    Standalone leaderboard score: 0.75436
# ---------------------------------------------------------------------------
ROBERTA_CFG = {
    "model_name": "roberta-base",
    "max_len": 128,
    "checkpoint_path": CHECKPOINT_DIR / "roberta_best.pt",
}

# ---------------------------------------------------------------------------
# 5. Ensemble weights (06_ensemble.ipynb) — weighted rank ensemble
#    Tuned empirically against the leaderboard. Must sum to 1.0.
#    Final ensemble leaderboard score: 0.76018 (best overall)
# ---------------------------------------------------------------------------
ENSEMBLE_WEIGHTS = {
    "tfidf": 0.15,
    "lstm": 0.20,
    "deberta": 0.40,
    "roberta": 0.25,
}

assert abs(sum(ENSEMBLE_WEIGHTS.values()) - 1.0) < 1e-9, "Ensemble weights must sum to 1.0!"

# ---------------------------------------------------------------------------
# Individual model standalone scores (for reference / README / app.py display)
# ---------------------------------------------------------------------------
LEADERBOARD_SCORES = {
    "tfidf": 0.7510,
    "lstm": 0.7543,
    "roberta": 0.75436,
    "deberta": 0.7547,
    "ensemble": 0.76018,
}


# WANDB SETTINGS
WANDB_PROJECT = "23f2003236-t22026"
WANDB_ENTITY = "23f2003236"  


