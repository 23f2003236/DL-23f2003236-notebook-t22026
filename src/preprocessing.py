"""
preprocessing.py
================
Text cleaning and prompt-building functions, extracted from the notebooks.
Every model needs a slightly different input format:

  - TF-IDF     : one flat string per question (3 strategies tried)
  - LSTM       : prompt + all 5 options concatenated, word-tokenized
  - DeBERTa    : question expanded into 5 (prompt, single option) pairs
  - RoBERTa    : 5 stacked (prompt, option) pairs per question (multiple-choice)
"""

from typing import List
import pandas as pd

from .config import OPTION_COLS, ANSWER_MAP


# ---------------------------------------------------------------------------
# Shared cleaning (identical in all 4 notebooks)
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Lowercase + strip whitespace. Applied to every text column at load time."""
    return str(text).lower().strip()


def clean_dataframe(df: pd.DataFrame, cols: List[str] = None) -> pd.DataFrame:
    """Apply clean_text() to prompt + option columns of a dataframe (returns a copy)."""
    cols = cols or (["prompt"] + OPTION_COLS)
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()
    return df


def stratified_split(train_df: pd.DataFrame, val_size: float = 0.2, seed: int = 42):
    """
    Manual per-class stratified split, identical across all 4 training notebooks.
    Keeps the per-answer-letter distribution the same in train and val.
    """
    import numpy as np

    np.random.seed(seed)
    train_idx, val_idx = [], []
    for ans in "ABCDE":
        idx = train_df[train_df["answer"] == ans].index.tolist()
        np.random.shuffle(idx)
        cut = int(len(idx) * (1.0 - val_size))
        train_idx += idx[:cut]
        val_idx += idx[cut:]

    tr = train_df.loc[train_idx].reset_index(drop=True)
    va = train_df.loc[val_idx].reset_index(drop=True)
    return tr, va


# ---------------------------------------------------------------------------
# TF-IDF text-builder strategies (02_baseline.ipynb)
# ---------------------------------------------------------------------------
def build_v1_simple(df: pd.DataFrame):
    """prompt + all options concatenated (naive baseline)"""
    return (df["prompt"] + " " + df["A"] + " " + df["B"] + " " +
            df["C"] + " " + df["D"] + " " + df["E"]).values


def build_v2_repeated(df: pd.DataFrame):
    """prompt repeated before EACH option: 'Q optA Q optB Q optC Q optD Q optE'"""
    q = df["prompt"]
    return (q + " " + df["A"] + " " + q + " " + df["B"] + " " +
            q + " " + df["C"] + " " + q + " " + df["D"] + " " + q + " " + df["E"]).values


def build_v3_labeled(df: pd.DataFrame):
    """explicit 'option a: ...' labels + trigrams — best-performing strategy"""
    return (df["prompt"] +
            " option a: " + df["A"] +
            " option b: " + df["B"] +
            " option c: " + df["C"] +
            " option d: " + df["D"] +
            " option e: " + df["E"]).values


TEXT_BUILDERS = {
    "v1_simple": build_v1_simple,
    "v2_repeated": build_v2_repeated,
    "v3_labeled": build_v3_labeled,
}


def build_tfidf_text(df: pd.DataFrame, strategy: str = "v3_labeled"):
    """Dispatch to the chosen TF-IDF text-builder strategy."""
    return TEXT_BUILDERS[strategy](df)


def build_tfidf_text_single(prompt: str, options: List[str], strategy: str = "v3_labeled") -> str:
    """Same as build_tfidf_text but for a single question (used at inference time)."""
    a, b, c, d, e = options
    prompt = clean_text(prompt)
    a, b, c, d, e = [clean_text(x) for x in (a, b, c, d, e)]

    if strategy == "v1_simple":
        return f"{prompt} {a} {b} {c} {d} {e}"
    if strategy == "v2_repeated":
        return (f"{prompt} {a} {prompt} {b} {prompt} {c} "
                f"{prompt} {d} {prompt} {e}")
    if strategy == "v3_labeled":
        return (f"{prompt} option a: {a} option b: {b} option c: {c} "
                f"option d: {d} option e: {e}")
    raise ValueError(f"Unknown strategy: {strategy}")


# ---------------------------------------------------------------------------
# LSTM: prompt + all options concatenated (03_lstm.ipynb)
# ---------------------------------------------------------------------------
def build_lstm_text(prompt: str, options: List[str]) -> str:
    """Combined text used by the LSTM tokenizer: prompt + A + B + C + D + E."""
    prompt = clean_text(prompt)
    a, b, c, d, e = [clean_text(x) for x in options]
    return f"{prompt} {a} {b} {c} {d} {e}"


# ---------------------------------------------------------------------------
# DeBERTa: expand each question into 5 (prompt, option) pairs (04_DeBERTa.ipynb)
# ---------------------------------------------------------------------------
def expand_pairs(df: pd.DataFrame, is_test: bool = False) -> pd.DataFrame:
    """
    Convert each question row into 5 rows: one per option.
    text  = 'prompt option_text'
    label = 1 if this option is correct else 0 (only when not is_test)
    """
    rows = []
    for _, r in df.iterrows():
        correct = r.get("answer", None)
        for opt in OPTION_COLS:
            rows.append({
                "id": r["id"],
                "option": opt,
                "text": f"{r['prompt']} {r[opt]}",
                "label": int(opt == correct) if not is_test else -1,
            })
    return pd.DataFrame(rows)


def build_deberta_pairs_single(prompt: str, options: List[str]) -> List[str]:
    """Same expansion, but for a single question at inference time.
    Returns a list of 5 strings, one per option, in A-E order."""
    prompt = clean_text(prompt)
    return [f"{prompt} {clean_text(opt)}" for opt in options]


# ---------------------------------------------------------------------------
# RoBERTa: 5 stacked (prompt, option) pairs for AutoModelForMultipleChoice
# (05_RoBERTa.ipynb) — tokenizer handles the pairing, this just prepares lists
# ---------------------------------------------------------------------------
def build_roberta_inputs_single(prompt: str, options: List[str]):
    """
    Returns (prompts_x5, options_x5) ready to feed straight into the
    tokenizer as tokenizer([prompt]*5, options, ...).
    """
    prompt = clean_text(prompt)
    options = [clean_text(opt) for opt in options]
    return [prompt] * 5, options