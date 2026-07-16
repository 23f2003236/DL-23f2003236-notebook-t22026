"""
metrics.py
==========
MAP@3 (Mean Average Precision @ 3) — the competition metric used identically
across all 6 notebooks:

    Position of correct answer  |  Score
    ----------------------------|-------
    1st                         |  1.00
    2nd                         |  0.50
    3rd                         |  0.33
    Not in top-3                |  0.00
"""

from typing import Sequence
import numpy as np

try:
    import torch
except ImportError:  # torch not required for the TF-IDF-only path
    torch = None


def map_at_3(y_true: Sequence[int], y_proba: np.ndarray) -> float:
    """
    MAP@3 from a numpy probability matrix.

    Args:
        y_true:  1D array/list of true integer class labels (0-4)
        y_proba: 2D array (n_samples, n_classes) of predicted probabilities

    Returns:
        float MAP@3 score
    """
    scores = []
    for i, true in enumerate(y_true):
        top3 = np.argsort(y_proba[i])[-3:][::-1]
        hit = np.where(top3 == true)[0]
        score = 1.0 / (hit[0] + 1) if len(hit) else 0.0
        scores.append(score)
    return float(np.mean(scores))


def map_at_3_from_logits(logits, labels) -> float:
    """
    Same metric, but takes raw torch logits + integer label tensor
    (used in the LSTM and RoBERTa training loops).
    """
    if torch is None:
        raise ImportError("torch is required for map_at_3_from_logits")
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    return map_at_3(labels, probs)


def validate_submission_format(predictions: Sequence[str]) -> int:
    """
    Sanity check used at the end of every notebook: every prediction string
    must be exactly 3 space-separated letters from A-E.

    Returns the number of malformed rows (should be 0).
    """
    errors = sum(
        1
        for p in predictions
        if len(str(p).split()) != 3 or not all(c in "ABCDE" for c in str(p).split())
    )
    return errors