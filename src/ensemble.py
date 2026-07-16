"""
ensemble.py
===========
Weighted rank ensemble (06_ensemble.ipynb) — combines TF-IDF, LSTM, DeBERTa,
and RoBERTa top-3 predictions into one final top-3, using reciprocal-rank
scoring weighted by each model's trustworthiness.

For every model's top-3 list:
    1st choice -> weight * 1.0
    2nd choice -> weight * 0.5
    3rd choice -> weight * 0.333

Scores are summed across all 4 models; the 3 options with the highest
combined score become the final prediction, in order.
"""

from collections import defaultdict
from typing import Dict, List

from .config import ENSEMBLE_WEIGHTS


def rank_ensemble(top3_by_model: Dict[str, List[str]],
                   weights: Dict[str, float] = None) -> List[str]:
    """
    Args:
        top3_by_model: e.g. {
            "tfidf":   ["B", "A", "D"],
            "lstm":    ["B", "D", "C"],
            "deberta": ["B", "A", "C"],
            "roberta": ["A", "B", "D"],
        }
        weights: model_name -> weight (defaults to config.ENSEMBLE_WEIGHTS)

    Returns:
        Final top-3 option letters, e.g. ["B", "A", "D"]
    """
    weights = weights or ENSEMBLE_WEIGHTS
    scores = defaultdict(float)

    for model_name, top3 in top3_by_model.items():
        w = weights.get(model_name, 0.0)
        for rank, option in enumerate(top3):
            scores[option] += w * (1.0 / (rank + 1))

    best_3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
    return [option for option, _ in best_3]


def rank_ensemble_with_scores(top3_by_model: Dict[str, List[str]],
                               weights: Dict[str, float] = None):
    """Same as rank_ensemble but also returns the raw per-option score dict,
    useful for the Streamlit app to show a confidence breakdown."""
    weights = weights or ENSEMBLE_WEIGHTS
    scores = defaultdict(float)

    for model_name, top3 in top3_by_model.items():
        w = weights.get(model_name, 0.0)
        for rank, option in enumerate(top3):
            scores[option] += w * (1.0 / (rank + 1))

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top3_final = [option for option, _ in ranked[:3]]
    return top3_final, dict(ranked)