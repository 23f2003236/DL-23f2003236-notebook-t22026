"""
predict.py
==========
The single public entrypoint used by app.py. Everything else in src/ is
plumbing — this is the one function a Streamlit UI (or any other frontend)
needs to call.
"""

from typing import List, Dict

from .inference import predict_all_models
from .ensemble import rank_ensemble_with_scores
from .config import LEADERBOARD_SCORES


def predict(prompt: str, options: List[str]) -> Dict:
    """
    Full pipeline: run all 4 models -> ensemble -> return everything the UI
    needs to render (final answer, per-model breakdown, confidence scores).

    Args:
        prompt:  the question text
        options: list of 5 option strings, in A, B, C, D, E order

    Returns:
        {
            "final_top3": ["B", "A", "D"],
            "per_model_top3": {
                "tfidf": [...], "lstm": [...], "deberta": [...], "roberta": [...]
            },
            "ensemble_scores": {"B": 0.62, "A": 0.41, ...},
            "leaderboard_scores": {...}   # for display / transparency
        }
    """
    if len(options) != 5:
        raise ValueError(f"Expected exactly 5 options (A-E), got {len(options)}")

    per_model_top3 = predict_all_models(prompt, options)
    final_top3, ensemble_scores = rank_ensemble_with_scores(per_model_top3)

    return {
        "final_top3": final_top3,
        "per_model_top3": per_model_top3,
        "ensemble_scores": ensemble_scores,
        "leaderboard_scores": LEADERBOARD_SCORES,
    }