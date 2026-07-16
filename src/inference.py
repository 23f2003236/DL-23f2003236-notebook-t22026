"""
inference.py
============
Loads all 4 trained models (TF-IDF, LSTM, DeBERTa, RoBERTa) and exposes a
single interface to get each model's top-3 prediction for one question.
Model loading is lazy + cached, so app.py only pays the load cost once.
"""

from typing import Dict, List

from .models import TFIDFModel, LSTMModel, DeBERTaModel, RoBERTaModel
from .utils import get_logger

logger = get_logger(__name__)

# Module-level cache so Streamlit doesn't reload models on every rerun
_MODELS = {}


def load_all_models(which: List[str] = None) -> Dict[str, object]:
    """
    Load (or return cached) model instances.

    Args:
        which: subset of ["tfidf", "lstm", "deberta", "roberta"] to load.
               Defaults to all four.
    """
    which = which or ["tfidf", "lstm", "deberta", "roberta"]

    loaders = {
        "tfidf": TFIDFModel,
        "lstm": LSTMModel,
        "deberta": DeBERTaModel,
        "roberta": RoBERTaModel,
    }

    for name in which:
        if name not in _MODELS:
            logger.info(f"Loading {name} model...")
            _MODELS[name] = loaders[name]().load()
            logger.info(f"{name} model loaded.")

    return {name: _MODELS[name] for name in which}


def predict_single_model(model_name: str, prompt: str, options: List[str]) -> List[str]:
    """Run one model on one question, return its top-3 option letters."""
    models = load_all_models([model_name])
    return models[model_name].predict_top3_single(prompt, options)


def predict_all_models(prompt: str, options: List[str]) -> Dict[str, List[str]]:
    """
    Run all 4 models on one question.

    Returns:
        {
            "tfidf":   ["B", "A", "D"],
            "lstm":    ["B", "D", "C"],
            "deberta": ["B", "A", "C"],
            "roberta": ["A", "B", "D"],
        }
    """
    models = load_all_models()
    return {
        name: model.predict_top3_single(prompt, options)
        for name, model in models.items()
    }