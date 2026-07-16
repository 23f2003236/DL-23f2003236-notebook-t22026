"""
models/tfidf.py
================
TF-IDF + Logistic Regression baseline wrapper (02_baseline.ipynb).
Loads the pickled vectorizer + classifier and exposes a predict_proba()
that returns per-option probabilities in A-E order.
"""

from pathlib import Path
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ..config import TFIDF_CFG
from ..preprocessing import build_tfidf_text_single
from ..utils import load_pickle


class TFIDFModel:
    """Wraps the fitted TfidfVectorizer + LogisticRegression pair."""

    def __init__(self, model_path: Path = None, vectorizer_path: Path = None,
                 strategy: str = None):
        self.model_path = Path(model_path or TFIDF_CFG["model_path"])
        self.vectorizer_path = Path(vectorizer_path or TFIDF_CFG["vectorizer_path"])
        self.strategy = strategy or TFIDF_CFG["text_strategy"]
        self.model: LogisticRegression = None
        self.vectorizer: TfidfVectorizer = None

    def load(self) -> "TFIDFModel":
        self.model = load_pickle(self.model_path)
        self.vectorizer = load_pickle(self.vectorizer_path)
        return self

    def predict_proba_single(self, prompt: str, options: List[str]) -> np.ndarray:
        """Returns a (5,) probability array in A-E order for one question."""
        if self.model is None or self.vectorizer is None:
            self.load()
        text = build_tfidf_text_single(prompt, options, strategy=self.strategy)
        X = self.vectorizer.transform([text])
        proba = self.model.predict_proba(X)[0]
        return proba

    def predict_top3_single(self, prompt: str, options: List[str]) -> List[str]:
        """Returns top-3 option letters (e.g. ['B', 'D', 'A']) for one question."""
        from ..config import REVERSE_MAP
        proba = self.predict_proba_single(prompt, options)
        top3_idx = np.argsort(proba)[-3:][::-1]
        return [REVERSE_MAP[i] for i in top3_idx]