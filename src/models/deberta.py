"""
models/deberta.py
==================
DeBERTa-v3-small binary option scorer (04_DeBERTa.ipynb).

Approach: each (prompt, option) pair is scored independently as
"wrong" vs "correct". At inference we softmax and use P(correct)
to rank the 5 options for a question.

Architecture:
    DeBERTa backbone (pretrained) -> [CLS] embedding (768-dim)
        -> Dropout(0.3) -> Linear(768 -> 2)  [wrong, correct]
"""

from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

from ..config import DEBERTA_CFG
from ..preprocessing import build_deberta_pairs_single
from ..utils import get_device


class DeBERTaOptionScorer(nn.Module):
    """Exact copy of the notebook's model class."""

    def __init__(self, model_name: str, dropout: float = 0.3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 2)  # binary: wrong vs correct

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        out = self.backbone(**kwargs)
        cls = out.last_hidden_state[:, 0, :]   # [CLS] token representation
        return self.fc(self.drop(cls))          # (B, 2)


class DeBERTaModel:
    """Convenience wrapper: loads tokenizer + checkpoint, scores all 5 options."""

    def __init__(self, checkpoint_path: Path = None, device=None):
        self.checkpoint_path = Path(checkpoint_path or DEBERTA_CFG["checkpoint_path"])
        self.device = device or get_device()
        self.tokenizer = None
        self.model: DeBERTaOptionScorer = None

    def load(self) -> "DeBERTaModel":
        self.tokenizer = AutoTokenizer.from_pretrained(DEBERTA_CFG["model_name"])
        self.model = DeBERTaOptionScorer(
            DEBERTA_CFG["model_name"], dropout=DEBERTA_CFG["dropout"]
        ).to(self.device)
        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        self.model.eval()
        return self

    @torch.no_grad()
    def predict_proba_single(self, prompt: str, options: List[str]) -> np.ndarray:
        """
        Returns a (5,) array of P(correct) scores, one per option (A-E order).
        NOTE: these 5 scores are independent sigmoid-like scores, not a
        joint softmax over the 5 options (unlike RoBERTa) — that mirrors
        exactly how the notebook ranks options at inference time.
        """
        if self.model is None:
            self.load()

        pair_texts = build_deberta_pairs_single(prompt, options)
        scores = []
        for text in pair_texts:
            enc = self.tokenizer(
                text,
                max_length=DEBERTA_CFG["max_len"],
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            token_type_ids = enc.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)

            logits = self.model(input_ids, attention_mask, token_type_ids)
            p_correct = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()[0]
            scores.append(p_correct)

        return np.array(scores)

    def predict_top3_single(self, prompt: str, options: List[str]) -> List[str]:
        from ..config import REVERSE_MAP
        scores = self.predict_proba_single(prompt, options)
        top3_idx = np.argsort(scores)[-3:][::-1]
        return [REVERSE_MAP[i] for i in top3_idx]