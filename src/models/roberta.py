"""
models/roberta.py
==================
RoBERTa via AutoModelForMultipleChoice (05_RoBERTa.ipynb).

Unlike DeBERTa (5 independent binary passes), RoBERTa sees all 5 options
together and produces one joint softmax over the 5 choices — the
multiple-choice head is built into `AutoModelForMultipleChoice`, no
custom classification head needed.
"""

from pathlib import Path
from typing import List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMultipleChoice

from ..config import ROBERTA_CFG
from ..preprocessing import build_roberta_inputs_single
from ..utils import get_device


class RoBERTaModel:
    """Convenience wrapper: loads tokenizer + checkpoint, predicts option probs."""

    def __init__(self, checkpoint_path: Path = None, device=None):
        self.checkpoint_path = Path(checkpoint_path or ROBERTA_CFG["checkpoint_path"])
        self.device = device or get_device()
        self.tokenizer = None
        self.model: AutoModelForMultipleChoice = None

    def load(self) -> "RoBERTaModel":
        self.tokenizer = AutoTokenizer.from_pretrained(ROBERTA_CFG["model_name"])
        self.model = AutoModelForMultipleChoice.from_pretrained(
            ROBERTA_CFG["model_name"]
        ).to(self.device)
        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        self.model.eval()
        return self

    @torch.no_grad()
    def predict_proba_single(self, prompt: str, options: List[str]) -> np.ndarray:
        """Returns a (5,) joint-softmax probability array in A-E order."""
        if self.model is None:
            self.load()

        prompts_x5, options_x5 = build_roberta_inputs_single(prompt, options)
        enc = self.tokenizer(
            prompts_x5, options_x5,
            max_length=ROBERTA_CFG["max_len"],
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        # Reshape to (batch=1, num_choices=5, seq_len) as AutoModelForMultipleChoice expects
        input_ids = enc["input_ids"].unsqueeze(0).to(self.device)
        attention_mask = enc["attention_mask"].unsqueeze(0).to(self.device)

        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        proba = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
        return proba

    def predict_top3_single(self, prompt: str, options: List[str]) -> List[str]:
        from ..config import REVERSE_MAP
        proba = self.predict_proba_single(prompt, options)
        top3_idx = np.argsort(proba)[-3:][::-1]
        return [REVERSE_MAP[i] for i in top3_idx]