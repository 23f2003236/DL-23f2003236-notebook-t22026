"""
models/lstm.py
==============
Custom word-level tokenizer + Bidirectional LSTM classifier
(03_lstm.ipynb — the nn.LSTM-backed production version, not the
from-scratch educational one, since that's what was actually trained).

Architecture:
    [Input IDs] -> Embedding -> BiLSTM x2 -> Dropout -> Linear -> 5 logits
"""

from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn

from ..config import LSTM_CFG
from ..preprocessing import build_lstm_text
from ..utils import load_pickle, get_device


class MCQTokenizer:
    """Simple whitespace word-level tokenizer with a fixed-size vocabulary."""

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word2idx = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.idx2word = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}
        self.vocab_built = False

    def tokenize(self, text: str) -> List[str]:
        return str(text).lower().split()

    def build_vocab(self, texts: List[str]):
        counter = Counter()
        for text in texts:
            counter.update(self.tokenize(text))

        most_common = counter.most_common(self.vocab_size - 2)
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        self.vocab_built = True

    def encode(self, text: str) -> List[int]:
        words = self.tokenize(text)
        return [self.word2idx.get(w, 1) for w in words]

    def pad_or_truncate(self, ids: List[int], max_len: int) -> List[int]:
        if len(ids) >= max_len:
            return ids[:max_len]
        return ids + [0] * (max_len - len(ids))


class LSTMClassifier(nn.Module):
    """Embedding -> BiLSTM x2 -> Dropout -> Linear head. Exact copy from the notebook."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers,
                 num_classes, dropout, pad_idx: int = 0):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=pad_idx
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)

        output, (hidden, cell) = self.lstm(embedded)

        forward_hidden = hidden[-2]   # last layer, forward direction
        backward_hidden = hidden[-1]  # last layer, backward direction
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)

        out = self.dropout(combined)
        logits = self.fc(out)
        return logits


class LSTMModel:
    """Convenience wrapper: loads tokenizer + trained checkpoint, predicts on raw text."""

    def __init__(self, checkpoint_path: Path = None, tokenizer_path: Path = None, device=None):
        self.checkpoint_path = Path(checkpoint_path or LSTM_CFG["checkpoint_path"])
        self.tokenizer_path = Path(tokenizer_path or LSTM_CFG["tokenizer_path"])
        self.device = device or get_device()
        self.tokenizer: MCQTokenizer = None
        self.model: LSTMClassifier = None

    def load(self) -> "LSTMModel":
        # The tokenizer was pickled from inside a notebook, where MCQTokenizer
        # lived in the __main__ module. When unpickling from any other script
        # (like this one, or app.py), Python looks for MCQTokenizer in
        # __main__ and fails unless we register it there first.
        import sys
        sys.modules["__main__"].MCQTokenizer = MCQTokenizer

        self.tokenizer = load_pickle(self.tokenizer_path)
        self.model = LSTMClassifier(
            vocab_size=LSTM_CFG["vocab_size"],
            embed_dim=LSTM_CFG["embed_dim"],
            hidden_dim=LSTM_CFG["hidden_dim"],
            num_layers=LSTM_CFG["num_layers"],
            num_classes=LSTM_CFG["num_classes"],
            dropout=LSTM_CFG["dropout"],
        ).to(self.device)
        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        self.model.eval()
        return self

    @torch.no_grad()
    def predict_proba_single(self, prompt: str, options: List[str]) -> np.ndarray:
        if self.model is None:
            self.load()
        text = build_lstm_text(prompt, options)
        ids = self.tokenizer.encode(text)
        ids = self.tokenizer.pad_or_truncate(ids, LSTM_CFG["max_len"])
        id_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(self.device)

        logits = self.model(id_tensor)
        proba = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return proba

    def predict_top3_single(self, prompt: str, options: List[str]) -> List[str]:
        from ..config import REVERSE_MAP
        proba = self.predict_proba_single(prompt, options)
        top3_idx = np.argsort(proba)[-3:][::-1]
        return [REVERSE_MAP[i] for i in top3_idx]