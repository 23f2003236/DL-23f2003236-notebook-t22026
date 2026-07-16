"""
dataset.py
==========
PyTorch Dataset classes for batch training/evaluation (used if you ever
retrain a model, not needed for single-question inference in predict.py,
which calls the tokenizer directly for speed/simplicity).
"""

from typing import List

import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import ANSWER_MAP, OPTION_COLS
from .preprocessing import build_lstm_text


class MCQDataset(Dataset):
    """LSTM dataset (03_lstm.ipynb) — one combined text per question."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int, is_test: bool = False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = build_lstm_text(row["prompt"], [row[c] for c in OPTION_COLS])

        ids = self.tokenizer.encode(text)
        ids = self.tokenizer.pad_or_truncate(ids, self.max_len)
        id_tensor = torch.tensor(ids, dtype=torch.long)

        if self.is_test:
            return id_tensor

        label = ANSWER_MAP[row["answer"]]
        return id_tensor, torch.tensor(label, dtype=torch.long)


class OptionDataset(Dataset):
    """DeBERTa dataset (04_DeBERTa.ipynb) — expects a pre-expanded pairs_df
    (see preprocessing.expand_pairs), one row per (question, option) pair."""

    def __init__(self, pairs_df: pd.DataFrame, tokenizer, max_len: int, is_test: bool = False):
        self.df = pairs_df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_len = max_len
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tok(
            row["text"],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)
        if not self.is_test:
            item["label"] = torch.tensor(row["label"], dtype=torch.long)
        return item


class MCQDatasetHF(Dataset):
    """RoBERTa dataset (05_RoBERTa.ipynb) — AutoModelForMultipleChoice format:
    5 stacked (prompt, option) encodings per question."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int, is_test: bool = False):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_len = max_len
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prompt = row["prompt"]
        options = [row[c] for c in OPTION_COLS]

        enc = self.tok(
            [prompt] * 5, options,
            max_length=self.max_len, truncation=True, padding="max_length",
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"],        # (5, max_len)
            "attention_mask": enc["attention_mask"],  # (5, max_len)
        }
        if not self.is_test:
            item["labels"] = torch.tensor(ANSWER_MAP[row["answer"]], dtype=torch.long)
        return item