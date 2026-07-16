"""
utils.py
========
Shared helpers used across every model module: reproducibility seeding,
device detection, checkpoint save/load, and a simple logger.
"""

import logging
import os
import random
import pickle
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def seed_everything(seed: int = 42) -> None:
    """Set every RNG (python, numpy, torch) for reproducibility.
    Matches the set_seed()/torch.manual_seed() calls in all training notebooks.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """Return 'cuda' if available else 'cpu' (same logic as every notebook)."""
    if torch is None:
        return "cpu"
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(model, path: Path) -> None:
    """Save a torch model's state_dict to disk (creates parent dirs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model, path: Path, device=None):
    """Load a torch model's state_dict from disk in-place and return it in eval mode."""
    device = device or get_device()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def save_pickle(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def get_logger(name: str = "smart_mcq_solver") -> logging.Logger:
    """Simple stdout logger, reused everywhere instead of print()."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger