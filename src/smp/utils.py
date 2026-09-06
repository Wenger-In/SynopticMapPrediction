"""Small, dependency-light helpers shared by forecasting scripts."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np


def set_random_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, TensorFlow, and PyTorch when they are installed."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """Return a centered moving average with the original array length."""
    if window_size < 1:
        raise ValueError("window_size must be positive")
    values = np.asarray(data)
    return np.convolve(values, np.ones(window_size) / window_size, mode="same")


def require_file(path: str | Path, description: str = "input file") -> Path:
    """Validate an input path and provide an actionable error message."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Missing {description}: {resolved}. Configure it in config/local.json "
            "or with SMP_DATA_ROOT."
        )
    return resolved


def min_max_normalize(data: np.ndarray) -> np.ndarray:
    """Scale an array to [0, 1], rejecting constant input."""
    values = np.asarray(data)
    data_min = np.nanmin(values)
    span = np.nanmax(values) - data_min
    if span == 0:
        raise ValueError("Cannot normalize a constant array")
    return (values - data_min) / span
