from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import yaml


def load_csv(path: str | Path, usecols: Optional[Iterable[str | int]] = None) -> pd.DataFrame:
    """
    Load a CSV into a DataFrame. Use `df.values` for ndarray.
    """
    return pd.read_csv(path, usecols=usecols)


def load_npy(path: str | Path) -> np.ndarray:
    return np.load(path)


def load_npz(path: str | Path, keys: Optional[Iterable[str]] = None) -> dict[str, np.ndarray] | np.ndarray:
    with np.load(path) as z:
        if keys is None:
            return {k: z[k] for k in z.files}
        if isinstance(keys, (list, tuple)):
            return {k: z[k] for k in keys}
        raise TypeError("keys must be an iterable of strings or None")


def load_yaml_dict(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
