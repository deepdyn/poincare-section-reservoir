from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
import yaml


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def make_run_dir(
    root: str | Path = "artifacts/runs",
    dataset: Optional[str] = None,
    prefix: Optional[str] = None,
) -> Path:
    """
    Create a timestamped run directory like artifacts/runs/{dataset}/{YYYY-mm-dd_HH-MM-SS}.
    """
    root = Path(root)
    parts = [root]
    if dataset:
        parts.append(Path(dataset))
    if prefix:
        parts.append(Path(prefix))
    parts.append(Path(_ts()))
    run_dir = Path(*parts)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ---------- YAML ----------

def save_yaml(path: str | Path, obj: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(obj), f, sort_keys=False)


def load_yaml(path: str | Path) -> Mapping[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------- JSON ----------

def save_json(path: str | Path, obj: Any, *, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------- Pickle ----------

def save_pickle(path: str | Path, obj: Any, *, protocol: int = pickle.HIGHEST_PROTOCOL) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=protocol)


def load_pickle(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------- NumPy ----------

def save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


# ---------- CSV / DataFrame ----------

def save_csv(path: str | Path, df: pd.DataFrame, *, index: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


# ---------- Matplotlib figures ----------

def save_figure(fig, path: str | Path, *, dpi: int = 300, tight_layout: bool = True) -> None:
    """
    Save a Matplotlib figure and close it to free memory.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight_layout:
        try:
            fig.tight_layout()
        except Exception:
            pass
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass
