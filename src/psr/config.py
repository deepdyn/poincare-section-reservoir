from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml


# ---------------------------
# Dataclass-based configuration
# ---------------------------

@dataclass
class DatasetConfig:
    name: str = "lorenz63"
    params: Dict[str, Any] = field(default_factory=lambda: {"dt": 0.01, "T": 20000, "seed": 0})


@dataclass
class ModelConfig:
    reservoir_size: int = 300
    leak: float = 0.3
    spectral_radius: float = 0.9
    input_scale: float = 1.0
    # Section/graph specifics (kept generic; refine as you implement):
    sections: int = 1
    section_rotation_deg: float = 0.0
    bins: int = 64
    block_sparsity: float = 0.05
    activation: str = "tanh"


@dataclass
class TrainConfig:
    ridge_alpha: float = 1e-3
    washout: int = 500
    closed_loop: bool = True
    seed: int = 0


@dataclass
class EvalConfig:
    horizons: list[int] = field(default_factory=lambda: [200, 400, 800])
    repeats: int = 5


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    # Where to write runs/figures
    artifacts_dir: str = "artifacts"


# ---------------------------
# YAML I/O
# ---------------------------

def _read_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(obj), f, sort_keys=False)


# ---------------------------
# Dict <-> Dataclass helpers
# ---------------------------

def _merge_dicts(base: Dict[str, Any], upd: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in upd.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def _apply_overrides(d: Dict[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply dotted-key overrides, e.g., {'train.seed': 42}."""
    out = dict(d)
    for dk, v in overrides.items():
        keys = dk.split(".")
        cur = out
        for k in keys[:-1]:
            if k not in cur or not isinstance(cur[k], dict):
                cur[k] = {}
            cur = cur[k]
        cur[keys[-1]] = v
    return out


def _to_config(d: Dict[str, Any]) -> Config:
    return Config(
        dataset=DatasetConfig(**d.get("dataset", {})),
        model=ModelConfig(**d.get("model", {})),
        train=TrainConfig(**d.get("train", {})),
        eval=EvalConfig(**d.get("eval", {})),
        artifacts_dir=d.get("artifacts_dir", "artifacts"),
    )


def asdict_config(cfg: Config) -> Dict[str, Any]:
    return asdict(cfg)


# ---------------------------
# Public API
# ---------------------------

def load_config(
    path: Optional[str | Path] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Config:
    """
    Load a Config from YAML and optional dotted-key overrides.

    Example:
        cfg = load_config("configs/default.yaml", overrides={"train.seed": 123, "model.bins": 128})
    """
    base: Dict[str, Any] = {}
    if path:
        base = _read_yaml(Path(path))
    if overrides:
        base = _apply_overrides(base, overrides)
    return _to_config(base)


def save_config(cfg: Config, path: str | Path) -> None:
    _write_yaml(Path(path), asdict_config(cfg))
