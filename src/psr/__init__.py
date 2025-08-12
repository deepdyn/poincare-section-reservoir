"""
PSR: Poincaré Section Reservoirs — geometry-inspired reservoir computing.
"""

from .config import (
    Config,
    DatasetConfig,
    ModelConfig,
    TrainConfig,
    EvalConfig,
    load_config,
    asdict_config,
)
from .utils.logging import get_logger, setup_logging

__all__ = [
    "Config",
    "DatasetConfig",
    "ModelConfig",
    "TrainConfig",
    "EvalConfig",
    "load_config",
    "asdict_config",
    "get_logger",
    "setup_logging",
]

__version__ = "0.1.0"
