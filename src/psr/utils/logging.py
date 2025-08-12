from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


_LOG_FORMAT = "[%(levelname)s %(asctime)s %(name)s] %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def setup_logging(level: str | int = "INFO", log_file: Optional[str | Path] = None) -> None:
    """
    Configure root logging once for the whole package.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Clear existing handlers to avoid duplicate logs in notebooks.
    root = logging.getLogger()
    if root.handlers:
        for h in list(root.handlers):
            root.removeHandler(h)

    root.setLevel(level)

    stream = logging.StreamHandler()
    stream.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(stream)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        root.addHandler(fh)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a namespaced logger (after setup_logging has been called).
    """
    return logging.getLogger(name if name else "psr")
