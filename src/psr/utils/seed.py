from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_global_seed(seed: Optional[int] = 0, *, deterministic_torch: bool = True) -> None:
    """
    Seed Python, NumPy, and (optionally) PyTorch if available.

    Args:
        seed: Seed value. If None, does nothing.
        deterministic_torch: If True and torch is available, set deterministic flags.
    """
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    except Exception:
        # PyTorch not installed or environment not suitable — ignore silently.
        pass
