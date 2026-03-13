"""
Utilities for deterministic execution.
"""

import random
import numpy as np


def set_global_seed(seed: int) -> None:
    """
    Set seeds for Python and NumPy.

    Parameters
    ----------
    seed : int
        Random seed
    """
    random.seed(seed)
    np.random.seed(seed)