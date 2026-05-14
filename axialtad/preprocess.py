"""Preprocessing utilities for AxialTAD Hi-C patch input."""
from __future__ import annotations

import numpy as np


def standardize_patch(a: np.ndarray, patch: int) -> np.ndarray:
    """Per-row standardization for Hi-C patch input.

    Args:
        a: Input array of shape (..., patch, patch, 1) or similar.
        patch: Patch size.

    Returns:
        Standardized array.
    """
    oshape = a.shape
    a = a.reshape(-1, patch).astype('float32')
    mean = np.mean(a, axis=1).reshape(-1, 1)
    a = a - mean
    s = (np.sqrt(a.var(axis=1)) + 1e-10).reshape(-1, 1)
    a = a / s
    return a.reshape(oshape)
