"""Normalized mean square error, the standard DPD modeling-accuracy metric."""
from __future__ import annotations

import numpy as np

from dpd_ml_project.metrics.evm import compute_evm


def compute_nmse(reference: np.ndarray, measured: np.ndarray) -> float:
    """NMSE in dB. Identical definition to ``compute_evm`` (error power / ref power)."""
    return compute_evm(reference=reference, measured=measured, bypass=False)
