"""Unified captured-data structure shared by every DPD architecture.

A ``Capture`` holds the aligned complex signals of one pipeline pass. Online
estimators (LS/RLS/Kalman) read the complex arrays directly; the NN path will
add a tensor view later (P4) without changing this structure.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Capture:
    """Aligned signals for one pipeline pass.

    Attributes:
        x_in:   reference TX signal (pre-AWGN).
        dpd_in: DPD input (post-AWGN) - the signal fed to the predistorter.
        pa_in:  predistorted signal = PA input.
        pa_out: PA output.
        rf_out_no_prd: PA output of the un-predistorted signal (reference for
                       spectrum plots); None when not captured.
        meta:   free-form metadata (iteration index, snr_db, temperature, ...).
    """
    x_in: np.ndarray
    dpd_in: np.ndarray
    pa_in: np.ndarray
    pa_out: np.ndarray
    rf_out_no_prd: np.ndarray | None = None
    meta: dict = field(default_factory=dict)
