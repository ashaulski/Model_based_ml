"""Batch Least-Squares adapter for any linear-in-parameters predistorter.

Indirect-learning post-distorter identification: solve for coeffs so that
``model.basis(pa_out) @ coeffs`` best fits ``pa_in`` (the desired PA input) in a
least-squares sense. Stateless (batch) - each ``update`` recomputes from scratch.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class LsAdapter:
    """Batch least-squares estimator operating via ``model.basis``."""

    def __init__(self, num_samples: int | None = None) -> None:
        # None -> use every sample from start_data_index onward (overdetermined).
        self.num_samples = num_samples

    def reset(self) -> None:
        pass

    def update(self, model: Any, capture: Any) -> None:
        # Regressors from PA output, target is the PA input (indirect learning).
        phi = model.basis(capture.pa_out)
        target = np.asarray(capture.pa_in, dtype=complex).reshape(-1)
        start = model.cfg.start_data_index

        if self.num_samples is None:
            end = len(target)
        else:
            end = min(len(target), start + self.num_samples)

        A = phi[start:end]
        b = target[start:end]
        model.coeffs = (np.linalg.pinv(A) @ b).reshape(-1)
