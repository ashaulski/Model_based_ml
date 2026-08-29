"""Recursive Least Squares adapter for any linear-in-parameters predistorter.

Indirect-learning post-distorter identification: fit coeffs so that
``model.basis(pa_out) @ coeffs`` approximates ``pa_in`` (the desired PA input).
State (covariance P, cost) persists across ``update`` calls for tracking.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class RlsAdapter:
    """Exponentially-weighted RLS estimator operating via ``model.basis``."""

    def __init__(self, lambda_rls: float = 0.999) -> None:
        self.lambda_rls = float(lambda_rls)
        self.P: np.ndarray | None = None
        self.cost_rls: float = 0.0

    def reset(self) -> None:
        self.P = None
        self.cost_rls = 0.0

    def update(self, model: Any, capture: Any) -> None:
        # Regressors from PA output, target is the PA input (indirect learning).
        phi = model.basis(capture.pa_out)
        target = np.asarray(capture.pa_in, dtype=complex).reshape(-1)
        start = model.cfg.start_data_index
        num_coeff = phi.shape[1]

        if self.P is None:
            self.P = np.eye(num_coeff, dtype=complex)

        omega = np.asarray(model.coeffs, dtype=complex).reshape(-1, 1)
        lam_inv = self.lambda_rls ** -1

        for n in range(len(target) - start):
            u = phi[start + n].reshape(1, -1)
            gamma = 1.0 / (1.0 + lam_inv * (u @ self.P @ u.conj().T))
            g = lam_inv * gamma * (self.P @ u.conj().T)
            e = target[start + n] - (u @ omega)
            omega = omega + g * e
            self.P = lam_inv * self.P - g @ g.conj().T / gamma
            self.cost_rls = self.lambda_rls * self.cost_rls + float((gamma * np.abs(e) ** 2).item().real)

        model.coeffs = omega.reshape(-1)
