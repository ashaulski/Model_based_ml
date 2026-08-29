"""Kalman-filter adapter for any linear-in-parameters predistorter.

Indirect-learning post-distorter identification: treat the coefficient vector as
a random-walk state and track it with a scalar-measurement Kalman filter so that
``model.basis(pa_out) @ coeffs`` follows ``pa_in``. The covariance ``P`` persists
across ``update`` calls for iteration-to-iteration tracking.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class KalmanAdapter:
    """Scalar-measurement Kalman estimator operating via ``model.basis``."""

    def __init__(self, r_kal: float = 1e-6, q_kal: float = 1e-6) -> None:
        self.r_kal = float(r_kal)          # measurement noise (scalar)
        self.q_kal = float(q_kal)          # process noise (diagonal scale)
        self.P: np.ndarray | None = None   # coefficient covariance, persists

    def reset(self) -> None:
        self.P = None

    def update(self, model: Any, capture: Any) -> None:
        # Regressors from PA output, target is the PA input (indirect learning).
        phi = model.basis(capture.pa_out)
        target = np.asarray(capture.pa_in, dtype=complex).reshape(-1)
        start = model.cfg.start_data_index
        num_coeff = phi.shape[1]

        if self.P is None:
            self.P = np.eye(num_coeff, dtype=complex)

        eye = np.eye(num_coeff, dtype=complex)
        Q = self.q_kal * eye
        theta = np.asarray(model.coeffs, dtype=complex).reshape(-1, 1)

        for n in range(len(target) - start):
            u = phi[start + n].reshape(-1, 1)          # measurement vector
            x = target[start + n]                       # scalar measurement
            P_pred = self.P + Q
            denom = (u.conj().T @ P_pred @ u + self.r_kal)
            K = P_pred @ u / denom
            innovation = x - (u.conj().T @ theta).item()
            theta = theta + K * innovation
            self.P = (eye - K @ u.conj().T) @ P_pred

        model.coeffs = theta.reshape(-1)
