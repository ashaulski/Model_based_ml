"""GMP (Generalized Memory Polynomial) predistorter.

Owns the single source of truth for the GMP basis: ``basis(x) -> Phi``. Both
``predistort`` (y = Phi @ coeffs) and the linear adapters (LS/RLS/Kalman, which
call ``model.basis(...)``) use it, so the monomial construction is defined once.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from dpd_ml_project.core.per_arc_config import GmpConfig


class GmpPredistorter:
    """Linear-in-parameters GMP predistorter."""

    def __init__(self, cfg: GmpConfig | None = None, coeffs: np.ndarray | None = None) -> None:
        self.cfg = cfg or GmpConfig()
        if coeffs is None:
            # Identity init: only the first non-lagging term (k=0, l=0) is 1, so
            # predistort(x) == x until the adapter learns. No warm-up bypass needed.
            c = np.zeros(self.cfg.Ncoeff, dtype=complex)
            c[0] = 1.0 + 0.0j
            self.coeffs = c
        else:
            self.coeffs = np.asarray(coeffs, dtype=complex).reshape(-1)

    def basis(self, x: np.ndarray) -> np.ndarray:
        """Build the GMP monomial basis matrix Phi of shape (N, Ncoeff)."""
        cfg = self.cfg
        Ka, La = cfg.Ka, cfg.La
        Kb, Lb, Mb = cfg.Kb, cfg.Lb, cfg.Mb
        Kc, Lc, Mc = cfg.Kc, cfg.Lc, cfg.Mc

        x = np.asarray(x, dtype=complex).reshape(-1)
        num_samples = len(x)
        phi = np.zeros((num_samples, cfg.Ncoeff), dtype=complex)

        # non-lagging polynomial terms
        for n in range(num_samples):
            for k in range(Ka):
                for l in range(La):
                    x_cur = x[n - l] if n - l >= 0 else 0j
                    phi[n][k * La + l] = x_cur * abs(x_cur) ** k

        # negative lagging polynomial terms
        for n in range(num_samples):
            for k in range(Kb):
                for l in range(Lb):
                    for m in range(1, Mb + 1):
                        x_cur = x[n - l] if n - l >= 0 else 0j
                        x_cross = x[n - l - m] if n - l - m >= 0 else 0j
                        phi[n][Ka * La + k * (Lb * Mb) + l * Mb + (m - 1)] = x_cur * abs(x_cross) ** k

        # positive lagging polynomial terms
        for n in range(num_samples):
            for k in range(Kc):
                for l in range(Lc):
                    for m in range(1, Mc + 1):
                        x_cur = x[n - l] if n - l >= 0 else 0j
                        cross_idx = n - l + m
                        x_cross = x[cross_idx] if 0 <= cross_idx < num_samples else 0j
                        phi[n][Ka * La + Kb * Lb * Mb + k * (Lc * Mc) + l * Mc + (m - 1)] = x_cur * abs(x_cross) ** k

        return phi

    def predistort(self, x: np.ndarray) -> np.ndarray:
        return self.basis(x) @ self.coeffs

    def get_state(self) -> dict[str, Any]:
        return {"coeffs": self.coeffs.copy(), "cfg": self.cfg}

    def set_state(self, state: dict[str, Any]) -> None:
        self.coeffs = np.asarray(state["coeffs"], dtype=complex).reshape(-1)
        if "cfg" in state and state["cfg"] is not None:
            self.cfg = state["cfg"]
