"""Protocol interfaces that define a DPD architecture.

A DPD architecture is split into two responsibilities:
    * Predistorter - a parametric map ``x -> y`` that applies predistortion.
    * Adapter      - an algorithm that fits/updates the predistorter's params
                     from a captured (pa_in, pa_out) record.

Linear-in-parameters predistorters (GMP, memory polynomial, ...) additionally
expose ``basis(x) -> Phi`` so the same LS/RLS/Kalman adapters work on all of
them without knowing the specific basis.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Predistorter(Protocol):
    """Applies predistortion and exposes serializable state."""

    def predistort(self, x: np.ndarray) -> np.ndarray: ...

    def get_state(self) -> dict[str, Any]: ...

    def set_state(self, state: dict[str, Any]) -> None: ...


@runtime_checkable
class LinearPredistorter(Predistorter, Protocol):
    """A predistorter that is linear in its coefficient vector."""

    coeffs: np.ndarray

    def basis(self, x: np.ndarray) -> np.ndarray:
        """Return the basis (regressor) matrix Phi of shape (N, Ncoeff)."""
        ...


@runtime_checkable
class Adapter(Protocol):
    """Fits/updates a predistorter's parameters from a Capture."""

    def update(self, model: Predistorter, capture: "Any") -> None: ...
