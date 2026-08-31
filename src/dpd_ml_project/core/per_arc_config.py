"""Per-architecture configuration dataclasses.

Each DPD architecture owns its own knobs here so the shared pipeline and the
common simulation config stay architecture-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GmpConfig:
    """Generalized Memory Polynomial basis parameters.

    Term families:
        - non-lagging:      Ka polynomial orders x La memory taps
        - negative lagging: Kb x Lb x Mb
        - positive lagging: Kc x Lc x Mc
    """
    Ka: int = 4
    La: int = 3
    Kb: int = 3
    Lb: int = 3
    Mb: int = 1
    Kc: int = 3
    Lc: int = 3
    Mc: int = 1

    @property
    def Ncoeff(self) -> int:
        return self.Ka * self.La + self.Kb * self.Lb * self.Mb + self.Kc * self.Lc * self.Mc

    @property
    def start_data_index(self) -> int:
        # First sample with all memory taps in range.
        return int(max(self.La, self.Lb + self.Mb, self.Lc - self.Mc))


@dataclass
class NnConfig:
    """Neural (GRU) DPD hyperparameters. Placeholder for the NN path (P4)."""
    input_size: int = 2
    hidden_size: int = 64
    num_outputs: int = 2
    num_layers: int = 2
    fc1_size: int = 64
    fc2_size: int = 32
    dropout: float = 0.1
    enhance_features: bool = False
    lr: float = 2e-4
    batch_size: int = 64


@dataclass
class VdtdnnConfig:
    """Vector-decomposed time-delay NN (VDTDNN / AVDTDNN) hyperparameters.

    Nonlinearity acts on the input magnitude only; phase is restored by linear
    cos/sin weighting. ``augmented=True`` selects AVDTDNN (higher-order
    magnitude terms in the input vector).
    """
    memory_depth: int = 4
    neurons_per_group: int = 6
    activation: str = "abs"
    augmented: bool = False
    powers: tuple[int, ...] = (1, 2, 3)
    linear_term: bool = True

    # training knobs (paper: two-stage Adam step size)
    lr: float = 1e-2
    lr_final: float = 1e-3
    lr_switch_frac: float = 0.87
    batch_size: int = 256
    num_epochs: int = 150
    patience: int = 15

    @property
    def num_groups(self) -> int:
        """One phase-recovery group per memory tap."""
        return self.memory_depth + 1

    @property
    def num_neurons(self) -> int:
        return self.neurons_per_group * self.num_groups

    @property
    def in_powers(self) -> tuple[int, ...]:
        return tuple(self.powers) if self.augmented else (1,)

    @property
    def in_dim(self) -> int:
        return self.num_groups * len(self.in_powers)

    @property
    def num_coefficients(self) -> int:
        """Trainable coefficient count (matches eqs. 15/18 of the paper)."""
        num_phase_units = self.num_neurons + (self.num_groups if self.linear_term else 0)
        return (self.in_dim + 1) * self.num_neurons + 4 * num_phase_units
