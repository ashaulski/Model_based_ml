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
