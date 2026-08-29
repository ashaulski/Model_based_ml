"""Shared, architecture-independent simulation configuration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


def _build_data_carriers(pilot_carriers: list[int]) -> list[int]:
    """Return OFDM data carriers excluding DC and pilot tones."""
    return [k for k in range(-26, 26) if k != 0 and k not in pilot_carriers]


@dataclass
class SimConfig:
    """Global signal/simulation parameters shared by every DPD architecture."""
    tx_sample_rate: float = 20e6
    oversample: int = 8
    signal_rms_dbp: float = -8.0
    snr_db: float = 70.0

    # OFDM structure is fixed by the waveform, so expose as class-level constants.
    FFT_SIZE: ClassVar[int] = 64 * 8
    CP_LEN: ClassVar[int] = 64 * 8 // 4
    PILOT_CARRIERS: ClassVar[list[int]] = [-21, -7, 7, 21]
    PILOT_VALUES: ClassVar[list[int]] = [1, 1, 1, -1]
    DATA_CARRIERS: ClassVar[list[int]] = _build_data_carriers(PILOT_CARRIERS)

    @property
    def dpd_sample_rate(self) -> float:
        return self.tx_sample_rate * self.oversample
