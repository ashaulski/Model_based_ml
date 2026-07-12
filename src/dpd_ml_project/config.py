# config.py
"""Configuration classes for DPD pipeline and analysis."""
import numpy as np
from dataclasses import dataclass
from typing import ClassVar


def _build_data_carriers(pilot_carriers: list[int]) -> list[int]:
    """Return OFDM data carriers excluding DC and pilot tones."""
    return [k for k in range(-26, 26) if k != 0 and k not in pilot_carriers]

@dataclass
class Config:
    """Main configuration class for DPD pipeline and analysis."""
    tx_sample_rate: float = 20e6
    dpd_sample_rate: float = tx_sample_rate*8  # 640e6
    signal_rms_dbp: float = -10.0  # backoff to ~15 dB below saturation
    FFT_SIZE: ClassVar[int] = 64*8 # 2048
    CP_LEN: ClassVar[int] = 64*8//4 # 512
    PILOT_CARRIERS: ClassVar[list[int]] = [-21, -7, 7, 21]
    PILOT_VALUES: ClassVar[list[int]] = [1, 1, 1, -1]
    DATA_CARRIERS: ClassVar[list[int]] = _build_data_carriers(PILOT_CARRIERS)
    Ka: int = 4  # polynmial order
    La: int = 3    # memory depth
    Kb: int = 3  # negative lagging polynomial order
    Lb: int = 2    # negative lagging memory depth
    Mb: int = 1    # negative lagging order
    Kc: int = 3  # positive lagging polynomial order
    Lc: int = 2    # positive lagging memory depth
    Mc: int = 1    # positive lagging order
    Ncoeff: int = Ka*La + Kb*Lb*Mb + Kc*Lc*Mc  # total number of DPD coefficients
    Nvec_cal_mat: int = Ncoeff  # number of vectors used in calibration matrix
    start_data_index = np.max([La, Lb + Mb, Lc - Mc])
    iterations_per_sym: int = 2500  # number of iterations per OFDM symbol (for tracking)
