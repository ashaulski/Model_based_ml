import numpy as np


def check_relative_mask(
    measured: np.ndarray,
    max_avg_power: float = 2.0,
    bypass: bool = False,
) -> bool:
    """Placeholder relative mask check using average power threshold."""
    if bypass:
        return True
    measured = np.asarray(measured, dtype=complex).reshape(-1)

    if measured.size == 0:
        return True
    avg_power = np.mean(np.abs(measured) ** 2)
    return bool(avg_power <= max_avg_power)
