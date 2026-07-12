import numpy as np


def compute_evm(
    reference: np.ndarray,
    measured: np.ndarray,
    bypass: bool = False,
) -> float:
    """Compute RMS EVM in percent for two equal-length vectors."""
    if bypass:
        return 0.0
    reference = np.asarray(reference, dtype=complex).reshape(-1)
    measured = np.asarray(measured, dtype=complex).reshape(-1)

    n = min(len(reference), len(measured))
    if n == 0:
        return 0.0
    err_pow = np.sum(np.abs(measured[:n] - reference[:n]) ** 2)
    ref_pow = np.sum(np.abs(reference[:n]) ** 2)
    if ref_pow == 0:
        return 0.0
    return float(10*np.log10(err_pow / ref_pow))
