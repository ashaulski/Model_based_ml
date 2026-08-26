"""GMP (Generalized Memory Polynomial) predistortion.
====================================================

Inputs:
    tx_iq: complex baseband input samples (1-D array).
    coeffs: complex GMP coefficient vector (length Ncoeff).
    bypass: if True, return the input unchanged with a zero basis matrix.

Outputs:
    y_out: complex predistorted output samples (1-D array).
    y_ordered: monomial basis matrix, shape (num_samples, Ncoeff).

Functionality:
    Create the monomial terms of the polynomial according to polynomial order
    and memory depth, then calculate the polynomial by applying the coefficients
    on the monomials.
====================================================
"""

from dpd_ml_project import config
import numpy as np




def apply_predistortion_gmp(
    tx_iq: np.ndarray,
    coeffs: np.ndarray,
    bypass: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a GMP predistortion model.

    """
    # initialize local variables from config
    _Ka = config.Config.Ka
    _La = config.Config.La
    _Kb = config.Config.Kb
    _Lb = config.Config.Lb
    _Mb = config.Config.Mb
    _Kc = config.Config.Kc
    _Lc = config.Config.Lc
    _Mc = config.Config.Mc
    _num_coeffs = config.Config.Ncoeff

    tx_iq = np.asarray(tx_iq, dtype=complex).reshape(-1)
    coeffs = np.asarray(coeffs, dtype=complex).reshape(-1)

    # Initialize y_ordered matrix
    _num_samples = len(tx_iq)
    y_ordered = np.zeros((_num_samples, _num_coeffs), dtype=complex)
                                  
    if bypass:
        return tx_iq.copy(), y_ordered

    # non lagging polynomial terms calculation
    for indx_x in range(len(tx_iq)):
        for k in range(_Ka):
            for l in range(_La):
                x_current = tx_iq[indx_x - l] if indx_x - l >= 0 else 0j
                y_ordered[indx_x][k * _La + l] = x_current * abs(x_current) ** (k)

    # negative lagging polynomial terms calculation
    for indx_x in range(len(tx_iq)):
        for k in range(_Kb):
            for l in range(_Lb):
                for m in range(1,_Mb+1):
                    x_current = tx_iq[indx_x - l] if indx_x - l >= 0 else 0j
                    x_current_cross = tx_iq[indx_x - l - m] if indx_x - l - m >= 0 else 0j
                    y_ordered[indx_x][_Ka*_La + k * (_Lb * _Mb) + l * _Mb + (m-1)] = x_current * abs(x_current_cross) ** (k)

    # positive lagging polynomial terms calculation
    for indx_x in range(len(tx_iq)):
        for k in range(_Kc):
            for l in range(_Lc):
                for m in range(1, _Mc+1):
                    x_current = tx_iq[indx_x - l] if indx_x - l >= 0 else 0j
                    cross_idx = indx_x - l + m
                    x_current_cross = tx_iq[cross_idx] if 0 <= cross_idx < len(tx_iq) else 0j
                    y_ordered[indx_x][_Ka*_La + _Kb*_Lb*_Mb + k * (_Lc * _Mc) + l * _Mc + (m-1)] = x_current * abs(x_current_cross) ** (k)

    # calculate the output signal by multiplying the ordered matrix with the coefficients
    y_out = y_ordered @ coeffs

    return y_out, y_ordered