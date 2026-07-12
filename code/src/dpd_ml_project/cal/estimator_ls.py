from dpd_ml_project import config
import numpy as np
from dpd_ml_project.cal import calibrator



def estimate_coefficients_ls(
    calibrator,  # Calibrator instance        
    rf_out: np.ndarray,
    tx_prd_out: np.ndarray,
    y_ordered: np.ndarray,
) -> "calibrator.Calibrator":
    """Estimate DPD coefficients using batch least-squares method.
    
    Args:
        rf_out: RF output signal.
        tx_prd_out: Predistorted signal output.
        y_ordered: polynomial's monomials structured from RF output signal 


    Returns:
        Updated (or unchanged) Calibrator instance.
    """
    Ka = config.Config.Ka
    La = config.Config.La
    Kb = config.Config.Kb
    Lb = config.Config.Lb
    Mb = config.Config.Mb
    Kc = config.Config.Kc
    Lc = config.Config.Lc
    Mc = config.Config.Mc
    Ncoeff = config.Config.Ncoeff    
    Nvec_cal_mat = 17

    rf_out = np.asarray(rf_out, dtype=complex).reshape(-1)
    tx_prd_out = np.asarray(tx_prd_out, dtype=complex).reshape(-1)
    y_ordered = np.asarray(y_ordered, dtype=complex)

    Y = np.zeros((Nvec_cal_mat, Ncoeff), dtype=complex)

    start_data_index = np.max([La, Lb + Mb, Lc - Mc])

    # non lagging polynomial terms
    vec_index = -1
    for indx_x in range(start_data_index, Nvec_cal_mat+start_data_index):
        vec_index += 1
        coeff_index = -1
        for k in range(Ka):
            for l in range(La):
                coeff_index += 1
                x_current = rf_out[indx_x - l] if indx_x - l >= 0 else 0j
                Y[vec_index, coeff_index] = x_current * abs(x_current) ** (k)

    # negative lagging polynomial terms
    vec_index = -1
    locked_coeff_index = coeff_index
    for indx_x in range(start_data_index, Nvec_cal_mat+start_data_index):
        vec_index += 1
        coeff_index = locked_coeff_index
        for k in range(Kb):
            for l in range(Lb):
                for m in range(1, Mb+1):
                    coeff_index += 1
                    x_current = rf_out[indx_x - l] if indx_x - l >= 0 else 0j
                    x_current_cross = rf_out[indx_x - l - m] if indx_x - l - m >= 0 else 0j
                    Y[vec_index, coeff_index] = x_current * abs(x_current_cross) ** (k)

    # positive lagging polynomial terms
    vec_index = -1
    locked_coeff_index = coeff_index
    for indx_x in range(start_data_index, Nvec_cal_mat+start_data_index):
        vec_index += 1
        coeff_index = locked_coeff_index
        for k in range(Kc):
            for l in range(Lc):
                for m in range(1, Mc+1):
                    coeff_index += 1
                    x_current = rf_out[indx_x - l] if indx_x - l >= 0 else 0j
                    cross_idx = indx_x - l + m
                    x_current_cross = rf_out[cross_idx] if 0 <= cross_idx < len(rf_out) else 0j
                    Y[vec_index, coeff_index] = x_current * abs(x_current_cross) ** (k)

    y_ordered = y_ordered[start_data_index:start_data_index+Nvec_cal_mat]
    X = tx_prd_out[start_data_index:start_data_index+Nvec_cal_mat]
    W = np.linalg.pinv(y_ordered) @ X
    calibrator.coeffs = np.asarray(W, dtype=complex).reshape(-1)
    
    return calibrator