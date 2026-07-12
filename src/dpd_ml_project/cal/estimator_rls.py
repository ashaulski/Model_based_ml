from dpd_ml_project import config
import numpy as np

from dpd_ml_project.cal import calibrator


def estimate_coefficients_rls(
    calibrator,  # Calibrator instance
    tx_prd_out: np.ndarray,
    y_ordered: np.ndarray,
) -> "calibrator.Calibrator":
    """Estimate DPD coefficients using recursive least-squares method.
    
    Args:
        calibrator: Calibrator instance.
        tx_prd_out: Predistorted signal output.
        y_ordered: polynomial's monomials structured from RF output signal 
    Returns:
        Updated (or unchanged) Calibrator instance.
    """
    start_data_index = config.Config.start_data_index
    tx_prd_out = np.asarray(tx_prd_out, dtype=complex).reshape(-1)
    y_ordered = np.asarray(y_ordered, dtype=complex)
    omega_n = np.asarray(calibrator.coeffs, dtype=complex).reshape(-1, 1)

    # scalar RLS update
    for n in range(len(tx_prd_out) - start_data_index):
    # for n in range(100):
        lambda_pm1 = calibrator.lambda_rls**-1
        u = y_ordered[start_data_index + n]  # input vector (monomials)
        u = np.array(u, dtype=complex).reshape(1, -1)  # row vector
        gamma_n = 1/(1+ lambda_pm1 * u @ calibrator.P_rls @ u.conj().T)  # scalar
        g_n = lambda_pm1 * gamma_n * calibrator.P_rls @ u.conj().T  # gain vector (Mx1)
        e_n = tx_prd_out[start_data_index + n] - u @ omega_n  # scalar
        omega_n = omega_n + g_n * e_n
        calibrator.P_rls = lambda_pm1 * calibrator.P_rls - g_n @ g_n.conj().T / gamma_n  # update RLS covariance
        calibrator.cost_rls = calibrator.lambda_rls * calibrator.cost_rls + gamma_n * np.abs(e_n)**2  # update RLS cost

    calibrator.coeffs = omega_n.reshape(-1)
    return calibrator