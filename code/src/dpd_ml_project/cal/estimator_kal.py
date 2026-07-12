from dpd_ml_project import config
import numpy as np

from dpd_ml_project.cal import calibrator


def estimate_coefficients_kal(
    calibrator,  # Calibrator instance
    tx_prd_out: np.ndarray,
    y_ordered: np.ndarray,
) -> "calibrator.Calibrator":
    """Estimate DPD coefficients using Kalman filter.
    
    Accesses P_kal from the Calibrator instance to maintain state across iterations.

    Args:
        tx_prd_out: Predistorted signal output.
        y_ordered: polynomial's monomials structured from RF output signal 
        calibrator: Calibrator instance (has access to self.P_kal).

    Returns:
        Updated Calibrator instance.
    """
   
    start_data_index = config.Config.start_data_index
    tx_prd_out = np.asarray(tx_prd_out, dtype=complex).reshape(-1)
    y_ordered = np.asarray(y_ordered, dtype=complex)

    # scalar Kalman filter update
    for n in range(len(tx_prd_out) - start_data_index):

        phi = np.array(y_ordered[start_data_index + n], dtype=complex)        
        phi = phi.reshape(-1, 1)  # column vector
        x = tx_prd_out[start_data_index + n]  # scalar

        # prediction step (no process noise, so prediction is just previous state)
        theta_pred = np.asarray(calibrator.coeffs, dtype=complex).reshape(-1, 1)  # column vector
        P_pred = calibrator.P_kal + calibrator.Q_kal  # predicted covariance is previous covariance plus process noise

        # Kalman gain
        denom = phi.conj().T @ P_pred @ phi + calibrator.R_kal  # scalar
        K = P_pred @ phi / denom  # column vector (Mx1)

        # update step
        innovation = x - (phi.conj().T @ theta_pred).item()  # scalar
        theta_upd = theta_pred + K * innovation  # column vector (Mx1)
        P_upd = (np.eye(len(K)) - K @ phi.conj().T) @ P_pred  # updated covariance

        # store updated state and covariance back in calibrator
        calibrator.coeffs = theta_upd.reshape(-1)
        calibrator.P_kal = P_upd

    return calibrator