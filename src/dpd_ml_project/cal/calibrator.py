from __future__ import annotations

from typing import Literal
import numpy as np
from dpd_ml_project import config

from dpd_ml_project.cal.estimator_kal import estimate_coefficients_kal
from dpd_ml_project.cal.estimator_ls import estimate_coefficients_ls
from dpd_ml_project.cal.estimator_rls import estimate_coefficients_rls


class Calibrator:
    """Stateful DPD coefficient estimator.

    Maintains coefficients and Kalman state across iterations so that RLS 
    and Kalman can track changes incrementally.

    Args:
        mode: "ls" for batch least-squares estimation, "rls" for recursive 
              least-squares tracking, "kal" for Kalman estimation.
    """

    def __init__(self, mode: Literal["ls", "rls", "kal"] = "ls") -> None:
        cfg = config.Config()
        Ka = cfg.Ka
        La = cfg.La
        Kb = cfg.Kb
        Lb = cfg.Lb
        Mb = cfg.Mb
        Kc = cfg.Kc
        Lc = cfg.Lc
        Mc = cfg.Mc
        num_a_coeffs = Ka * La
        num_b_coeffs = Kb * Lb * Mb
        num_c_coeffs = Kc * Lc * Mc
        self.Nvec_cal_mat = config.Config.Nvec_cal_mat      # Number of vectors used for calibration matrix
        self.mode: Literal["ls", "rls", "kal"] = mode
        self.coeffs: np.ndarray = np.ones((num_a_coeffs + num_b_coeffs + num_c_coeffs,), dtype=complex)
        self.pre_coeffs: np.ndarray = np.ones((num_a_coeffs + num_b_coeffs + num_c_coeffs,), dtype=complex)
        # Kalman filter initialization
        self.P_kal = np.eye(config.Config.Ncoeff)  # Kalman covariance matrix, persists across iterations
        self.R_kal = 1e-6   # Kalman measurement noise covariance
        self.Q_kal = 1e-6 * np.eye(config.Config.Ncoeff)  # Kalman process noise covariance
        # RLS initialization
        self.P_rls = np.eye(config.Config.Ncoeff)  # RLS covariance matrix
        self.lambda_rls = 0.999  # RLS forgetting factor
        self.cost_rls = 0.0  # RLS cost, persists across iterations
        self.e_rls = 0.0  # RLS error, persists across iterations
        self.g_rls = np.zeros((config.Config.Ncoeff, 1), dtype=complex)  # RLS gain vector, persists across iterations

        self.prev_ib_evm_db: float = 0.0  # previous iteration's in-band EVM


    def update(
        self,
        tx_prd_out: np.ndarray | None = None,
        tx_pod_out: np.ndarray | None = None,
        rf_out: np.ndarray | None = None,
        y_ordered: np.ndarray | None = None,
        bypass: bool = False
    ) -> "Calibrator":
        """Estimate or update DPD coefficients and store them.

        Args:
            tx_prd_out: Predistorted signal output.
            tx_pod_out: Post-distorted signal output.
            rf_out: RF output signal.
            y_ordered: Ordered signal for coefficient estimation.
            bypass: If True, return current coefficients unchanged.

        Returns:
            Updated (or unchanged) Calibrator instance.
        """
        if bypass:
            return self

        if self.mode == "ls":
            self = estimate_coefficients_ls(self, rf_out, tx_prd_out, y_ordered)
        elif self.mode == "rls":
            self = estimate_coefficients_rls(self, tx_prd_out, y_ordered)
        elif self.mode == "kal":
            self = estimate_coefficients_kal(self, tx_prd_out, y_ordered)
        return self
