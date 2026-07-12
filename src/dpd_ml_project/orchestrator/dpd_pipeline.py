import numpy as np
import matplotlib.pyplot as plt
import mplcursors

from dataclasses import dataclass, field

from dpd_ml_project.cal.calibrator import Calibrator
from dpd_ml_project.channel.pa_model import apply_pa_model
from dpd_ml_project.channel.awgn import apply_awgn
from dpd_ml_project.dpd.dpd import apply_predistortion
from dpd_ml_project.metrics.evm import compute_evm
from dpd_ml_project.metrics.mask import check_relative_mask
from dpd_ml_project.siggen.SigGen import gen_lsig
from dpd_ml_project import config



@dataclass(slots=True)
class ModuleBypass:
    """Configure which modules are bypassed (pass-through)."""
    siggen: bool = True
    estimator: bool = True
    dprd: bool = True # predistortion
    dpod: bool = True # postdistortion    
    awgn: bool = True
    pa_model: bool = True
    capture: bool = True
    evm: bool = True
    mask: bool = True

@dataclass(slots=True)
class SetUpConfig:
    """Configuration for the entire DPD setup."""
    num_ofdm_sym: int = 10
    iterations_per_sym: int = 200
    initial_coeffs: np.ndarray | None = None
    snr_db: float = 70
    signal_rms_dbp: float = -10

@dataclass(slots=True)
class IterationConfig:
    """Configuration for a single DPD iteration.

    Attributes:
        bypass: Module bypass flags (default: all modules active).
        calibrator: Stateful calibrator instance (holds coefficients and
                    estimation mode across iterations).
    """
    bypass: ModuleBypass = field(default_factory=ModuleBypass)
    calibrator: Calibrator = field(default_factory=Calibrator)
    setup: SetUpConfig = field(default_factory=SetUpConfig)
    temperature_celsius: float = 25.0
    iteration_index: int = 0
    iterations_per_sym: int = 200  # number of iterations per OFDM symbol (for tracking)

@dataclass(slots=True)
class IterationResult:
    """Result of a single DPD iteration."""
    status: str = "not_started"
    evm_db: float = 0.0
    mask_pass: bool = False
    siggen_out: np.ndarray = field(default_factory=lambda: np.array([], dtype=complex))
    prd_in_iq: np.ndarray = field(default_factory=lambda: np.array([], dtype=complex))
    pa_in_iq: np.ndarray = field(default_factory=lambda: np.array([], dtype=complex))
    pa_out_iq: np.ndarray = field(default_factory=lambda: np.array([], dtype=complex))
    rf_out_no_prd: np.ndarray = field(default_factory=lambda: np.array([], dtype=complex))
    dpod_out_iq: np.ndarray = field(default_factory=lambda: np.array([], dtype=complex))
    cal_coeffs: np.ndarray = field(default_factory=lambda: np.array([], dtype=complex))
    calibrator_state: Calibrator = field(default_factory=Calibrator)



# -----------------------------------------------------------------------------
# Main iteration entry point
# -----------------------------------------------------------------------------
def run_iteration(iter_config: IterationConfig | None = None) -> IterationResult:

    _FFT_SIZE = config.Config.FFT_SIZE
    _CP_LEN = config.Config.CP_LEN

    """Run one DPD iteration with hardcoded data flow.

    Data flow (fixed):
        siggen -> dpd -> pa_model -> capture ->
        -> calibrator -> metrics

    Args:
        config: Iteration configuration (bypass flags, calibration mode).
                Defaults to all modules active with LS estimation.

    Returns:
        IterationResult with status, EVM, mask pass, and IQ vectors.
    """
    if iter_config is None:
        # Keep stateful calibrator data (e.g., prev_ib_evm_db) across
        # repeated run_iteration() calls when caller does not pass config.
        default_cfg = getattr(run_iteration, "_default_config", None)
        if default_cfg is None:
            default_cfg = IterationConfig()
            setattr(run_iteration, "_default_config", default_cfg)
        cfg = default_cfg
    else:
        cfg = iter_config
    bp = cfg.bypass

    # --- Signal generation ---
    tx_iq = gen_lsig(bypass=bp.siggen, repeat_bits_every_call=False, bits_seed=123, signal_rms_dbp=cfg.setup.signal_rms_dbp)

    # --- AWGN channel ---
    tx_iq_with_awgn = apply_awgn(tx_iq, snr_db=cfg.setup.snr_db, bypass=bp.awgn)

    # --- Apply predistortion with current coefficients ---
    coeffs = np.asarray(cfg.calibrator.coeffs, dtype=complex).reshape(-1)
    tx_prd_out, _ = apply_predistortion(tx_iq_with_awgn, coeffs, bypass=bp.dprd)

    # --- PA / channel model ---
    rf_out = apply_pa_model(tx_prd_out, bypass=bp.pa_model)

    # --- PA without predistortion (for comparison) ---
    rf_out_no_prd = apply_pa_model(tx_iq_with_awgn, bypass=bp.pa_model) 
    
    # a try to to correct gain according to inband before estimation - the try didn't work well
    # =======================================================================================
    # before post distortion apply equalization on in-band portion of the signal
    # this will probaly is done on "real" product because the PA output need to be downconvert 
    # and time/phase aligned to the input signal
    rf_out_fd = np.fft.fftshift(np.fft.fft(rf_out[_CP_LEN//2:_CP_LEN//2+_FFT_SIZE]))[_FFT_SIZE//2-26:_FFT_SIZE//2-1]
    rf_out_fd_magdb = 20*np.log10(np.mean(np.abs(rf_out_fd)))
    prd_in_fd = np.fft.fftshift(np.fft.fft(tx_iq_with_awgn[_CP_LEN//2:_CP_LEN//2+_FFT_SIZE]))[_FFT_SIZE//2-26:_FFT_SIZE//2-1]
    prd_in_fd_magdb = 20*np.log10(np.mean(np.abs(prd_in_fd)))
    gain_correction = 1 # 10**((prd_in_fd_magdb - rf_out_fd_magdb)/20)
    # =======================================================================================

    # --- Apply post-distortion with current coefficients --- ---
    tx_pod_out, y_ordered = apply_predistortion(gain_correction*rf_out, coeffs, bypass=bp.dpod)


    # a try to to check inband EVM of previous iteration and decide to update coefficients or not
    # do not correct coefficient if no improvmant in EVM
    # =======================================================================================
    # before post distortion apply equalization on in-band portion of the signal
    # this will probaly is done on "real" product because the PA output need to be downconvert 
    # and time/phase aligned to the input signal
    rf_out_fd = np.fft.fftshift(np.fft.fft(rf_out[_CP_LEN//2:_CP_LEN//2+_FFT_SIZE]))[_FFT_SIZE//2-26:_FFT_SIZE//2+26]
    prd_in_fd = np.fft.fftshift(np.fft.fft(tx_iq_with_awgn[_CP_LEN//2:_CP_LEN//2+_FFT_SIZE]))[_FFT_SIZE//2-26:_FFT_SIZE//2+26]
    evm_ib_db = 20*np.log10(np.mean(np.abs(prd_in_fd-rf_out_fd))/np.mean(np.abs(prd_in_fd)))
    # if (evm_ib_db > cfg.calibrator.prev_ib_evm_db) & (cfg.iteration_index > 0):
    #     # update coefficients only if inband EVM improved
    #     cfg.calibrator.coeffs = cfg.calibrator.pre_coeffs
    # cfg.calibrator.prev_ib_evm_db = evm_ib_db
    # =======================================================================================

    # --- Update calibrator (LS or RLS) --- 
    # calibration works @640MHz rate, so use upsampled signal and PA input for estimation
    cfg.calibrator = cfg.calibrator.update(tx_prd_out, tx_pod_out, rf_out, y_ordered, bypass=bp.estimator)
    coeffs = np.asarray(cfg.calibrator.coeffs, dtype=complex).reshape(-1)


    # --- Metrics ---
    n = min(len(tx_iq), len(rf_out))
    evm = compute_evm(reference=tx_iq[:n], measured=rf_out[:n], bypass=bp.evm)
    mask_pass = check_relative_mask(measured=rf_out, bypass=bp.mask)

    status = "ok" if mask_pass else "needs_tuning"


    # # debug
    # x_prd = apply_predistortion(tx_iq, cfg.calibrator.coeffs, bypass=False)
    # Y_prd = apply_pa_model(x_prd, bypass=False)


    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(np.abs(tx_iq), label='Predistorted Input', color='blue')
    # plt.plot(np.abs(Y_prd), label='prd PA Output', color='orange')
    # plt.plot(np.abs(rf_out), label='PA Output', color='red')
    # plt.title('Magnitude Comparison')
    # plt.xlabel('Sample Index')
    # plt.ylabel('Magnitude')
    # plt.legend()
    # plt.subplot(1, 2, 2)
    # plt.plot(np.angle(tx_iq), label='Predistorted Input', color='blue')
    # plt.plot(np.angle(Y_prd), label='pfd PA Output', color='orange')
    # plt.plot(np.angle(rf_out), label='PA Output', color='red')
    # plt.title('Phase Comparison')
    # plt.xlabel('Sample Index')
    # plt.ylabel('Phase (radians)')
    # plt.legend()
    # plt.tight_layout()
    # # plt.show()













    return IterationResult(
        status=status,
        evm_db=evm,
        mask_pass=mask_pass,
        siggen_out=tx_iq,
        prd_in_iq=tx_iq_with_awgn,
        pa_in_iq=tx_prd_out,
        pa_out_iq=rf_out,
        rf_out_no_prd=rf_out_no_prd,
        dpod_out_iq=tx_pod_out,
        cal_coeffs=coeffs.copy(),
        calibrator_state=cfg.calibrator,
    )
