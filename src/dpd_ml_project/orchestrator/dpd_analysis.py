from dataclasses import dataclass
from pathlib import Path
import mplcursors

import matplotlib.pyplot as plt
import numpy as np

from dpd_ml_project import config
from dpd_ml_project.orchestrator.dpd_pipeline import IterationConfig, ModuleBypass, run_iteration
from dpd_ml_project.siggen.SigGen import _carrier_to_index


@dataclass(slots=True)
class BypassAnalysisResult:
    spectrum_plot: Path
    evm_plot: Path
    evm_db: np.ndarray
    mean_evm_db: float

# run analysis on entire set of iteration results
def run_analysis_all(
    pipeline_results,
    iteration_configs: IterationConfig,
    output_dir: Path,
) -> BypassAnalysisResult:


    # definitions for plotting and analysis
    _FFT_SIZE = config.Config.FFT_SIZE
    _CP_LEN = config.Config.CP_LEN
    _DATA_CARRIERS = config.Config.DATA_CARRIERS
    _data_carriers_index = [_carrier_to_index(k) for k in _DATA_CARRIERS]
    _pa_in_start_idx = _CP_LEN//2  # start data index at half of CP
    num_iter =len(pipeline_results)

    siggen_out_no_cp = np.empty((num_iter, _FFT_SIZE), dtype=complex)
    dprd_in_no_cp = np.empty((num_iter, _FFT_SIZE), dtype=complex)
    pa_in_no_cp = np.empty((num_iter, _FFT_SIZE), dtype=complex)
    pa_out_no_cp = np.empty((num_iter, _FFT_SIZE), dtype=complex)
    rf_out_no_prd_no_cp = np.empty((num_iter, _FFT_SIZE), dtype=complex)
    dpod_in_out_cp = np.empty((num_iter, _FFT_SIZE), dtype=complex)  # post-distorted signal (PA output)


    # capture time domain signals
    for i in range (num_iter):
        siggen_out_no_cp[i,:] = np.array(pipeline_results[i].siggen_out[_pa_in_start_idx:_pa_in_start_idx+_FFT_SIZE])
        dprd_in_no_cp[i,:] = np.array(pipeline_results[i].prd_in_iq[_pa_in_start_idx:_pa_in_start_idx+_FFT_SIZE])
        pa_in_no_cp[i,:] = np.array(pipeline_results[i].pa_in_iq[_pa_in_start_idx:_pa_in_start_idx+_FFT_SIZE])
        pa_out_no_cp[i,:] = np.array(pipeline_results[i].pa_out_iq[_pa_in_start_idx:_pa_in_start_idx+_FFT_SIZE])
        dpod_in_out_cp[i,:] = np.array(pipeline_results[i].dpod_out_iq[_pa_in_start_idx:_pa_in_start_idx+_FFT_SIZE])  # post-distorted signal (PA output)
        rf_out_no_prd_no_cp[i,:] = np.array(pipeline_results[i].rf_out_no_prd[_pa_in_start_idx:_pa_in_start_idx+_FFT_SIZE])
 




    # -------------------------------------
    # meters and plot section
    # -------------------------------------

    # plot frequency domain magnitude
    dprd_in_fd = np.fft.fftshift(np.fft.fft(dprd_in_no_cp,axis=1),axes=1)
    dprd_in_magdb = 20*np.log10((np.mean(np.abs(dprd_in_fd),axis=0)))
    dprd_in_ib_avg_magdb = np.mean(dprd_in_magdb[_FFT_SIZE//2-25:_FFT_SIZE//2-1])

    pa_in_fd = np.fft.fftshift(np.fft.fft(pa_in_no_cp,axis=1),axes=1)
    pa_in_magdb = 20*np.log10(np.abs(pa_in_fd))

    pa_out_fd = np.fft.fftshift(np.fft.fft(pa_out_no_cp,axis=1),axes=1)
    pa_out_magdb_fd = 20*np.log10(np.mean(np.abs(pa_out_fd[40:,:]), axis=0))

    pa_out_no_pred_fd = np.fft.fftshift(np.fft.fft(rf_out_no_prd_no_cp,axis=1),axes=1)
    pa_out_no_pred_magdb_fd = 20*np.log10(np.mean(np.abs(pa_out_no_pred_fd), axis=0))


    plt.figure(figsize=(10, 5))
    plt.plot(dprd_in_magdb-dprd_in_ib_avg_magdb, label='Original signal', color='blue')
    plt.plot(pa_out_no_pred_magdb_fd-dprd_in_ib_avg_magdb, label='PA Output without pre distortion (1\'st iteration)', color='red')
    plt.plot(pa_out_magdb_fd-dprd_in_ib_avg_magdb, label=f'PA Output with pre distortion ({num_iter} iteration)', color='green')
    plt.title('Magnitude Comparison')
    plt.legend()
    plt.tight_layout()
    # plt.savefig(RUN_OUTPUT_DIR / "frequency_domain_comparison.png", dpi=160)

    # # calculate inband MSE comparison
    evm_ib_db = np.empty((num_iter,), dtype=float)
    test_data_ib_fd_no_predistorsion = dprd_in_fd[:,_FFT_SIZE//2-32:_FFT_SIZE//2+32]
    pa_out_fd_ib = pa_out_fd[:,_FFT_SIZE//2-32:_FFT_SIZE//2+32]
    for i in range(num_iter):
        mse_ib = np.mean(np.abs(test_data_ib_fd_no_predistorsion[i,:]-pa_out_fd_ib[i,:])**2,axis=0)
        evm_ib_curr_db = 10*np.log10(mse_ib/np.mean(np.abs(test_data_ib_fd_no_predistorsion[i,:])**2,axis=0))
        print(f"Iteration {i+1}: EVM Inband: {evm_ib_curr_db:.2f} dB")
        evm_ib_db[i] = evm_ib_curr_db

    plt.figure(figsize=(10, 5))
    plt.plot(evm_ib_db, label='EVM Inband no DPD', color='blue')
    plt.title('Inband EVM trend over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('EVM (dB)')
    plt.legend()
    plt.tight_layout()

    # calculate time domain EVM
    evm_td_db = np.empty((num_iter,), dtype=float)
    for i in range(num_iter):
        mse_td = np.mean(np.abs(dprd_in_no_cp[i,:]-pa_out_no_cp[i,:])**2,axis=0)
        evm_td_curr_db = 10*np.log10(mse_td/np.mean(np.abs(dprd_in_no_cp[i,:])**2,axis=0))
        print(f"Iteration {i+1}: EVM Time Domain: {evm_td_curr_db:.2f} dB")
        evm_td_db[i] = evm_td_curr_db

    plt.figure(figsize=(10, 5))
    plt.plot(evm_td_db, label='EVM Time Domain no DPD', color='green')
    plt.title('Time Domain EVM trend over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('EVM (dB)')
    plt.legend()
    plt.tight_layout()

    plt.show()
