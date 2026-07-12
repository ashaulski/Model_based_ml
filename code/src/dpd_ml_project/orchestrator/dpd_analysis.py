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


# single iteration analysis function
def run_analysis(
    pipeline_result,
    iteration_config: IterationConfig,
    output_dir: Path,
) -> BypassAnalysisResult:


    # definitions for plotting and analysis
    _FFT_SIZE = config.Config.FFT_SIZE
    _CP_LEN = config.Config.CP_LEN
    _DATA_CARRIERS = config.Config.DATA_CARRIERS
    _data_carriers_index = [_carrier_to_index(k) for k in _DATA_CARRIERS]
    _pa_in_start_idx = _CP_LEN//2  # start data index at half of CP


    # capture time domain signals
    siggen_out_no_cp = np.array(pipeline_result.siggen_out[_CP_LEN:])
    dprd_in_no_cp = np.array(pipeline_result.prd_in_iq[_pa_in_start_idx:_pa_in_start_idx+_FFT_SIZE])
    pa_in_no_cp = np.array(pipeline_result.pa_in_iq[_pa_in_start_idx:_pa_in_start_idx+_FFT_SIZE])
    pa_out_no_cp = np.array(pipeline_result.pa_out_iq[_pa_in_start_idx:_pa_in_start_idx+_FFT_SIZE])
    dpod_in_out_cp = np.array(pipeline_result.dpod_out_iq[_pa_in_start_idx:_pa_in_start_idx+_FFT_SIZE])  # post-distorted signal (PA output)
 

    # visualize the signals in time domain
    fig0, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.plot(np.abs(dprd_in_no_cp), label='Predistorted Input', color='blue')
    ax1.plot(np.abs(pa_in_no_cp), label='PA Input', color='orange')
    ax1.plot(np.abs(pa_out_no_cp), label='PA Output', color='red')
    # ax1.plot(np.abs(dpod_in_out_cp), label='Post-Distorted Output', color='red')
    ax1.set_title('Magnitude Comparison')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Magnitude')
    ax1.legend()
    ax2.plot(np.angle(dprd_in_no_cp), label='Predistorted Input')
    ax2.plot(np.angle(pa_in_no_cp), label='PA Input')
    ax2.plot(np.angle(pa_out_no_cp), label='PA Output')
    # ax2.plot(np.angle(dpod_in_out_cp), label='Post-Distorted Output')
    ax2.set_title('Phase Comparison')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Phase (radians)')
    ax2.legend()
    mplcursors.cursor(hover=True)
    td_plot = output_dir / f"time_domain_{iteration_config.iteration_index}.png"
    fig0.savefig(td_plot, dpi=140)
    # plt.show()
    

    # convert to frequency domain
    # and calculate PSD
    siggen_out_fd = np.fft.fft(siggen_out_no_cp, n=_FFT_SIZE, norm="ortho")
    dprd_in_fd = np.fft.fft(dprd_in_no_cp, n=_FFT_SIZE, norm="ortho")
    pa_in_fd = np.fft.fft(pa_in_no_cp, n=_FFT_SIZE, norm="ortho")
    pa_out_fd = np.fft.fft(pa_out_no_cp, n=_FFT_SIZE, norm="ortho")
    evm_fd = np.fft.fft(dprd_in_no_cp-pa_out_no_cp, n=_FFT_SIZE, norm="ortho")  # error vector magnitude in frequency domain
    # Spectrum (dB) over all FFT bins
    eps = 1e-12
    siggen_fd_db = 20.0 * np.log10(np.abs(np.fft.fftshift(siggen_out_fd)) + eps)
    dprd_in_fd_db = 20.0 * np.log10(np.abs(np.fft.fftshift(dprd_in_fd)) + eps)
    pa_in_fd_db = 20.0 * np.log10(np.abs(np.fft.fftshift(pa_in_fd)) + eps)
    pa_out_fd_db = 20.0 * np.log10(np.abs(np.fft.fftshift(pa_out_fd)) + eps)
    evm_fd_db = 20.0 * np.log10(np.abs(evm_fd) + eps)
    bins_index = np.arange(-_FFT_SIZE // 2, _FFT_SIZE // 2)

    fig1, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(bins_index, siggen_fd_db, label="TX")
    ax1.plot(bins_index, dprd_in_fd_db, color="orange", label="TX at PD output")
    ax1.plot(bins_index, pa_out_fd_db, color="green", label="PA output")
    ax1.set_title("Spectrum (FFT bins)")
    ax1.set_xlabel("Subcarrier index (shifted)")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    spectrum_plot = output_dir / f"spectrum_{iteration_config.iteration_index}.png"
    fig1.tight_layout()
    mplcursors.cursor(hover=True)
    fig1.savefig(spectrum_plot, dpi=140)
    # plt.close(fig1)

    # EVM magnitude per FFT bin (full-bandwidth; subcarrier remapping is not
    # valid for an upsampled signal whose length is not an integer multiple of
    # the original 64-point FFT size)

    fig2, ax2 = plt.subplots(figsize=(9, 4))    
    ax2.plot(_DATA_CARRIERS, evm_fd_db[_data_carriers_index], color="red", label="EVM at used subcarriers")
    ax2.set_title(f"EVM [dB] at used subcarriers")
    ax2.set_xlabel("Subcarrier index")
    ax2.set_ylabel("EVM (dB)")
    ax2.grid(True, alpha=0.3)   
    evm_plot = output_dir / f"evm_{iteration_config.iteration_index}.png"
    fig2.tight_layout()
    mplcursors.cursor(hover=True)
    fig2.savefig(evm_plot, dpi=140)

    return BypassAnalysisResult(
        spectrum_plot=spectrum_plot,
        evm_plot=evm_plot,
        evm_db=np.asarray(evm_fd_db[_data_carriers_index], dtype=float),
        mean_evm_db=10*np.log10(np.mean(10**(evm_fd_db[_data_carriers_index]/20)) + eps),
    )



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


    # capture time domain signals
    for i, pipeline_results in enumerate(pipeline_results):
        siggen_out_no_cp = np.array(pipeline_results.siggen_out[_CP_LEN:])
        dprd_in_no_cp = np.array(pipeline_results.prd_in_iq[_pa_in_start_idx:_pa_in_start_idx+_FFT_SIZE])
        pa_in_no_cp = np.array(pipeline_results.pa_in_iq[_pa_in_start_idx:_pa_in_start_idx+_FFT_SIZE])
        pa_out_no_cp = np.array(pipeline_results.pa_out_iq[_pa_in_start_idx:_pa_in_start_idx+_FFT_SIZE])
        dpod_in_out_cp = np.array(pipeline_results.dpod_out_iq[_pa_in_start_idx:_pa_in_start_idx+_FFT_SIZE])  # post-distorted signal (PA output)
 

    # # -------------------------------------
    # # meters and plot section
    # # -------------------------------------

    # # plot frequency domain magnitude
    # test_data_rs  = test_data[:,:,_CP_LEN//2:_NFFT+_CP_LEN//2]
    # test_data_fd = np.fft.fftshift(np.fft.fft(test_data_rs,axis=2),axes=2)
    # test_data_magdb_fd = 20*np.log10((np.mean(np.abs(test_data_fd),axis=0)))
    # rf_out_rs = rf_out.reshape(num_ofdm_sym,-1)[:,_CP_LEN//2:_NFFT+_CP_LEN//2]
    # rf_out_fd = np.fft.fftshift(np.fft.fft(rf_out_rs,axis=1),axes=1)
    # rf_out_magdb_fd = 20*np.log10(np.mean(np.abs(rf_out_fd),axis=0))

    # plt.figure(figsize=(10, 5))
    # plt.plot(test_data_magdb_fd[0, :], label='Original signal', color='blue')
    # plt.plot(test_data_magdb_fd[1, :], label='PA Output without pre distortion', color='red')
    # plt.plot(rf_out_magdb_fd, label='PA Output with pre distortion', color='green')
    # plt.title('Magnitude Comparison')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(RUN_OUTPUT_DIR / "frequency_domain_comparison.png", dpi=160)

    # # calculate inband MSE comparison
    # test_data_ib_fd_no_predistorsion = test_data_fd[:,:,_NFFT//2-32:_NFFT//2+32]
    # mse_ib_no_predistorsion = np.mean(np.abs(test_data_ib_fd_no_predistorsion[:,0,:]-test_data_ib_fd_no_predistorsion[:,1,:])**2,axis=0)
    # evm_ib_db_no_predistorsion = 10*np.log10(mse_ib_no_predistorsion/np.mean(np.abs(test_data_ib_fd_no_predistorsion[:,0,:])**2,axis=0))

    # mse_ib_with_predistorsion = np.mean(np.abs(test_data_ib_fd_no_predistorsion[:,0,:]-rf_out_fd[:,_NFFT//2-32:_NFFT//2+32])**2,axis=0)
    # evm_ib_db_with_predistorsion = 10*np.log10(mse_ib_with_predistorsion/np.mean(np.abs(test_data_ib_fd_no_predistorsion[:,0,:])**2,axis=0))

    # # calculate time domain MSE comparison
    # mse_td_no_predistorsion = np.mean(np.abs(test_data_rs[:,0,_CP_LEN//2:-_CP_LEN//2]-test_data_rs[:,1,_CP_LEN//2:-_CP_LEN//2])**2,axis=(0,1))
    # mse_td_with_predistorsion = np.mean(np.abs(test_data_rs[:,0,_CP_LEN//2:-_CP_LEN//2]-rf_out_rs[:,_CP_LEN//2:-_CP_LEN//2])**2,axis=(0,1))
    # evm_td_no_predistorsion = 10*np.log10(mse_td_no_predistorsion/np.mean(np.abs(test_data_rs[:,0,_CP_LEN//2:-_CP_LEN//2])**2,axis=(0,1)))
    # evm_td_with_predistorsion = 10*np.log10(mse_td_with_predistorsion/np.mean(np.abs(test_data_rs[:,0,64:-64])**2,axis=(0,1)))

    # print("EVM time domain no DPD: ", evm_td_no_predistorsion)
    # print("EVM time domain with DPD: ", evm_td_with_predistorsion)


    # plt.figure(figsize=(10, 5))
    # plt.plot(evm_ib_db_no_predistorsion, label='EVM Inband no DPD', color='blue')
    # plt.plot(evm_ib_db_with_predistorsion, label='EVM Inband with DPD', color='red')
    # plt.title('Inband MSE Comparison')
    # plt.legend()
    # plt.show()


@dataclass(slots=True)
class BypassAnalysisResult:
    spectrum_plot: Path
    evm_plot: Path
    evm_db: np.ndarray
    mean_evm_db: float


def run_analysis(
    pipeline_result,
    iteration_config: IterationConfig,
    output_dir: Path,
) -> BypassAnalysisResult:


    # definitions for plotting and analysis
    _FFT_SIZE = config.Config.FFT_SIZE
    _CP_LEN = config.Config.CP_LEN
    _DATA_CARRIERS = config.Config.DATA_CARRIERS
    _data_carriers_index = [_carrier_to_index(k) for k in _DATA_CARRIERS]
    _pa_in_start_idx = _CP_LEN//2  # start data index at half of CP


    # capture time domain signals
    siggen_out_no_cp = np.array(pipeline_result.siggen_out[_CP_LEN:])
    dprd_in_no_cp = np.array(pipeline_result.prd_in_iq[_pa_in_start_idx:_pa_in_start_idx+_FFT_SIZE])
    pa_in_no_cp = np.array(pipeline_result.pa_in_iq[_pa_in_start_idx:_pa_in_start_idx+_FFT_SIZE])
    pa_out_no_cp = np.array(pipeline_result.pa_out_iq[_pa_in_start_idx:_pa_in_start_idx+_FFT_SIZE])
    dpod_in_out_cp = np.array(pipeline_result.dpod_out_iq[_pa_in_start_idx:_pa_in_start_idx+_FFT_SIZE])  # post-distorted signal (PA output)
 

    # visualize the signals in time domain
    fig0, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.plot(np.abs(dprd_in_no_cp), label='Predistorted Input', color='blue')
    ax1.plot(np.abs(pa_in_no_cp), label='PA Input', color='orange')
    ax1.plot(np.abs(pa_out_no_cp), label='PA Output', color='red')
    # ax1.plot(np.abs(dpod_in_out_cp), label='Post-Distorted Output', color='red')
    ax1.set_title('Magnitude Comparison')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Magnitude')
    ax1.legend()
    ax2.plot(np.angle(dprd_in_no_cp), label='Predistorted Input')
    ax2.plot(np.angle(pa_in_no_cp), label='PA Input')
    ax2.plot(np.angle(pa_out_no_cp), label='PA Output')
    # ax2.plot(np.angle(dpod_in_out_cp), label='Post-Distorted Output')
    ax2.set_title('Phase Comparison')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Phase (radians)')
    ax2.legend()
    mplcursors.cursor(hover=True)
    td_plot = output_dir / f"time_domain_{iteration_config.iteration_index}.png"
    fig0.savefig(td_plot, dpi=140)
    # plt.show()
    

    # convert to frequency domain
    # and calculate PSD
    siggen_out_fd = np.fft.fft(siggen_out_no_cp, n=_FFT_SIZE, norm="ortho")
    dprd_in_fd = np.fft.fft(dprd_in_no_cp, n=_FFT_SIZE, norm="ortho")
    pa_in_fd = np.fft.fft(pa_in_no_cp, n=_FFT_SIZE, norm="ortho")
    pa_out_fd = np.fft.fft(pa_out_no_cp, n=_FFT_SIZE, norm="ortho")
    evm_fd = np.fft.fft(dprd_in_no_cp-pa_out_no_cp, n=_FFT_SIZE, norm="ortho")  # error vector magnitude in frequency domain
    # Spectrum (dB) over all FFT bins
    eps = 1e-12
    siggen_fd_db = 20.0 * np.log10(np.abs(np.fft.fftshift(siggen_out_fd)) + eps)
    dprd_in_fd_db = 20.0 * np.log10(np.abs(np.fft.fftshift(dprd_in_fd)) + eps)
    pa_in_fd_db = 20.0 * np.log10(np.abs(np.fft.fftshift(pa_in_fd)) + eps)
    pa_out_fd_db = 20.0 * np.log10(np.abs(np.fft.fftshift(pa_out_fd)) + eps)
    evm_fd_db = 20.0 * np.log10(np.abs(evm_fd) + eps)
    bins_index = np.arange(-_FFT_SIZE // 2, _FFT_SIZE // 2)

    fig1, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(bins_index, siggen_fd_db, label="TX")
    ax1.plot(bins_index, dprd_in_fd_db, color="orange", label="TX at PD output")
    ax1.plot(bins_index, pa_out_fd_db, color="green", label="PA output")
    ax1.set_title("Spectrum (FFT bins)")
    ax1.set_xlabel("Subcarrier index (shifted)")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    spectrum_plot = output_dir / f"spectrum_{iteration_config.iteration_index}.png"
    fig1.tight_layout()
    mplcursors.cursor(hover=True)
    fig1.savefig(spectrum_plot, dpi=140)
    # plt.close(fig1)

    # EVM magnitude per FFT bin (full-bandwidth; subcarrier remapping is not
    # valid for an upsampled signal whose length is not an integer multiple of
    # the original 64-point FFT size)

    fig2, ax2 = plt.subplots(figsize=(9, 4))    
    ax2.plot(_DATA_CARRIERS, evm_fd_db[_data_carriers_index], color="red", label="EVM at used subcarriers")
    ax2.set_title(f"EVM [dB] at used subcarriers")
    ax2.set_xlabel("Subcarrier index")
    ax2.set_ylabel("EVM (dB)")
    ax2.grid(True, alpha=0.3)   
    evm_plot = output_dir / f"evm_{iteration_config.iteration_index}.png"
    fig2.tight_layout()
    mplcursors.cursor(hover=True)
    fig2.savefig(evm_plot, dpi=140)

    return BypassAnalysisResult(
        spectrum_plot=spectrum_plot,
        evm_plot=evm_plot,
        evm_db=np.asarray(evm_fd_db[_data_carriers_index], dtype=float),
        mean_evm_db=10*np.log10(np.mean(10**(evm_fd_db[_data_carriers_index]/20)) + eps),
    )


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

    dd = 1
    # mse_ib_no_predistorsion = np.mean(np.abs(test_data_ib_fd_no_predistorsion[:,0,:]-test_data_ib_fd_no_predistorsion[:,1,:])**2,axis=0)
    # evm_ib_db_no_predistorsion = 10*np.log10(mse_ib_no_predistorsion/np.mean(np.abs(test_data_ib_fd_no_predistorsion[:,0,:])**2,axis=0))

    # mse_ib_with_predistorsion = np.mean(np.abs(test_data_ib_fd_no_predistorsion[:,0,:]-rf_out_fd[:,_NFFT//2-32:_NFFT//2+32])**2,axis=0)
    # evm_ib_db_with_predistorsion = 10*np.log10(mse_ib_with_predistorsion/np.mean(np.abs(test_data_ib_fd_no_predistorsion[:,0,:])**2,axis=0))

    # # calculate time domain MSE comparison
    # mse_td_no_predistorsion = np.mean(np.abs(test_data_rs[:,0,_CP_LEN//2:-_CP_LEN//2]-test_data_rs[:,1,_CP_LEN//2:-_CP_LEN//2])**2,axis=(0,1))
    # mse_td_with_predistorsion = np.mean(np.abs(test_data_rs[:,0,_CP_LEN//2:-_CP_LEN//2]-rf_out_rs[:,_CP_LEN//2:-_CP_LEN//2])**2,axis=(0,1))
    # evm_td_no_predistorsion = 10*np.log10(mse_td_no_predistorsion/np.mean(np.abs(test_data_rs[:,0,_CP_LEN//2:-_CP_LEN//2])**2,axis=(0,1)))
    # evm_td_with_predistorsion = 10*np.log10(mse_td_with_predistorsion/np.mean(np.abs(test_data_rs[:,0,64:-64])**2,axis=(0,1)))

    # print("EVM time domain no DPD: ", evm_td_no_predistorsion)
    # print("EVM time domain with DPD: ", evm_td_with_predistorsion)


    # plt.figure(figsize=(10, 5))
    # plt.plot(evm_ib_db_no_predistorsion, label='EVM Inband no DPD', color='blue')
    # plt.plot(evm_ib_db_with_predistorsion, label='EVM Inband with DPD', color='red')
    # plt.title('Inband MSE Comparison')
    # plt.legend()
    # plt.show()