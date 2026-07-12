from pathlib import Path

from dpd_ml_project.cal.calibrator import Calibrator
from dpd_ml_project.orchestrator.dpd_analysis import run_analysis, run_analysis_all
from dpd_ml_project.orchestrator.dpd_pipeline import IterationConfig, IterationResult,  SetUpConfig, run_iteration

test_name = "test_cal_ewrls"

setup_config = SetUpConfig(
    num_ofdm_sym=50,
    iterations_per_sym=500,
    initial_coeffs=None,
    snr_db=70,
    signal_rms_dbp=-10,
)

itrs_config = [IterationConfig() for _ in range(setup_config.num_ofdm_sym)]
itrs_results = [IterationResult() for _ in range(setup_config.num_ofdm_sym)]
shared_calibrator = Calibrator()
shared_calibrator.mode = "rls"
shared_calibrator.iterations_per_sym = setup_config.iterations_per_sym


for i in range(setup_config.num_ofdm_sym):
    itrs_config[i].setup = setup_config
    itrs_config[i].iteration_index = i
    itrs_config[i].calibrator = shared_calibrator
    itrs_config[i].bypass.estimator = False
    itrs_config[i].bypass.siggen = False
    itrs_config[i].bypass.pa_model  = False
    itrs_config[i].bypass.awgn = False
    itrs_config[i].bypass.evm = False
    if i == 0:
        itrs_config[i].bypass.dpod = False
    if i >= 1:
        itrs_config[i].bypass.dprd = False      # activate pre/post-distortion at iteration 2 and onwards
        itrs_config[i].bypass.dpod = False


# run the iterations and store results
# ==============================
for i in range(setup_config.num_ofdm_sym):
    itrs_results[i] = run_iteration(itrs_config[i])

# run analysis on the iteration results and check outputs
# ==============================
output_dir = Path("tests") / "results" / test_name
output_dir.mkdir(parents=True, exist_ok=True)

run_analysis_all(itrs_results, itrs_config, output_dir=output_dir)

# for i in range(setup_config.num_ofdm_sym):
#     result = run_analysis(itrs_results[i], itrs_config[i], output_dir=output_dir)
#     print(f"Iteration {i+1}/{setup_config.num_ofdm_sym} - Status: {itrs_results[i].status}, EVM   : {itrs_results[i].evm_db:.2f} dB, Mask Pass: {itrs_results[i].mask_pass}")    
