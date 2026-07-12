from pathlib import Path

from dpd_ml_project.cal.calibrator import Calibrator
from dpd_ml_project.orchestrator.dpd_analysis import run_analysis
from dpd_ml_project.orchestrator.dpd_pipeline import IterationConfig, IterationResult,  SetUpConfig, run_iteration

test_name = "test_cal_kal"

setup_config = SetUpConfig(
    num_iterations=10,
    initial_coeffs=None,
    snr_db=70,
    signal_rms_dbp=-10,
)

itrs_config = [IterationConfig() for _ in range(setup_config.num_iterations)]
itrs_results = [IterationResult() for _ in range(setup_config.num_iterations)]
shared_calibrator = Calibrator()


for i in range(setup_config.num_iterations):
    itrs_config[i].setup = setup_config
    itrs_config[i].iteration_index = i
    itrs_config[i].calibrator = shared_calibrator
    itrs_config[i].bypass.estimator = False
    itrs_config[i].bypass.siggen = False
    itrs_config[i].bypass.pa_model  = False
    itrs_config[i].bypass.awgn = False
    if i >= 1:
        itrs_config[i].bypass.dprd = False      # activate pre/post-distortion at iteration 2 and onwards
        itrs_config[i].bypass.dpod = False


# run the iterations and store results
# ==============================
for i in range(setup_config.num_iterations):
    shared_calibrator.mode = "ls" if i == 0 else "rls"
    itrs_results[i] = run_iteration(itrs_config[i])

# run analysis on the iteration results and check outputs
# ==============================
output_dir = Path("tests") / "results" / test_name
output_dir.mkdir(parents=True, exist_ok=True)

for i in range(setup_config.num_iterations):
    result = run_analysis(itrs_results[i], itrs_config[i], output_dir=output_dir)
    print(f"Iteration {i+1}/{setup_config.num_iterations} - Status: {itrs_results[i].status}, EVM   : {itrs_results[i].evm_percent:.2f}%, Mask Pass: {itrs_results[i].mask_pass}")    
