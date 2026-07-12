from pathlib import Path

from dpd_ml_project.orchestrator.dpd_analysis import run_analysis
from dpd_ml_project.orchestrator.dpd_pipeline import IterationConfig, ModuleBypass, SetUpConfig, run_iteration


bypass_all = ModuleBypass(
    siggen=False,
    estimator=True,
    dprd=False,
    dpod=True,
    pa_model=False,
    capture=True,
    evm=True,
    mask=True,
    awgn=False,
)

setup_config = SetUpConfig(
    num_iterations=1,
    initial_coeffs=None,
    snr_db=70,
    signal_rms_dbp=-20,
)

itr_config = IterationConfig(bypass=bypass_all, setup=setup_config)
pipeline_result = run_iteration(itr_config)
test_name = "test_dpd"

def test_analysis():
    output_dir = Path("tests") / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = run_analysis(pipeline_result, output_dir, test_name=test_name)
    assert result.spectrum_plot.exists(), "Spectrum plot not created"
    assert isinstance(result.evm_db, list)
    assert isinstance(result.mean_evm_db, float)
    print(f"Spectrum figure: {result.spectrum_plot}")
    print(f"Mean EVM (dB): {result.mean_evm_db:.9f}")