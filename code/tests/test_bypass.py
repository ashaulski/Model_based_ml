from pathlib import Path
import tempfile
import numpy as np

from dpd_ml_project.orchestrator.dpd_analysis import run_analysis
from dpd_ml_project.orchestrator.dpd_pipeline import IterationConfig, ModuleBypass, run_iteration


bypass_all = ModuleBypass(
    siggen=False,
    estimator=True,
    dprd=True,
    dpod=True,
    pa_model=True,
    capture=False,
    evm=True,
    mask=True,
)
config = IterationConfig(bypass=bypass_all)
pipeline_result = run_iteration(config)
test_name = "test_bypass"

def test_analysis():
    output_dir = Path("tests") / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_analysis(pipeline_result, output_dir, test_name=test_name)
    assert result.spectrum_plot.exists(), "Spectrum plot not created"
    assert result.evm_plot.exists(), "EVM plot not created"
    print(f"Spectrum figure: {result.spectrum_plot}")
    print(f"EVM figure: {result.evm_plot}")
