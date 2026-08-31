"""Driver: train the GRU inverse model, then run it as a predistorter.

The GRU is fit offline on clean input/output pairs (``apply_dpd=False`` so that
``pa_in == dpd_in`` and ``pa_out == PA(dpd_in)``), reproducing the legacy GRU
training. It is then evaluated closed-loop (``apply_dpd=True``) and analyzed.

Run: python -m dpd_ml_project.experiments.run_gru
"""
from __future__ import annotations

import numpy as np
import torch

from dpd_ml_project.analysis.analyze import analyze_run
from dpd_ml_project.core.common_config import SimConfig
from dpd_ml_project.core.registry import build_dpd
from dpd_ml_project.orchestrator.pipeline import StepResult, run_step


def _make_capture(model, adapter, sim: SimConfig, seed: int) -> StepResult:
    """One un-predistorted pass -> a clean (pa_in == dpd_in, pa_out) training pair."""
    return run_step(model, adapter, sim, apply_dpd=False, adapt=False,
                    bits_seed=seed, capture_no_dpd_ref=False)


def run(num_train: int = 40, num_eval: int = 20, epochs: int = 15,
        seed: int = 1234) -> list[StepResult]:
    """Train the GRU on ``num_train`` symbols, then return ``num_eval`` eval steps."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    sim = SimConfig(signal_rms_dbp=-7.0, snr_db=70.0)
    model, adapter = build_dpd({"arch": "gru", "adapter": "gru_sgd", "lr": 2e-4})

    # run "num_train" OFDM symbols through data path: sig_gen -> awgn -> predistort -> PA -> capture
    # get a list of "num_train" with dimension (num_train x num_samples_per_symbol)
    # for each point on data path
    train_caps = [_make_capture(model, adapter, sim, seed=1000 + i).capture
                  for i in range(num_train)]
    # 
    for epoch in range(epochs):
        for cap in train_caps:
            adapter.update(model, cap)
        print(f"epoch {epoch + 1:2d}/{epochs}  train MSE = {adapter.last_loss:.3e}")

    # --- closed-loop evaluation with the trained predistorter ---
    results: list[StepResult] = []
    for i in range(num_eval):
        results.append(run_step(model, adapter, sim, apply_dpd=True, adapt=False,
                                bits_seed=2000 + i, iteration_index=i,
                                capture_no_dpd_ref=True))
    return results


def main(analyze: bool = True) -> None:
    results = run()
    evms = [r.evm_db for r in results]
    print("-" * 40)
    print(f"eval EVM: first = {evms[0]:7.2f} dB  best = {min(evms):7.2f} dB  "
          f"mean = {np.mean(evms):7.2f} dB")

    if analyze:
        analyze_run(results, show=True)


if __name__ == "__main__":
    main()
