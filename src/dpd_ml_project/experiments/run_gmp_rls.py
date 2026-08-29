"""Driver: run the GMP + RLS architecture through the generic pipeline.

Run: python -m dpd_ml_project.experiments.run_gmp_rls
"""
from __future__ import annotations

import numpy as np

from dpd_ml_project.analysis.analyze import analyze_run
from dpd_ml_project.core.common_config import SimConfig
from dpd_ml_project.core.registry import build_dpd
from dpd_ml_project.orchestrator.pipeline import StepResult, run_step


def run(num_iters: int = 40, seed: int = 1234) -> list[StepResult]:
    """Run ``num_iters`` GMP+RLS passes; return the per-step results."""
    np.random.seed(seed)
    sim = SimConfig(signal_rms_dbp=-10.0, snr_db=70.0)
    model, adapter = build_dpd({"arch": "gmp", "adapter": "rls", "lambda_rls": 0.999})

    results: list[StepResult] = []
    for i in range(num_iters):
        results.append(run_step(model, adapter, sim, apply_dpd=True, adapt=True,
                                bits_seed=123, iteration_index=i,
                                capture_no_dpd_ref=True))
    return results


def main(analyze: bool = True) -> None:
    results = run()
    evms = [r.evm_db for r in results]
    for i, evm in enumerate(evms):
        if i < 5 or i % 10 == 0 or i == len(evms) - 1:
            print(f"iter {i:3d}  EVM = {evm:7.2f} dB")
    print("-" * 40)
    print(f"first EVM = {evms[0]:7.2f} dB")
    print(f"best  EVM = {min(evms):7.2f} dB")
    print(f"last  EVM = {evms[-1]:7.2f} dB")

    if analyze:
        analyze_run(results, show=True)


if __name__ == "__main__":
    main()
