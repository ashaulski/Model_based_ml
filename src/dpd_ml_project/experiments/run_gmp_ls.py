"""Driver: run the GMP + LS architecture through the generic pipeline.

Run: python -m dpd_ml_project.experiments.run_gmp_ls
"""
from __future__ import annotations

import numpy as np

from dpd_ml_project.core.common_config import SimConfig
from dpd_ml_project.core.registry import build_dpd
from dpd_ml_project.orchestrator.pipeline import run_step


def run(num_iters: int = 50, seed: int = 1234) -> list[float]:
    """Run ``num_iters`` GMP+LS passes; return the EVM (dB) trajectory."""
    np.random.seed(seed)
    sim = SimConfig(signal_rms_dbp=-8.0, snr_db=70.0)
    model, adapter = build_dpd({"arch": "gmp", "adapter": "ls"})

    evms: list[float] = []
    for i in range(num_iters):
        result = run_step(model, adapter, sim, apply_dpd=True, adapt=True,
                          bits_seed=123, iteration_index=i)
        evms.append(result.evm_db)
    return evms


def main() -> None:
    evms = run()
    for i, evm in enumerate(evms):
        if i < 5 or i % 10 == 0 or i == len(evms) - 1:
            print(f"iter {i:3d}  EVM = {evm:7.2f} dB")
    print("-" * 40)
    print(f"first EVM = {evms[0]:7.2f} dB")
    print(f"best  EVM = {min(evms):7.2f} dB")
    print(f"last  EVM = {evms[-1]:7.2f} dB")


if __name__ == "__main__":
    main()
