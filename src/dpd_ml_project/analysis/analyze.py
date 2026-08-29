"""Analysis helpers for the modular DPD pipeline.

Modular port of the legacy ``orchestrator.dpd_analysis.run_analysis_all``:
produces the magnitude-spectrum comparison and the in-band / time-domain EVM
trends from a list of ``StepResult`` (which carry a ``Capture`` each), decoupled
from the legacy ``config.Config`` / ``IterationResult`` stack.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from dpd_ml_project.core.common_config import SimConfig


@dataclass
class AnalysisResult:
    """Aggregated EVM trends over a run and optional saved-plot paths."""
    evm_ib_db: np.ndarray
    evm_td_db: np.ndarray
    mean_evm_ib_db: float
    mean_evm_td_db: float
    spectrum_png: Path | None = None
    evm_ib_png: Path | None = None
    evm_td_png: Path | None = None


def analyze_run(
    results: Sequence[Any],
    *,
    converged_from: int = 40,
    output_dir: Path | None = None,
    show: bool = True,
) -> AnalysisResult:
    """Plot spectra and EVM trends for a sequence of pipeline steps.

    Args:
        results: sequence of ``StepResult`` whose captures were produced with
                 ``run_step(..., capture_no_dpd_ref=True)``.
        converged_from: first iteration index to average for the "with-DPD" PA
                        spectrum (legacy used a fixed 40); clamped to the run.
        output_dir: where PNGs would be written (saving is currently disabled).
        show: call ``plt.show()`` at the end.
    """
    _FFT = SimConfig.FFT_SIZE
    _CP = SimConfig.CP_LEN
    start = _CP // 2                      # skip half the CP
    num_iter = len(results)
    if num_iter == 0:
        raise ValueError("results is empty - nothing to analyze")

    captures = [r.capture for r in results]
    if any(c.rf_out_no_prd is None for c in captures):
        raise ValueError(
            "captures lack rf_out_no_prd; run the pipeline with "
            "run_step(..., capture_no_dpd_ref=True)"
        )

    def _window(sig: np.ndarray) -> np.ndarray:
        return np.asarray(sig[start:start + _FFT], dtype=complex)

    dprd_in = np.stack([_window(c.dpd_in) for c in captures])
    pa_out = np.stack([_window(c.pa_out) for c in captures])
    pa_out_no_prd = np.stack([_window(c.rf_out_no_prd) for c in captures])

    # clamp the convergence window so the average is never over an empty slice
    cf = converged_from if 0 <= converged_from < num_iter else 0

    # ---- frequency-domain magnitude comparison ----
    dprd_in_fd = np.fft.fftshift(np.fft.fft(dprd_in, axis=1), axes=1)
    dprd_in_magdb = 20 * np.log10(np.mean(np.abs(dprd_in_fd), axis=0))
    dprd_in_ib_avg_magdb = np.mean(dprd_in_magdb[_FFT // 2 - 25:_FFT // 2 - 1])

    pa_out_fd = np.fft.fftshift(np.fft.fft(pa_out, axis=1), axes=1)
    pa_out_magdb_fd = 20 * np.log10(np.mean(np.abs(pa_out_fd[cf:, :]), axis=0))

    pa_out_no_pred_fd = np.fft.fftshift(np.fft.fft(pa_out_no_prd, axis=1), axes=1)
    pa_out_no_pred_magdb_fd = 20 * np.log10(np.mean(np.abs(pa_out_no_pred_fd), axis=0))

    plt.figure(figsize=(10, 5))
    plt.plot(dprd_in_magdb - dprd_in_ib_avg_magdb, label='Original signal', color='blue')
    plt.plot(pa_out_no_pred_magdb_fd - dprd_in_ib_avg_magdb,
             label="PA Output without pre distortion (1'st iteration)", color='red')
    plt.plot(pa_out_magdb_fd - dprd_in_ib_avg_magdb,
             label=f'PA Output with pre distortion ({num_iter} iteration)', color='green')
    plt.title('Magnitude Comparison')
    plt.legend()
    plt.tight_layout()
    spectrum_png = None
    # if output_dir is not None:
    #     spectrum_png = Path(output_dir) / "frequency_domain_comparison.png"
    #     plt.savefig(spectrum_png, dpi=160)

    # ---- in-band EVM trend ----
    evm_ib_db = np.empty(num_iter, dtype=float)
    ib = slice(_FFT // 2 - 32, _FFT // 2 + 32)
    ref_ib = dprd_in_fd[:, ib]
    pa_out_ib = pa_out_fd[:, ib]
    for i in range(num_iter):
        mse_ib = np.mean(np.abs(ref_ib[i, :] - pa_out_ib[i, :]) ** 2)
        evm_ib_db[i] = 10 * np.log10(mse_ib / np.mean(np.abs(ref_ib[i, :]) ** 2))
        print(f"Iteration {i + 1}: EVM Inband: {evm_ib_db[i]:.2f} dB")

    plt.figure(figsize=(10, 5))
    plt.plot(evm_ib_db, label='EVM Inband', color='blue')
    plt.title('Inband EVM trend over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('EVM (dB)')
    plt.legend()
    plt.tight_layout()
    evm_ib_png = None
    # if output_dir is not None:
    #     evm_ib_png = Path(output_dir) / "evm_inband_trend.png"
    #     plt.savefig(evm_ib_png, dpi=160)

    # ---- time-domain EVM trend ----
    evm_td_db = np.empty(num_iter, dtype=float)
    for i in range(num_iter):
        mse_td = np.mean(np.abs(dprd_in[i, :] - pa_out[i, :]) ** 2)
        evm_td_db[i] = 10 * np.log10(mse_td / np.mean(np.abs(dprd_in[i, :]) ** 2))
        print(f"Iteration {i + 1}: EVM Time Domain: {evm_td_db[i]:.2f} dB")

    plt.figure(figsize=(10, 5))
    plt.plot(evm_td_db, label='EVM Time Domain', color='green')
    plt.title('Time Domain EVM trend over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('EVM (dB)')
    plt.legend()
    plt.tight_layout()
    evm_td_png = None
    # if output_dir is not None:
    #     evm_td_png = Path(output_dir) / "evm_timedomain_trend.png"
    #     plt.savefig(evm_td_png, dpi=160)

    if show:
        plt.show()

    return AnalysisResult(
        evm_ib_db=evm_ib_db,
        evm_td_db=evm_td_db,
        mean_evm_ib_db=float(np.mean(evm_ib_db)),
        mean_evm_td_db=float(np.mean(evm_td_db)),
        spectrum_png=spectrum_png,
        evm_ib_png=evm_ib_png,
        evm_td_png=evm_td_png,
    )
