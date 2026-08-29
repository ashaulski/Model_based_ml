"""Generic, architecture-agnostic DPD pipeline.

One pass: siggen -> awgn -> predistort -> PA -> capture -> adapt -> metrics.
The predistorter and adapter are injected, so any architecture runs unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dpd_ml_project.channel.awgn import apply_awgn
from dpd_ml_project.channel.pa_model import apply_pa_model
from dpd_ml_project.core.capture import Capture
from dpd_ml_project.core.common_config import SimConfig
from dpd_ml_project.metrics.evm import compute_evm
from dpd_ml_project.siggen.SigGen import gen_lsig


@dataclass
class StepResult:
    """Outcome of one pipeline pass."""
    capture: Capture
    evm_db: float
    coeffs: np.ndarray | None = None
    meta: dict = field(default_factory=dict)


def run_step(
    model: Any,
    adapter: Any,
    sim: SimConfig,
    *,
    apply_dpd: bool = True,
    adapt: bool = True,
    bits_seed: int = 123,
    iteration_index: int = 0,
    capture_no_dpd_ref: bool = False,
) -> StepResult:
    """Run one DPD iteration with an injected predistorter and adapter."""
    tx_iq = gen_lsig(
        bypass=False,
        signal_rms_dbp=sim.signal_rms_dbp,
        repeat_bits_every_call=False,
        bits_seed=bits_seed,
    )
    dpd_in = apply_awgn(tx_iq, snr_db=sim.snr_db, bypass=False)
    pa_in = model.predistort(dpd_in) if apply_dpd else dpd_in.copy()
    pa_out = apply_pa_model(pa_in, bypass=False)

    # PA output of the un-predistorted signal, used as a reference in analysis.
    rf_out_no_prd = apply_pa_model(dpd_in, bypass=False) if capture_no_dpd_ref else None

    capture = Capture(
        x_in=tx_iq,
        dpd_in=dpd_in,
        pa_in=pa_in,
        pa_out=pa_out,
        rf_out_no_prd=rf_out_no_prd,
        meta={"iteration_index": iteration_index, "snr_db": sim.snr_db},
    )

    if adapt:
        adapter.update(model, capture)

    n = min(len(tx_iq), len(pa_out))
    evm_db = compute_evm(reference=tx_iq[:n], measured=pa_out[:n], bypass=False)

    coeffs = getattr(model, "coeffs", None)
    return StepResult(
        capture=capture,
        evm_db=evm_db,
        coeffs=None if coeffs is None else np.asarray(coeffs).copy(),
        meta={"iteration_index": iteration_index},
    )
