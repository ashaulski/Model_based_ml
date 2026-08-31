"""Driver: train a VDTDNN / AVDTDNN inverse model, then run it as a predistorter.

Follows the paper's indirect-learning setup: the model is fit offline on clean
(pa_out -> pa_in) pairs, monitored on a held-out validation set with a two-stage
Adam step size and early stopping, then evaluated closed-loop on a test set.

Run: python -m dpd_ml_project.experiments.run_vdtdnn
"""
from __future__ import annotations

import numpy as np
import torch

from dpd_ml_project.analysis.analyze import analyze_run
from dpd_ml_project.analysis.training_curve import plot_training_curve
from dpd_ml_project.core.common_config import SimConfig
from dpd_ml_project.core.per_arc_config import VdtdnnConfig
from dpd_ml_project.core.registry import build_dpd
from dpd_ml_project.metrics.nmse import compute_nmse
from dpd_ml_project.orchestrator.pipeline import StepResult, run_step


def _make_captures(model, adapter, sim: SimConfig, num: int, seed0: int) -> list:
    # when creting the training/validation sets, we don't apply DPD and don't adapt the model
    # because we want to capture clean input/output pairs for training the inverse model
    return [run_step(model, adapter, sim, apply_dpd=False, adapt=False,
                     bits_seed=seed0 + i, capture_no_dpd_ref=False).capture
            for i in range(num)]


def eval_nmse(model, captures) -> float:
    """Mean inverse-model NMSE (dB): predict pa_in from pa_out."""
    return float(np.mean([
        compute_nmse(reference=cap.pa_in, measured=model.predistort(cap.pa_out))
        for cap in captures
    ]))


def run(
    num_train: int = 40,
    num_val: int = 20,
    num_test: int = 40,
    cfg: VdtdnnConfig | None = None,
    augmented: bool = True,
    seed: int = 1234,
    verbose: bool = True,
) -> tuple[list[StepResult], list[dict]]:
    """Train, validate, and closed-loop test the model.

    Returns ``(test_results, history)``.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    sim = SimConfig(signal_rms_dbp=-7.0, snr_db=70.0)

    cfg = cfg or VdtdnnConfig(augmented=augmented)
    arch = "avdtdnn" if cfg.augmented else "vdtdnn"
    model, adapter = build_dpd({"arch": arch, "adapter": "vdtdnn_sgd", "vdtdnn_config": cfg})

    if verbose:
        print(f"{arch.upper()}  M={cfg.memory_depth}  G={cfg.num_neurons}  "
              f"act={cfg.activation}  coeffs={model.num_coefficients()} "
              f"(formula {cfg.num_coefficients})")

    # Disjoint symbol sets: the signal generator draws fresh bits per call.
    train_caps = _make_captures(model, adapter, sim, num_train, seed0=1000)
    val_caps = _make_captures(model, adapter, sim, num_val, seed0=4000)

    history: list[dict] = []
    best_val = float("inf")
    best_state = model.get_state()
    epochs_since_best = 0
    switch_epoch = int(cfg.num_epochs * cfg.lr_switch_frac)

    for epoch in range(1, cfg.num_epochs + 1):
        adapter.set_lr(cfg.lr if epoch <= switch_epoch else cfg.lr_final)
        for cap in train_caps:
            adapter.update(model, cap)

        # compute epoch's train/val NMSE (dB) and record the history
        train_nmse = eval_nmse(model, train_caps)
        val_nmse = eval_nmse(model, val_caps)
        history.append({
            "epoch": epoch, "train_nmse_db": train_nmse, "val_nmse_db": val_nmse,
            "lr": adapter.lr, "batch_mse": adapter.last_loss,
        })

        # save the best model state (lowest val NMSE) and reset the early-stopping counter
        if val_nmse < best_val - 1e-3:
            best_val, best_state, epochs_since_best = val_nmse, model.get_state(), 0
        else:
            epochs_since_best += 1

        if verbose and (epoch <= 5 or epoch % 10 == 0 or epoch == cfg.num_epochs):
            print(f"epoch {epoch:3d}/{cfg.num_epochs}  lr={adapter.lr:.1e}  "
                  f"train NMSE = {train_nmse:7.2f} dB   val NMSE = {val_nmse:7.2f} dB")

        if epochs_since_best >= cfg.patience:
            if verbose:
                print(f"early stop at epoch {epoch} (best val NMSE = {best_val:.2f} dB)")
            break

    model.set_state(best_state)

    test_results = [
        run_step(model, adapter, sim, apply_dpd=True, adapt=False,
                 bits_seed=7000 + i, iteration_index=i, capture_no_dpd_ref=True)
        for i in range(num_test)
    ]
    return test_results, history


def main(analyze: bool = True) -> None:
    test_results, history = run()
    evms = [r.evm_db for r in test_results]
    print("-" * 40)
    print(f"best val NMSE  = {min(h['val_nmse_db'] for h in history):7.2f} dB")
    print(f"test EVM: best = {min(evms):7.2f} dB   mean = {np.mean(evms):7.2f} dB")

    if analyze:
        plot_training_curve(history, title="VDTDNN training curve", show=True)
        analyze_run(test_results, show=True)


if __name__ == "__main__":
    main()
