"""Driver: RLS-assisted CNN DPD trained by indirect learning (ILA).

Counterpart of ``run_hybrid_cnn_rls`` that needs no differentiable PA. The CNN
is fit as a *post-inverse*: capture ``(pa_in, pa_out)`` pairs with the canonical
NumPy PA, then regress ``cnn([pa_out | RLS_post(pa_out)]) -> pa_in``. The
learned post-inverse is deployed unchanged as the predistorter (classic ILA
copy). Validation and checkpoint selection stay spectral-first on the NumPy PA.

Trade-off vs the direct driver: no PA gradients are required, but the loss is
inverse-model MSE rather than the true spectral objective, and training pairs
come from un-predistorted captures (ILA distribution shift).

Run: python -m dpd_ml_project.experiments.run_hybrid_cnn_rls_indirect
"""
from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from dpd_ml_project.channel.pa_model import apply_pa_model
from dpd_ml_project.experiments.run_hybrid_cnn_rls import (
    DpdCnn,
    HybridCnnConfig,
    align_complex_waveform_numpy,
    build_datasets,
    build_model_complexity_summary,
    build_rls_baseline,
    compute_epoch_spectral_comparison_numpy,
    device,
    evaluate_held_out,
    save_model_checkpoint,
    save_training_checkpoint,
    to_builtin_value,
    update_spectral_learning_rate,
)
from dpd_ml_project.predistorters.gmp import GmpPredistorter


# -----------------------------------------------------------------------------
# Inverse-training data (post-inverse pairs, NumPy PA only)
# -----------------------------------------------------------------------------
class InversePairBatcher:
    """Yields (pa_in_iq, [pa_out_iq | rls_post_iq]) batches of whole symbols."""

    def __init__(self, pa_in_symbols, baseline_model: GmpPredistorter,
                 symbols_per_batch=16, shuffle=False, seed=0) -> None:
        pa_in = np.asarray(pa_in_symbols, dtype=np.complex64)
        pa_out = np.stack([np.asarray(apply_pa_model(s, bypass=False), dtype=np.complex64)
                           for s in pa_in])
        # RLS applied as a post-distorter on the capture, mirroring deployment features.
        rls_post = np.stack([np.asarray(baseline_model.predistort(s), dtype=np.complex64)
                             for s in pa_out])
        self.targets = torch.as_tensor(
            np.stack((pa_in.real, pa_in.imag), axis=-1), dtype=torch.float32)
        pa_out_iq = np.stack((pa_out.real, pa_out.imag), axis=-1)
        rls_post_iq = np.stack((rls_post.real, rls_post.imag), axis=-1)
        self.features = torch.as_tensor(
            np.concatenate((pa_out_iq, rls_post_iq), axis=-1), dtype=torch.float32)
        self.symbols_per_batch = max(1, int(symbols_per_batch))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.iteration = 0

    def __iter__(self):
        indices = torch.arange(len(self.targets))
        if self.shuffle:
            g = torch.Generator().manual_seed(self.seed + self.iteration)
            indices = indices[torch.randperm(len(indices), generator=g)]
        self.iteration += 1
        for start in range(0, len(indices), self.symbols_per_batch):
            batch = indices[start:start + self.symbols_per_batch]
            yield self.targets[batch], self.features[batch]


def inverse_fit_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Signal-power-normalized MSE (linear EVM squared) of the reconstructed PA input."""
    error_power = torch.mean(torch.sum((pred - target) ** 2, dim=-1))
    signal_power = torch.mean(torch.sum(target ** 2, dim=-1))
    return error_power / torch.clamp(signal_power, min=1e-12)


def numpy_closed_loop_evm_db(model, loader) -> float:
    """Deployment-mode time-domain EVM of PA(cnn(x)) vs x, using the NumPy PA."""
    was_training = model.training
    model.eval()
    error_power = signal_power = 0.0
    with torch.no_grad():
        for target_iq, features in loader:
            pred, _ = model(features.to(device))
            pred_c = torch.complex(pred[..., 0], pred[..., 1]).cpu().numpy()
            refs = torch.complex(target_iq[..., 0], target_iq[..., 1]).numpy()
            for ref, pd in zip(refs, pred_c):
                pa_out = align_complex_waveform_numpy(ref, apply_pa_model(pd, bypass=False))
                error_power += float(np.sum(np.abs(pa_out - ref) ** 2))
                signal_power += float(np.sum(np.abs(ref) ** 2))
    if was_training:
        model.train()
    return 20.0 * np.log10(max(np.sqrt(error_power / max(signal_power, 1e-24)), 1e-12))


# -----------------------------------------------------------------------------
# Training (inverse fit; closed-loop NumPy validation gates the checkpoint)
# -----------------------------------------------------------------------------
def train_indirect(model, optimizer, inverse_loader, valid_loader, cfg: HybridCnnConfig,
                   checkpoint_path: Path | None = None):
    if cfg.final_lr <= 0 or cfg.lr_patience < 1 or cfg.early_stopping_patience < 1:
        raise ValueError("Learning rate and patience must be positive.")
    if any(group["lr"] < cfg.final_lr for group in optimizer.param_groups):
        raise ValueError("Final LR must not exceed the initial LR.")

    history: dict[str, list] = {k: [] for k in (
        "learning_rate", "inverse_loss", "valid_evm", "valid_acpr_db", "valid_oob_db",
        "worst_acpr_delta_db", "worst_oob_delta_db")}

    spectral_improvement_margin_db = 0.1
    spectral_plateau_min_delta_db = 0.01
    model = model.to(device)

    baseline_valid = compute_epoch_spectral_comparison_numpy(model, valid_loader)
    baseline_valid_acpr_db = baseline_valid["rls_acpr_db"]
    baseline_valid_oob_db = baseline_valid["rls_oob_db"]
    best_val_evm_db = numpy_closed_loop_evm_db(model, valid_loader)
    best_epoch = 0
    best_state_dict = copy.deepcopy(model.state_dict())
    best_spectral_score = 0.0
    best_observed_spectral_score = float("inf")
    epochs_without_improvement = 0

    print(f"Indirect training target: max {cfg.max_epochs} epochs (inverse-fit MSE loss)")
    print("Canonical NumPy RLS validation baseline: ACPR %.2f dBc; far OOB %.2f dBc"
          % (baseline_valid_acpr_db, baseline_valid_oob_db))

    for epoch in range(cfg.max_epochs):
        model.train()
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))
        loss_epoch, batch_count = 0.0, 0

        for batch_idx, (target_iq, features) in enumerate(inverse_loader):
            optimizer.zero_grad()
            pred, _ = model(features.to(device))
            loss = inverse_fit_loss(pred, target_iq.to(device))
            if not np.isfinite(float(loss.detach().item())):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch + 1}, batch {batch_idx + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_epoch += loss.item()
            batch_count += 1

        valid_evm_epoch = numpy_closed_loop_evm_db(model, valid_loader)
        valid_cmp = compute_epoch_spectral_comparison_numpy(model, valid_loader)
        valid_acpr_db, valid_oob_db = valid_cmp["cnn_acpr_db"], valid_cmp["cnn_oob_db"]
        worst_acpr_delta = valid_cmp["worst_acpr_delta_db"]
        worst_oob_delta = valid_cmp["worst_oob_delta_db"]

        history["inverse_loss"].append(loss_epoch / max(batch_count, 1))
        history["valid_evm"].append(valid_evm_epoch)
        history["valid_acpr_db"].append(valid_acpr_db)
        history["valid_oob_db"].append(valid_oob_db)
        history["worst_acpr_delta_db"].append(worst_acpr_delta)
        history["worst_oob_delta_db"].append(worst_oob_delta)

        spectral_feasible = (
            valid_acpr_db <= baseline_valid_acpr_db - spectral_improvement_margin_db
            and valid_oob_db <= baseline_valid_oob_db - spectral_improvement_margin_db
            and worst_acpr_delta <= -spectral_improvement_margin_db
            and worst_oob_delta <= -spectral_improvement_margin_db
        )
        spectral_score = ((valid_acpr_db - baseline_valid_acpr_db)
                          + (valid_oob_db - baseline_valid_oob_db))
        if spectral_score < best_observed_spectral_score - spectral_plateau_min_delta_db:
            best_observed_spectral_score = spectral_score
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        improved = spectral_score < best_spectral_score - 1e-4
        tied = abs(spectral_score - best_spectral_score) <= 1e-4
        if spectral_feasible and (improved or (tied and valid_evm_epoch < best_val_evm_db)):
            best_val_evm_db = valid_evm_epoch
            best_epoch = epoch + 1
            best_state_dict = copy.deepcopy(model.state_dict())
            best_spectral_score = spectral_score

        status = ("Epoch %d; inverse loss %.3e; Val EVM %.2f dB; Val ACPR %.2f dBc; "
                  "Val OOB %.2f dBc; worst deltas ACPR/OOB %.2f/%.2f dB; feasible=%s; plateau=%d/%d"
                  % (epoch + 1, history["inverse_loss"][-1], valid_evm_epoch, valid_acpr_db,
                     valid_oob_db, worst_acpr_delta, worst_oob_delta, spectral_feasible,
                     epochs_without_improvement, cfg.early_stopping_patience))

        epochs_without_improvement, lr_reduced = update_spectral_learning_rate(
            optimizer, epochs_without_improvement, cfg.final_lr, cfg.lr_patience)
        if lr_reduced:
            print(f"Spectral plateau: LR reduced to {cfg.final_lr:.1e}; early-stop patience reset.")
        print(status + f"; lr={history['learning_rate'][-1]:.1e}")

        if checkpoint_path is not None:
            save_training_checkpoint(model, optimizer, checkpoint_path, {
                **history, "best_epoch": best_epoch, "best_val_evm_db": best_val_evm_db,
                "best_state_dict": best_state_dict,
                "baseline_valid_acpr_db": baseline_valid_acpr_db,
                "baseline_valid_oob_db": baseline_valid_oob_db,
                "best_spectral_score": best_spectral_score, "epochs_ran": epoch + 1,
            })

        at_final_lr = all(g["lr"] <= cfg.final_lr for g in optimizer.param_groups)
        if ((epoch + 1) >= cfg.min_epochs and at_final_lr
                and epochs_without_improvement >= cfg.early_stopping_patience):
            print("Stopping early at epoch %d because canonical spectral validation plateaued."
                  % (epoch + 1))
            break

    if best_state_dict is not None:
        print("Selected canonical spectral checkpoint: epoch %d; validation EVM %.2f dB"
              % (best_epoch, best_val_evm_db))
        model.load_state_dict(best_state_dict)

    return {**history, "model": model, "best_epoch": best_epoch,
            "best_val_evm_db": best_val_evm_db, "best_state_dict": best_state_dict,
            "baseline_valid_acpr_db": baseline_valid_acpr_db,
            "baseline_valid_oob_db": baseline_valid_oob_db,
            "epochs_ran": len(history["inverse_loss"])}


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def run(cfg: HybridCnnConfig | None = None, output_dir: Path | None = None):
    cfg = cfg or HybridCnnConfig()
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    output_dir = output_dir or (Path("runs") / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                                                + "_hybrid_cnn_rls_indirect"))
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_model = build_rls_baseline(cfg)
    # Deployment-mode loaders ([x | RLS(x)] features) for validation/held-out.
    (_, valid_loader, _, train_data, valid_data, test_data) = build_datasets(cfg, baseline_model)
    # Inverse-training pairs captured through the NumPy PA (no DPD in the loop).
    inverse_loader = InversePairBatcher(train_data[:, 0, :], baseline_model,
                                        symbols_per_batch=cfg.symbols_per_batch,
                                        shuffle=True, seed=cfg.dataset_seed)

    model = DpdCnn(input_size=4, hidden_size=cfg.hidden_size).to(device)
    complexity = build_model_complexity_summary(model, cfg.gmp)
    print("Small CNN cost: %d parameters; %d real MACs/sample"
          % (complexity["cnn_trainable_real_parameters"], complexity["cnn_real_macs_per_sample"]))

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.initial_lr, weight_decay=1e-6)
    summary = train_indirect(model, optimizer, inverse_loader, valid_loader, cfg,
                             checkpoint_path=output_dir / "cnn_dpd_training_checkpoint.pt")

    held_out = evaluate_held_out(model, test_data, baseline_model)
    lines = ["Backoff | Count | RLS ACPR | CNN ACPR | Delta | RLS OOB | CNN OOB | Delta"]
    for backoff_db, m in held_out["by_backoff"].items():
        r, c = m["rls"], m["cnn"]
        lines.append(f"{backoff_db:+d} dB | {m['count']} | {r['acpr_db']:.2f} | {c['acpr_db']:.2f} | "
                     f"{c['acpr_db'] - r['acpr_db']:+.2f} | {r['oob_db']:.2f} | "
                     f"{c['oob_db']:.2f} | {c['oob_db'] - r['oob_db']:+.2f}")
    (output_dir / "held_out_spectral_by_backoff.txt").write_text("\n".join(lines) + "\n",
                                                                 encoding="utf-8")
    print("\n".join(lines))

    weights_path = output_dir / "cnn_dpd_weights.pt"
    save_model_checkpoint(model, weights_path, {
        "input_size": 4, "hidden_size": cfg.hidden_size,
        "best_epoch": summary["best_epoch"],
        "best_val_evm_db": summary["best_val_evm_db"],
        "dataset_symbols": cfg.dataset_symbols, "backoff_db": cfg.backoff_db,
        "dataset_seed": cfg.dataset_seed,
        "train_symbols": len(train_data), "validation_symbols": len(valid_data),
        "test_symbols": len(test_data),
        "max_training_epochs": cfg.max_epochs, "min_training_epochs": cfg.min_epochs,
        "spectral_plateau_patience": cfg.early_stopping_patience,
        "training_mode": "indirect_learning_post_inverse_rls_residual",
        "training_loss": "signal_power_normalized_inverse_mse",
        "pa_in_training_graph": False,
        "spectral_improvement_margin_db": 0.1,
        "checkpoint_priority": "spectral_first_then_evm",
        "validation_spectral_evaluator": "canonical_numpy_pa",
        "early_stopping_metric": "canonical_spectral_score_plateau",
        "model_complexity": complexity,
        "held_out_spectral_by_backoff": to_builtin_value(held_out["by_backoff"]),
        "rls_baseline_coeffs": np.asarray(baseline_model.coeffs, dtype=complex),
    })
    print("Saved best model weights to:", weights_path.resolve())
    return model, summary, held_out


def main() -> None:
    run()


if __name__ == "__main__":
    main()
