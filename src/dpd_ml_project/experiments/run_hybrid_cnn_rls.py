"""Driver: RLS-assisted CNN DPD (small FPGA candidate).

Ported from the ``dpd_ml_reduced_best`` notebook. A small causal CNN learns a
residual on top of a GMP+RLS predistorter, trained by *direct* learning: the
prediction is pushed through a differentiable PA and compared with the clean
waveform, with per-backoff ACPR/OOB penalties relative to the RLS baseline.
Checkpoint selection is spectral-first (canonical NumPy PA evaluator); EVM is
the tie-breaker.

Run: python -m dpd_ml_project.experiments.run_hybrid_cnn_rls
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dpd_ml_project.channel.pa_model import apply_pa_model, apply_pa_model_torch
from dpd_ml_project.core.common_config import SimConfig
from dpd_ml_project.core.per_arc_config import GmpConfig
from dpd_ml_project.orchestrator.pipeline import run_step
from dpd_ml_project.predistorters.gmp import GmpPredistorter
from dpd_ml_project.siggen.SigGen import gen_lsig

_NFFT = SimConfig.FFT_SIZE
_CP_LEN = SimConfig.CP_LEN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class HybridCnnConfig:
    """Knobs for the RLS-assisted CNN experiment."""
    # GMP/RLS baseline (notebook used Ka=5, La/Lb/Lc=3)
    gmp: GmpConfig = field(default_factory=lambda: GmpConfig(
        Ka=5, La=3, Kb=3, Lb=3, Mb=1, Kc=3, Lc=3, Mc=1))
    lambda_rls: float = 0.99
    rls_symbols: int = 10
    rls_signal_rms_dbp: float = -10.0
    snr_db: float = 70.0

    # dataset
    dataset_symbols: int = 1000
    backoff_db: tuple[float, ...] = (-12.0, -15.0, -18.0, -21.0)
    dataset_seed: int = 10000
    symbols_per_batch: int = 16

    # model / training
    hidden_size: int = 32
    initial_lr: float = 1e-3
    final_lr: float = 1e-4
    lr_patience: int = 5
    max_epochs: int = 60
    min_epochs: int = 10
    early_stopping_patience: int = 15
    evm_weight: float = 1.0
    seed: int = 1234


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class DpdCnn(nn.Module):
    """Causal residual CNN: 4 -> 32 (5 taps) -> 16 (3 taps) -> 2.

    Seven-sample receptive field. Output is the RLS I/Q plus a scaled residual.
    """

    def __init__(self, input_size: int = 4, hidden_size: int = 32) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = 5
        self.second_kernel_size = 3
        self.backbone = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=self.kernel_size),
            nn.ReLU(),
            nn.ConstantPad1d((self.second_kernel_size - 1, 0), 0.0),
            nn.Conv1d(hidden_size, 16, kernel_size=self.second_kernel_size),
            nn.ReLU(),
        )
        self.head = nn.Conv1d(16, 2, kernel_size=1)
        # Preserve the calibrated RLS output at initialization.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, h=None):
        x_ch = F.pad(x.transpose(1, 2), (self.kernel_size - 1, 0))
        delta = self.head(self.backbone(x_ch)).transpose(1, 2)
        # Power-of-two residual scale can be implemented as a fixed-point shift.
        return x[..., 2:4] + delta * (1.0 / 16.0), None


def build_model_complexity_summary(model: DpdCnn, gmp_cfg: GmpConfig) -> dict[str, Any]:
    rls_complex_coefficients = int(gmp_cfg.Ncoeff)
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    total = int(sum(p.numel() for p in model.parameters()))
    return {
        "rls_complex_coefficients": rls_complex_coefficients,
        "rls_real_degrees_of_freedom": 2 * rls_complex_coefficients,
        "cnn_trainable_real_parameters": trainable,
        "cnn_total_real_parameters": total,
        "cnn_to_rls_real_dof_ratio": trainable / (2 * rls_complex_coefficients),
        "cnn_real_macs_per_sample": sum(
            m.in_channels * m.out_channels * m.kernel_size[0] // m.groups
            for m in model.modules() if isinstance(m, nn.Conv1d)
        ),
        "cnn_receptive_field_samples": model.kernel_size + model.second_kernel_size - 1,
        "cnn_weight_storage_bytes_int16_estimate": 2 * total,
        "cnn_residual_scale": 1.0 / 16.0,
    }


# -----------------------------------------------------------------------------
# PA equivalence gate
# -----------------------------------------------------------------------------
def assert_pa_model_equivalence(rms_db_levels=(-12.0, -15.0, -18.0, -21.0),
                                atol=2e-6, rtol=2e-5) -> dict[str, float]:
    """Fail fast if the differentiable PA diverges from the canonical NumPy PA."""
    symbol_len = int(_NFFT + _CP_LEN)
    rng = np.random.default_rng(20260723)
    symbols = []
    for rms_db in rms_db_levels:
        s = rng.standard_normal(symbol_len) + 1j * rng.standard_normal(symbol_len)
        s *= 10.0 ** (rms_db / 20.0) / np.sqrt(np.mean(np.abs(s) ** 2))
        symbols.append(s)
    symbols = np.asarray(symbols, dtype=np.complex128)
    numpy_output = np.stack([apply_pa_model(s, bypass=False) for s in symbols])
    torch_input = torch.as_tensor(np.stack((symbols.real, symbols.imag), axis=-1), dtype=torch.float64)
    with torch.no_grad():
        torch_iq = apply_pa_model_torch(torch_input)
    torch_output = torch.complex(torch_iq[..., 0], torch_iq[..., 1]).cpu().numpy()
    error = torch_output - numpy_output
    max_abs_error = float(np.max(np.abs(error)))
    relative_error_db = float(20.0 * np.log10(max(
        np.sqrt(np.mean(np.abs(error) ** 2) / max(np.mean(np.abs(numpy_output) ** 2), 1e-24)), 1e-15)))
    if not np.allclose(torch_output, numpy_output, atol=atol, rtol=rtol):
        raise AssertionError(
            f"PyTorch/NumPy PA mismatch: max abs error={max_abs_error:.3e}, "
            f"relative error={relative_error_db:.2f} dB")
    return {"max_abs_error": max_abs_error, "relative_error_db": relative_error_db}


# -----------------------------------------------------------------------------
# Alignment helpers
# -----------------------------------------------------------------------------
def align_complex_waveform_torch(reference: torch.Tensor, measured: torch.Tensor) -> torch.Tensor:
    ref = torch.complex(reference[..., 0], reference[..., 1])
    meas = torch.complex(measured[..., 0], measured[..., 1])
    gain = torch.sum(torch.conj(meas) * ref, dim=1, keepdim=True) / torch.clamp(
        torch.sum(torch.abs(meas) ** 2, dim=1, keepdim=True), min=1e-12)
    aligned = gain * meas
    return torch.stack((aligned.real, aligned.imag), dim=-1)


def align_complex_waveform_numpy(reference: np.ndarray, measured: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference).reshape(-1)
    measured = np.asarray(measured).reshape(-1)
    gain = np.sum(np.conj(measured) * reference) / max(np.sum(np.abs(measured) ** 2), 1e-24)
    return gain * measured


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
class OfdmBatcher:
    """Yields (target_iq, [input_iq | rls_baseline_iq]) per batch of whole symbols."""

    def __init__(self, ofdms, baseline_model: GmpPredistorter, symbols_per_batch=16,
                 drop_last=False, shuffle=False, seed=0) -> None:
        ofdms = np.asarray(ofdms, dtype=np.complex64)
        if ofdms.ndim != 3 or ofdms.shape[1] < 2:
            raise ValueError("Expected OFDM data with shape [symbols, 2, samples].")
        self.targets = torch.as_tensor(ofdms[:, 0, :], dtype=torch.complex64)
        model_inputs = np.asarray(ofdms[:, 1, :], dtype=np.complex64)
        baseline = np.stack([
            np.asarray(baseline_model.predistort(s), dtype=np.complex64) for s in model_inputs
        ])
        input_iq = np.stack((model_inputs.real, model_inputs.imag), axis=-1)
        baseline_iq = np.stack((baseline.real, baseline.imag), axis=-1)
        self.model_features = torch.as_tensor(
            np.concatenate((input_iq, baseline_iq), axis=-1), dtype=torch.float32)
        self.symbols_per_batch = max(1, int(symbols_per_batch))
        self.drop_last = bool(drop_last)
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
            if self.drop_last and len(batch) < self.symbols_per_batch:
                continue
            target = self.targets[batch]
            yield torch.stack((target.real, target.imag), dim=-1), self.model_features[batch]


# -----------------------------------------------------------------------------
# Spectral metrics
# -----------------------------------------------------------------------------
def _spectral_power_components_torch(pred, target, n_fft=_NFFT):
    pa_out = align_complex_waveform_torch(target, apply_pa_model_torch(pred))
    measured = torch.complex(pa_out[..., 0], pa_out[..., 1])
    fft_start = _CP_LEN // 2
    measured_fd = torch.fft.fftshift(
        torch.fft.fft(measured[:, fft_start:fft_start + n_fft], n=n_fft, dim=1, norm="ortho"), dim=1)
    power = torch.abs(measured_fd) ** 2
    center = n_fft // 2
    main_slice = slice(center - 32, center + 32)
    upper_slice = slice(center + 96 - 32, center + 96 + 32)
    lower_slice = slice(center - 96 - 32, center - 96 + 32)
    rest_mask = torch.ones(n_fft, dtype=torch.bool, device=pred.device)
    rest_mask[main_slice] = False
    rest_mask[upper_slice] = False
    rest_mask[lower_slice] = False
    main_power = torch.sum(power[:, main_slice], dim=1)
    adjacent_power = 0.5 * (torch.sum(power[:, upper_slice], dim=1)
                            + torch.sum(power[:, lower_slice], dim=1))
    rest_power = torch.sum(power[:, rest_mask], dim=1)
    target_backoff_db = 10.0 * torch.log10(torch.clamp(
        torch.mean(torch.sum(target ** 2, dim=-1), dim=1), min=1e-12))
    return main_power, adjacent_power, rest_power, target_backoff_db


def compute_acpr_db(waveform, main_half_bw_bins=32, adj_half_bw_bins=32, adj_offset_bins=96,
                    n_fft=_NFFT, return_components=False):
    waveform = np.asarray(waveform)
    if waveform.ndim == 1:
        waveform = waveform.reshape(1, -1)
    elif waveform.ndim > 2:
        waveform = waveform.reshape(-1, waveform.shape[-1])
    if waveform.shape[1] >= n_fft:
        centered = waveform[:, :n_fft]
    else:
        centered = np.pad(waveform, ((0, 0), (0, n_fft - waveform.shape[1])), mode="constant")
    spectrum = np.fft.fftshift(np.fft.fft(centered, n=n_fft, axis=1), axes=1)
    power = np.mean(np.abs(spectrum) ** 2, axis=0)
    center = n_fft // 2
    main_slice = slice(center - main_half_bw_bins, center + main_half_bw_bins)
    upper = slice(center + adj_offset_bins - adj_half_bw_bins, center + adj_offset_bins + adj_half_bw_bins)
    lower = slice(center - adj_offset_bins - adj_half_bw_bins, center - adj_offset_bins + adj_half_bw_bins)
    main_power = np.sum(power[main_slice])
    adj_power = 0.5 * (np.sum(power[upper]) + np.sum(power[lower]))
    rest_mask = np.ones_like(power, dtype=bool)
    rest_mask[main_slice] = False
    rest_mask[upper] = False
    rest_mask[lower] = False
    acpr_db = 10 * np.log10(max(adj_power, 1e-24) / max(main_power, 1e-24))
    rest_db = 10 * np.log10(max(np.sum(power[rest_mask]), 1e-24) / max(main_power, 1e-24))
    return (acpr_db, rest_db) if return_components else acpr_db


def compute_aligned_spectral_metrics_by_backoff(reference_symbols, aligned_outputs_by_method,
                                                n_fft=_NFFT):
    """Aggregate and per-backoff spectra using the canonical NumPy masks."""
    reference_symbols = np.asarray(reference_symbols)
    if reference_symbols.ndim == 1:
        reference_symbols = reference_symbols.reshape(1, -1)
    backoff_keys = np.rint(10.0 * np.log10(np.maximum(
        np.mean(np.abs(reference_symbols) ** 2, axis=1), 1e-24))).astype(int)
    fft_slice = slice(_CP_LEN // 2, _CP_LEN // 2 + n_fft)
    method_outputs = {name: np.asarray(v) for name, v in aligned_outputs_by_method.items()}

    def evaluate(values, mask):
        acpr_db, oob_db = compute_acpr_db(values[mask, fft_slice], n_fft=n_fft, return_components=True)
        return {"acpr_db": float(acpr_db), "oob_db": float(oob_db)}

    all_mask = np.ones(len(reference_symbols), dtype=bool)
    overall = {name: evaluate(v, all_mask) for name, v in method_outputs.items()}
    by_backoff = {}
    for key in sorted(np.unique(backoff_keys)):
        mask = backoff_keys == key
        by_backoff[int(key)] = {"count": int(np.sum(mask)),
                                **{name: evaluate(v, mask) for name, v in method_outputs.items()}}
    return {"overall": overall, "by_backoff": by_backoff}


def compute_epoch_spectral_comparison_numpy(model, loader):
    """Canonical non-differentiable validation using the exact NumPy PA."""
    was_training = model.training
    model.eval()
    references, cnn_pd, rls_pd = [], [], []
    with torch.no_grad():
        for target_iq, features in loader:
            target_iq = target_iq[:, :, :2].to(device)
            cnn_input = features.to(device)
            cnn_pred = model(cnn_input)[0]
            references.append(torch.complex(target_iq[..., 0], target_iq[..., 1]).cpu().numpy())
            cnn_pd.append(torch.complex(cnn_pred[..., 0], cnn_pred[..., 1]).cpu().numpy())
            rls_iq = cnn_input[..., 2:4]
            rls_pd.append(torch.complex(rls_iq[..., 0], rls_iq[..., 1]).cpu().numpy())
    if was_training:
        model.train()
    references = np.concatenate(references, axis=0)

    def pa_and_align(predistorted):
        return np.stack([align_complex_waveform_numpy(ref, apply_pa_model(sym, bypass=False))
                         for ref, sym in zip(references, np.concatenate(predistorted, axis=0))])

    metrics = compute_aligned_spectral_metrics_by_backoff(
        references, {"cnn": pa_and_align(cnn_pd), "rls": pa_and_align(rls_pd)}, n_fft=_NFFT)
    cnn_overall, rls_overall = metrics["overall"]["cnn"], metrics["overall"]["rls"]
    group_metrics = {
        key: {"count": g["count"],
              "cnn_acpr_db": g["cnn"]["acpr_db"], "rls_acpr_db": g["rls"]["acpr_db"],
              "cnn_oob_db": g["cnn"]["oob_db"], "rls_oob_db": g["rls"]["oob_db"],
              "acpr_delta_db": g["cnn"]["acpr_db"] - g["rls"]["acpr_db"],
              "oob_delta_db": g["cnn"]["oob_db"] - g["rls"]["oob_db"]}
        for key, g in metrics["by_backoff"].items()
    }
    return {
        "cnn_acpr_db": cnn_overall["acpr_db"], "cnn_oob_db": cnn_overall["oob_db"],
        "rls_acpr_db": rls_overall["acpr_db"], "rls_oob_db": rls_overall["oob_db"],
        "worst_acpr_delta_db": max((v["acpr_delta_db"] for v in group_metrics.values()),
                                   default=float("inf")),
        "worst_oob_delta_db": max((v["oob_delta_db"] for v in group_metrics.values()),
                                  default=float("inf")),
        "group_metrics": group_metrics,
    }


def compute_epoch_spectral_metrics(model, loader):
    """Differentiable-PA ACPR/OOB used for the cheaper train-set readout."""
    was_training = model.training
    model.eval()
    totals = [0.0, 0.0, 0.0]
    with torch.no_grad():
        for target_iq, features in loader:
            target_iq = target_iq[:, :, :2].to(device)
            pred = model(features.to(device))[0]
            comps = _spectral_power_components_torch(pred, target_iq)
            totals = [t + float(torch.sum(c).item()) for t, c in zip(totals, comps[:3])]
    if was_training:
        model.train()
    main_power, adjacent_power, rest_power = totals
    return (10.0 * np.log10(max(adjacent_power, 1e-24) / max(main_power, 1e-24)),
            10.0 * np.log10(max(rest_power, 1e-24) / max(main_power, 1e-24)))


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------
def original_signal_spectral_loss_torch(pred, target, main_half_bw_bins=32, adj_half_bw_bins=32,
                                        adj_offset_bins=96, n_fft=_NFFT):
    """Compare the PA output directly with the clean signal in every FFT region."""
    pa_out = align_complex_waveform_torch(target, torch.nan_to_num(apply_pa_model_torch(pred), nan=0.0))
    measured = torch.complex(pa_out[..., 0], pa_out[..., 1])
    reference = torch.complex(target[..., 0], target[..., 1])
    fft_start = _CP_LEN // 2
    fft_stop = fft_start + n_fft
    if measured.shape[1] < fft_stop:
        raise ValueError(f"Spectral loss needs at least {fft_stop} samples; got {measured.shape[1]}.")
    measured_fd = torch.fft.fftshift(
        torch.fft.fft(measured[:, fft_start:fft_stop], n=n_fft, dim=1, norm="ortho"), dim=1)
    reference_fd = torch.fft.fftshift(
        torch.fft.fft(reference[:, fft_start:fft_stop], n=n_fft, dim=1, norm="ortho"), dim=1)
    center = n_fft // 2
    main_slice = slice(center - main_half_bw_bins, center + main_half_bw_bins)
    upper = slice(center + adj_offset_bins - adj_half_bw_bins, center + adj_offset_bins + adj_half_bw_bins)
    lower = slice(center - adj_offset_bins - adj_half_bw_bins, center - adj_offset_bins + adj_half_bw_bins)
    rest_mask = torch.ones(n_fft, dtype=torch.bool, device=pred.device)
    rest_mask[main_slice] = False
    rest_mask[upper] = False
    rest_mask[lower] = False
    reference_main_power = torch.sum(torch.abs(reference_fd[:, main_slice]) ** 2, dim=1)

    def robust_region_nrmse(error_power):
        # Per-symbol normalization keeps every backoff equally influential.
        per_symbol = torch.sqrt(error_power / torch.clamp(reference_main_power, min=1e-12))
        tail_count = max(1, (per_symbol.numel() + 3) // 4)
        return per_symbol.mean() + 0.5 * torch.topk(per_symbol, k=tail_count, largest=True).values.mean()

    spectral_error = torch.abs(measured_fd - reference_fd) ** 2
    ib_loss = robust_region_nrmse(torch.sum(spectral_error[:, main_slice], dim=1))
    adjacent_loss = robust_region_nrmse(0.5 * (torch.sum(spectral_error[:, lower], dim=1)
                                               + torch.sum(spectral_error[:, upper], dim=1)))
    oob_loss = robust_region_nrmse(torch.sum(spectral_error[:, rest_mask], dim=1))

    measured_power = torch.mean(torch.abs(measured_fd) ** 2, dim=0)
    main_power = torch.sum(measured_power[main_slice])
    adjacent_power = 0.5 * (torch.sum(measured_power[lower]) + torch.sum(measured_power[upper]))
    acpr_db = 10.0 * torch.log10(torch.clamp(
        adjacent_power / torch.clamp(main_power, min=1e-12), min=1e-12))
    oob_db = 10.0 * torch.log10(torch.clamp(
        torch.sum(measured_power[rest_mask]) / torch.clamp(main_power, min=1e-12), min=1e-12))
    return ib_loss, adjacent_loss, oob_loss, acpr_db, oob_db


def per_backoff_spectral_penalty_torch(pred, target, baseline_pred, margin_db=0.1):
    """Penalize ACPR/OOB shortfall against the detached RLS baseline per backoff.

    Equal group weighting plus the worst violation stops an easy backoff from
    hiding a failing one. The margin matches checkpoint acceptance.
    """
    cnn = _spectral_power_components_torch(pred, target)
    with torch.no_grad():
        rls = _spectral_power_components_torch(baseline_pred.detach(), target)
    keys = torch.round(cnn[3].detach()).to(torch.int64)
    violations = []
    for key in torch.unique(keys):
        mask = keys == key
        for region in (1, 2):
            cnn_ratio = cnn[region][mask].sum() / cnn[0][mask].sum().clamp_min(1e-24)
            rls_ratio = rls[region][mask].sum() / rls[0][mask].sum().clamp_min(1e-24)
            delta_db = 10.0 * torch.log10(cnn_ratio.clamp_min(1e-24) / rls_ratio.clamp_min(1e-24))
            violations.append(F.relu(delta_db + margin_db).square())
    violations = torch.stack(violations)
    return violations.mean() + violations.max()


def adaptive_constrained_training_loss_torch(pred, target, baseline_pred, evm_weight=1.0):
    pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    pa_aligned = align_complex_waveform_torch(
        target, torch.nan_to_num(apply_pa_model_torch(pred), nan=0.0))
    error_power = torch.mean(torch.sum((pa_aligned - target) ** 2, dim=-1))
    signal_power = torch.mean(torch.sum(target ** 2, dim=-1))
    signal_evm = torch.sqrt(error_power / torch.clamp(signal_power, min=1e-12))
    ib_loss, adjacent_loss, oob_loss, acpr_db, oob_db = original_signal_spectral_loss_torch(pred, target)
    spectral_loss = 1.0 * ib_loss + 4.0 * adjacent_loss + 2.0 * oob_loss
    baseline_penalty = per_backoff_spectral_penalty_torch(pred, target, baseline_pred)
    total_loss = evm_weight * signal_evm + spectral_loss + 0.01 * baseline_penalty
    return total_loss, {
        "evm_contrib": evm_weight * signal_evm, "evm_term": signal_evm,
        "spectral_loss": spectral_loss, "baseline_improvement_penalty": baseline_penalty,
        "ib_loss": ib_loss, "adjacent_loss": adjacent_loss, "oob_loss": oob_loss,
        "acpr_db": acpr_db, "oob_db": oob_db,
    }


def get_mse_and_evm(model, loader):
    """In-band EVM measured on the PA output after gain alignment."""
    total_error_power = total_signal_power = 0.0
    total_values = 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for target_iq, features in loader:
            target_iq = target_iq[:, :, :2].to(device)
            pred, _ = model(features.to(device))
            pred_pa = align_complex_waveform_torch(target_iq, apply_pa_model_torch(pred))
            fft_start = _CP_LEN // 2
            fft_stop = fft_start + _NFFT
            pred_c = torch.complex(pred_pa[..., 0], pred_pa[..., 1])[:, fft_start:fft_stop]
            target_c = torch.complex(target_iq[..., 0], target_iq[..., 1])[:, fft_start:fft_stop]
            pred_fd = torch.fft.fftshift(torch.fft.fft(pred_c, n=_NFFT, dim=1, norm="ortho"), dim=1)
            target_fd = torch.fft.fftshift(torch.fft.fft(target_c, n=_NFFT, dim=1, norm="ortho"), dim=1)
            main_slice = slice(_NFFT // 2 - 32, _NFFT // 2 + 32)
            error = pred_fd[:, main_slice] - target_fd[:, main_slice]
            total_error_power += float(torch.sum(torch.abs(error) ** 2).item())
            total_signal_power += float(torch.sum(torch.abs(target_fd[:, main_slice]) ** 2).item())
            total_values += int(error.numel())
    if was_training:
        model.train()
    evm_linear = np.sqrt(total_error_power / max(total_signal_power, 1e-12))
    return (total_error_power / max(1, total_values),
            20 * np.log10(max(evm_linear, 1e-12)), 100 * evm_linear)


# -----------------------------------------------------------------------------
# Checkpointing
# -----------------------------------------------------------------------------
def to_builtin_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: to_builtin_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin_value(v) for v in value]
    return value


def save_model_checkpoint(model, path, metadata):
    torch.save({"model_state_dict": model.state_dict(),
                "metadata": to_builtin_value(metadata)}, path)


def save_training_checkpoint(model, optimizer, path, train_state):
    torch.save({"model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_state": to_builtin_value(train_state)}, path)


def update_spectral_learning_rate(optimizer, plateau_epochs, final_lr, lr_patience):
    """Enter the final LR stage after a spectral plateau; return reset patience."""
    if plateau_epochs >= lr_patience and any(g["lr"] > final_lr for g in optimizer.param_groups):
        for group in optimizer.param_groups:
            group["lr"] = min(group["lr"], final_lr)
        return 0, True
    return plateau_epochs, False


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train(model, optimizer, train_loader, valid_loader, cfg: HybridCnnConfig,
          checkpoint_path: Path | None = None):
    if cfg.final_lr <= 0 or cfg.lr_patience < 1 or cfg.early_stopping_patience < 1:
        raise ValueError("Learning rate and patience must be positive.")
    if any(group["lr"] < cfg.final_lr for group in optimizer.param_groups):
        raise ValueError("Final LR must not exceed the initial LR.")

    history: dict[str, list] = {k: [] for k in (
        "learning_rate", "train_evm", "valid_evm", "train_loss", "train_acpr_db",
        "valid_acpr_db", "train_oob_db", "valid_oob_db",
        "worst_acpr_delta_db", "worst_oob_delta_db")}

    spectral_improvement_margin_db = 0.1
    spectral_plateau_min_delta_db = 0.01
    model = model.to(device)

    baseline_valid = compute_epoch_spectral_comparison_numpy(model, valid_loader)
    baseline_valid_acpr_db = baseline_valid["rls_acpr_db"]
    baseline_valid_oob_db = baseline_valid["rls_oob_db"]
    _, best_val_evm_db, best_val_evm_percent = get_mse_and_evm(model, valid_loader)
    best_epoch = 0
    best_state_dict = copy.deepcopy(model.state_dict())
    best_spectral_score = 0.0
    best_observed_spectral_score = float("inf")
    epochs_without_improvement = 0

    print(f"Training target: max {cfg.max_epochs} epochs; EVM loss weight {cfg.evm_weight:.4f}")
    print("Canonical NumPy RLS validation baseline: ACPR %.2f dBc; far OOB %.2f dBc"
          % (baseline_valid_acpr_db, baseline_valid_oob_db))

    for epoch in range(cfg.max_epochs):
        model.train()
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))
        total_loss_epoch, batch_count = 0.0, 0

        for batch_idx, (target_iq, features) in enumerate(train_loader):
            cnn_input = features.to(device)
            target_iq = target_iq[:, :, :2].to(device)
            optimizer.zero_grad()
            pred, _ = model(cnn_input)
            total_loss, _ = adaptive_constrained_training_loss_torch(
                pred, target_iq, baseline_pred=cnn_input[..., 2:4], evm_weight=cfg.evm_weight)
            if not np.isfinite(float(total_loss.detach().item())):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch + 1}, batch {batch_idx + 1}")
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss_epoch += total_loss.item()
            batch_count += 1

        _, train_evm_epoch, _ = get_mse_and_evm(model, train_loader)
        _, valid_evm_epoch, valid_evm_percent_epoch = get_mse_and_evm(model, valid_loader)
        train_acpr_db, train_oob_db = compute_epoch_spectral_metrics(model, train_loader)
        valid_cmp = compute_epoch_spectral_comparison_numpy(model, valid_loader)
        valid_acpr_db, valid_oob_db = valid_cmp["cnn_acpr_db"], valid_cmp["cnn_oob_db"]
        worst_acpr_delta = valid_cmp["worst_acpr_delta_db"]
        worst_oob_delta = valid_cmp["worst_oob_delta_db"]

        history["train_evm"].append(train_evm_epoch)
        history["valid_evm"].append(valid_evm_epoch)
        history["train_loss"].append(total_loss_epoch / max(batch_count, 1))
        history["train_acpr_db"].append(train_acpr_db)
        history["valid_acpr_db"].append(valid_acpr_db)
        history["train_oob_db"].append(train_oob_db)
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
            best_val_evm_percent = valid_evm_percent_epoch
            best_epoch = epoch + 1
            best_state_dict = copy.deepcopy(model.state_dict())
            best_spectral_score = spectral_score

        loss_db = 20 * np.log10(max(history["train_loss"][-1], 1e-12))
        status = ("Epoch %d; Loss %.2f dB; Val EVM %.2f dB; Val ACPR %.2f dBc; Val OOB %.2f dBc; "
                  "worst deltas ACPR/OOB %.2f/%.2f dB; feasible=%s; plateau=%d/%d"
                  % (epoch + 1, loss_db, valid_evm_epoch, valid_acpr_db, valid_oob_db,
                     worst_acpr_delta, worst_oob_delta, spectral_feasible,
                     epochs_without_improvement, cfg.early_stopping_patience))

        epochs_without_improvement, lr_reduced = update_spectral_learning_rate(
            optimizer, epochs_without_improvement, cfg.final_lr, cfg.lr_patience)
        if lr_reduced:
            print(f"Spectral plateau: LR reduced to {cfg.final_lr:.1e}; early-stop patience reset.")
        print(status + f"; lr={history['learning_rate'][-1]:.1e}")

        if checkpoint_path is not None:
            save_training_checkpoint(model, optimizer, checkpoint_path, {
                **history, "best_epoch": best_epoch, "best_val_evm_db": best_val_evm_db,
                "best_val_evm_percent": best_val_evm_percent, "best_state_dict": best_state_dict,
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
        print("Selected canonical spectral checkpoint: epoch %d; validation EVM %.2f dB (%.2f%%)"
              % (best_epoch, best_val_evm_db, best_val_evm_percent))
        model.load_state_dict(best_state_dict)

    return {**history, "model": model, "best_epoch": best_epoch,
            "best_val_evm_db": best_val_evm_db, "best_val_evm_percent": best_val_evm_percent,
            "best_state_dict": best_state_dict,
            "baseline_valid_acpr_db": baseline_valid_acpr_db,
            "baseline_valid_oob_db": baseline_valid_oob_db,
            "epochs_ran": len(history["train_evm"])}


# -----------------------------------------------------------------------------
# Baseline + dataset
# -----------------------------------------------------------------------------
def build_rls_baseline(cfg: HybridCnnConfig) -> GmpPredistorter:
    """Run closed-loop GMP+RLS to obtain the baseline predistorter."""
    from dpd_ml_project.adapters.rls import RlsAdapter

    model = GmpPredistorter(cfg.gmp)
    adapter = RlsAdapter(lambda_rls=cfg.lambda_rls)
    sim = SimConfig(signal_rms_dbp=cfg.rls_signal_rms_dbp, snr_db=cfg.snr_db)
    for i in range(cfg.rls_symbols):
        run_step(model, adapter, sim, apply_dpd=True, adapt=True,
                 bits_seed=123, iteration_index=i)
    return model


def build_datasets(cfg: HybridCnnConfig, baseline_model: GmpPredistorter):
    symbols = np.stack([
        np.asarray(gen_lsig(bypass=False,
                            signal_rms_dbp=cfg.backoff_db[i % len(cfg.backoff_db)],
                            repeat_bits_every_call=True,
                            bits_seed=cfg.dataset_seed + i), dtype=np.complex64)
        for i in range(cfg.dataset_symbols)
    ])
    # Direct learning: the clean symbol is both the network input and the target.
    pairs = np.stack((symbols, symbols), axis=1)
    rng = np.random.default_rng(cfg.dataset_seed + cfg.dataset_symbols)
    pairs = pairs[rng.permutation(len(pairs))]

    num = len(pairs)
    train_end = max(1, int(round(0.8 * num)))
    valid_end = min(max(train_end + 1, int(round(0.9 * num))), num - 1)
    train_data, valid_data, test_data = pairs[:train_end], pairs[train_end:valid_end], pairs[valid_end:]
    if len(valid_data) == 0:
        valid_data = pairs[train_end - 1:train_end]
    if len(test_data) == 0:
        test_data = pairs[-1:]

    def make(data, shuffle):
        return OfdmBatcher(data, baseline_model, symbols_per_batch=cfg.symbols_per_batch,
                           shuffle=shuffle, seed=cfg.dataset_seed)

    return (make(train_data, True), make(valid_data, False), make(test_data, False),
            train_data, valid_data, test_data)


# -----------------------------------------------------------------------------
# Held-out evaluation
# -----------------------------------------------------------------------------
def evaluate_held_out(model, test_data, baseline_model: GmpPredistorter):
    """Compare no-DPD / RLS / CNN on the held-out symbols with the NumPy PA."""
    references = [np.asarray(p[0], dtype=np.complex64).reshape(-1) for p in test_data]

    def pa_align(symbols):
        return [align_complex_waveform_numpy(r, apply_pa_model(s, bypass=False))
                for r, s in zip(references, symbols)]

    no_dpd = pa_align(references)
    rls_pd = [np.asarray(baseline_model.predistort(r), dtype=np.complex64) for r in references]
    rls = pa_align(rls_pd)

    was_training = model.training
    model.eval()
    cnn_pd = []
    with torch.no_grad():
        for reference in references:
            iq = torch.stack((torch.as_tensor(reference.real, dtype=torch.float32),
                              torch.as_tensor(reference.imag, dtype=torch.float32)), dim=-1)
            baseline_c = torch.as_tensor(
                np.asarray(baseline_model.predistort(reference), dtype=np.complex64))
            baseline_iq = torch.stack((baseline_c.real, baseline_c.imag), dim=-1)
            cnn_input = torch.cat((iq, baseline_iq), dim=-1).unsqueeze(0).to(device)
            pred, _ = model(cnn_input)
            cnn_pd.append(torch.complex(pred[..., 0], pred[..., 1]).reshape(-1).cpu().numpy())
    if was_training:
        model.train()
    cnn = pa_align(cnn_pd)

    return compute_aligned_spectral_metrics_by_backoff(
        np.stack(references),
        {"no_dpd": np.stack(no_dpd), "rls": np.stack(rls), "cnn": np.stack(cnn)},
        n_fft=_NFFT,
    )


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def run(cfg: HybridCnnConfig | None = None, output_dir: Path | None = None):
    cfg = cfg or HybridCnnConfig()
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    output_dir = output_dir or (Path("runs") / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                                                + "_small_fpga_constrained"))
    output_dir.mkdir(parents=True, exist_ok=True)

    pa_report = assert_pa_model_equivalence()
    print("PA equivalence passed: max abs error %.3e; relative error %.2f dB"
          % (pa_report["max_abs_error"], pa_report["relative_error_db"]))

    baseline_model = build_rls_baseline(cfg)
    (train_loader, valid_loader, test_loader,
     train_data, valid_data, test_data) = build_datasets(cfg, baseline_model)

    model = DpdCnn(input_size=4, hidden_size=cfg.hidden_size).to(device)
    complexity = build_model_complexity_summary(model, cfg.gmp)
    print("Small CNN cost: %d parameters; %d real MACs/sample"
          % (complexity["cnn_trainable_real_parameters"], complexity["cnn_real_macs_per_sample"]))
    print("Model complexity: RLS %d complex coefficients (%d real DOF); "
          "CNN %d trainable real parameters (%.2fx RLS real DOF)"
          % (complexity["rls_complex_coefficients"], complexity["rls_real_degrees_of_freedom"],
             complexity["cnn_trainable_real_parameters"], complexity["cnn_to_rls_real_dof_ratio"]))

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.initial_lr, weight_decay=1e-6)
    summary = train(model, optimizer, train_loader, valid_loader, cfg,
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
        "best_val_evm_percent": summary["best_val_evm_percent"],
        "dataset_symbols": cfg.dataset_symbols, "backoff_db": cfg.backoff_db,
        "dataset_seed": cfg.dataset_seed,
        "train_symbols": len(train_data), "validation_symbols": len(valid_data),
        "test_symbols": len(test_data),
        "max_training_epochs": cfg.max_epochs, "min_training_epochs": cfg.min_epochs,
        "spectral_plateau_patience": cfg.early_stopping_patience,
        "evm_weight": cfg.evm_weight,
        "training_mode": "direct_full_symbol_rls_residual_per_backoff_penalty",
        "baseline_spectral_penalty": {"weight": 0.01, "margin_db": 0.1,
                                      "aggregation": "mean_plus_max_squared_db_violation"},
        "spectral_improvement_margin_db": 0.1,
        "checkpoint_priority": "spectral_first_then_evm",
        "validation_spectral_evaluator": "canonical_numpy_pa",
        "early_stopping_metric": "canonical_spectral_score_plateau",
        "pa_equivalence_report": pa_report,
        "model_complexity": complexity,
        "original_signal_loss_weights": {"evm": cfg.evm_weight, "inband": 1.0,
                                         "adjacent": 4.0, "far_oob": 2.0},
        "held_out_spectral_by_backoff": held_out["by_backoff"],
        "rls_baseline_coeffs": np.asarray(baseline_model.coeffs, dtype=complex),
    })
    print("Saved best model weights to:", weights_path.resolve())
    return model, summary, held_out


def main() -> None:
    run()


if __name__ == "__main__":
    main()
