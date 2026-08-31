"""Vector-decomposed time-delay neural network (VDTDNN / AVDTDNN) predistorter.

Physics-informed neural DPD (Zhang et al., IEEE Access 2019): the nonlinearity is
applied only to the input *magnitude*, and the phase is restored by linear
cos/sin weighting. Because magnitude and phase are derived from the input data,
the network holds only real parameters and trains with ordinary real gradients.

``augmented=True`` adds higher-order magnitude terms to the input (AVDTDNN).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from dpd_ml_project.core.per_arc_config import VdtdnnConfig

_ACTIVATIONS = {
    "abs": torch.abs,
    "relu": torch.relu,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
}


def vdtdnn_features(
    x: np.ndarray, memory_depth: int, powers: tuple[int, ...]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a complex signal into delayed magnitude / phase features.

    Returns ``(mag_feats, mag_taps, cos_taps, sin_taps)`` where ``mag_feats`` is
    ``(N, (M+1)*len(powers))`` and the tap tensors are ``(N, M+1)``.
    """
    x = np.asarray(x, dtype=complex).reshape(-1)
    num_samples = len(x)

    delayed = np.zeros((num_samples, memory_depth + 1), dtype=complex)
    for m in range(memory_depth + 1):
        if m == 0:
            delayed[:, 0] = x
        else:
            delayed[m:, m] = x[:num_samples - m]

    mag = np.abs(delayed)
    phase = np.angle(delayed)

    mag_t = torch.as_tensor(mag, dtype=torch.float32)
    mag_feats = torch.cat([mag_t ** p for p in powers], dim=1)
    cos_taps = torch.as_tensor(np.cos(phase), dtype=torch.float32)
    sin_taps = torch.as_tensor(np.sin(phase), dtype=torch.float32)
    return mag_feats, mag_t, cos_taps, sin_taps


class VdtdnnNet(nn.Module):
    """Magnitude MLP + phase-recovery layer producing I/Q."""

    def __init__(self, cfg: VdtdnnConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation: {cfg.activation!r}")
        self.act = _ACTIVATIONS[cfg.activation]

        self.fc_in = nn.Linear(cfg.in_dim, cfg.num_neurons)
        num_phase_units = cfg.num_neurons + (cfg.num_groups if cfg.linear_term else 0)
        # No output bias: a constant offset would be a nonphysical contribution.
        self.fc_out = nn.Linear(2 * num_phase_units, 2, bias=False)

        # Each hidden neuron is restored with the phase of its group's memory tap.
        group_of = torch.arange(cfg.num_neurons) // cfg.neurons_per_group
        self.register_buffer("group_of", group_of.long())

    def forward(
        self,
        mag_feats: torch.Tensor,
        mag_taps: torch.Tensor,
        cos_taps: torch.Tensor,
        sin_taps: torch.Tensor,
    ) -> torch.Tensor:
        a = self.act(self.fc_in(mag_feats))
        cos_g = cos_taps[:, self.group_of]
        sin_g = sin_taps[:, self.group_of]
        parts = [a * cos_g, a * sin_g]
        if self.cfg.linear_term:
            # Linear shortcut: |x(n-m)|*(cos,sin) is exactly the delayed I/Q tap.
            parts += [mag_taps * cos_taps, mag_taps * sin_taps]
        return self.fc_out(torch.cat(parts, dim=1))


class VdtdnnPredistorter:
    """Neural predistorter wrapping a VDTDNN / AVDTDNN network."""

    def __init__(self, cfg: VdtdnnConfig | None = None, device: torch.device | None = None) -> None:
        self.cfg = cfg or VdtdnnConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = VdtdnnNet(self.cfg).to(self.device)

    def features(self, x: np.ndarray) -> tuple[torch.Tensor, ...]:
        feats = vdtdnn_features(x, self.cfg.memory_depth, self.cfg.in_powers)
        return tuple(t.to(self.device) for t in feats)

    def predistort(self, x: np.ndarray) -> np.ndarray:
        feats = self.features(x)
        was_training = self.net.training
        self.net.eval()
        with torch.no_grad():
            out = self.net(*feats)
        if was_training:
            self.net.train()
        return torch.view_as_complex(out.cpu().contiguous()).numpy()

    def num_coefficients(self) -> int:
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def get_state(self) -> dict[str, Any]:
        return {
            "state_dict": {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()},
            "cfg": self.cfg,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self.net.load_state_dict(state["state_dict"])
        if state.get("cfg") is not None:
            self.cfg = state["cfg"]
