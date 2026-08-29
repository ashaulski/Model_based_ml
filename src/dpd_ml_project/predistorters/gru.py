"""GRU-based neural predistorter, wrapped for the modular DPD framework.

Unlike the linear GMP path, the GRU is a black-box sequence-to-sequence inverse
model: it learns ``pa_out -> pa_in`` and is applied as a predistorter by feeding
the desired signal through it. It exposes ``predistort`` / ``get_state`` /
``set_state`` (Predistorter protocol) but no ``basis`` - it is fit by a gradient
adapter (``adapters.gru_sgd.GruAdapter``), not by LS/RLS/Kalman.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from dpd_ml_project.core.per_arc_config import NnConfig


class _DpdGruNet(nn.Module):
    """Sequence-to-sequence GRU + FC head (one output sample per input sample)."""

    def __init__(self, input_size: int, hidden_size: int, num_outputs: int,
                 num_layers: int, fc1_size: int, fc2_size: int, dropout: float) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            dropout=(dropout if num_layers > 1 else 0.0), batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, fc1_size), nn.ReLU(),
            nn.Linear(fc1_size, fc2_size), nn.ReLU(),
            nn.Linear(fc2_size, num_outputs),
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None):
        rnn_out, h = self.rnn(x, h)
        return self.head(rnn_out), h


def complex_to_features(x: np.ndarray, enhance_features: bool) -> torch.Tensor:
    """Map a complex 1-D signal to a (1, N, F) real feature tensor (F=2 or 4)."""
    x = np.asarray(x, dtype=complex).reshape(-1)
    xr = torch.as_tensor(x.real, dtype=torch.float32)
    xi = torch.as_tensor(x.imag, dtype=torch.float32)
    if enhance_features:
        p = xr ** 2 + xi ** 2
        feats = torch.stack((xr, xi, p, p ** 2), dim=-1)
    else:
        feats = torch.stack((xr, xi), dim=-1)
    return feats.view(1, x.shape[0], -1)


class GruPredistorter:
    """Neural predistorter wrapping a GRU sequence model."""

    def __init__(self, cfg: NnConfig | None = None, device: torch.device | None = None) -> None:
        self.cfg = cfg or NnConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = _DpdGruNet(
            input_size=self.cfg.input_size,
            hidden_size=self.cfg.hidden_size,
            num_outputs=self.cfg.num_outputs,
            num_layers=self.cfg.num_layers,
            fc1_size=self.cfg.fc1_size,
            fc2_size=self.cfg.fc2_size,
            dropout=self.cfg.dropout,
        ).to(self.device)

    def predistort(self, x: np.ndarray) -> np.ndarray:
        feats = complex_to_features(x, self.cfg.enhance_features).to(self.device)
        was_training = self.net.training
        self.net.eval()
        with torch.no_grad():
            out, _ = self.net(feats)
        if was_training:
            self.net.train()
        iq = out[..., :2].reshape(-1, 2).cpu().contiguous()
        return torch.view_as_complex(iq).numpy()

    def get_state(self) -> dict[str, Any]:
        return {
            "state_dict": {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()},
            "cfg": self.cfg,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self.net.load_state_dict(state["state_dict"])
        if state.get("cfg") is not None:
            self.cfg = state["cfg"]
