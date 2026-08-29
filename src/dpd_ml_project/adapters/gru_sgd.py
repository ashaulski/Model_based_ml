"""Gradient (SGD) adapter that trains a neural predistorter online.

Fits the wrapped GRU as an inverse model - ``net(features(pa_out)) -> features(pa_in)``
- with mini-batch Adam. Optimizer and (optionally) the GRU hidden state persist
across ``update`` calls so calling it repeatedly is equivalent to online training.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from dpd_ml_project.predistorters.gru import complex_to_features


class GruAdapter:
    """Online Adam trainer for a ``GruPredistorter``."""

    def __init__(self, lr: float = 2e-4, batch_size: int = 64,
                 epochs_per_step: int = 1, carry_hidden: bool = True) -> None:
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.epochs_per_step = int(epochs_per_step)
        self.carry_hidden = carry_hidden
        self.criterion = nn.MSELoss()
        self._optimizer: torch.optim.Optimizer | None = None
        self._h: torch.Tensor | None = None
        self.last_loss: float = float("nan")

    def reset(self) -> None:
        self._optimizer = None
        self._h = None
        self.last_loss = float("nan")

    def update(self, model: Any, capture: Any) -> None:
        net = model.net
        device = model.device
        enh = model.cfg.enhance_features

        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)

        # Inverse model target: recover the PA input from the PA output.
        x_in = complex_to_features(capture.pa_out, enh).to(device)
        target = complex_to_features(capture.pa_in, enh).to(device)
        num_samples = x_in.shape[1]
        bs = self.batch_size

        net.train()
        for _ in range(self.epochs_per_step):
            h = self._h if self.carry_hidden else None
            for start in range(0, num_samples - bs + 1, bs):
                xb = x_in[:, start:start + bs, :]
                tb = target[:, start:start + bs, :]
                self._optimizer.zero_grad()
                pred, h = net(xb, h)
                loss = self.criterion(pred, tb)
                loss.backward()
                self._optimizer.step()
                h = h.detach()
                self.last_loss = float(loss.item())
            if self.carry_hidden:
                self._h = h.detach() if h is not None else None
