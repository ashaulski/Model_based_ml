"""Gradient (Adam) adapter for the VDTDNN / AVDTDNN predistorter.

Indirect learning: fits ``net(features(pa_out)) -> (I, Q) of pa_in``. The model
is feed-forward with memory carried by delay taps, so batches are plain sample
slices - no hidden state to carry between calls.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class VdtdnnAdapter:
    """Online/offline Adam trainer for a ``VdtdnnPredistorter``."""

    def __init__(self, lr: float = 1e-2, batch_size: int = 256,
                 epochs_per_step: int = 1, shuffle: bool = True) -> None:
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.epochs_per_step = int(epochs_per_step)
        self.shuffle = shuffle
        self.criterion = nn.MSELoss()
        self._optimizer: torch.optim.Optimizer | None = None
        self.last_loss: float = float("nan")

    def reset(self) -> None:
        self._optimizer = None
        self.last_loss = float("nan")

    def set_lr(self, lr: float) -> None:
        self.lr = float(lr)
        if self._optimizer is not None:
            # optimizer has groups, optionally 
            # we can have different learning rates for different groups
            for group in self._optimizer.param_groups:
                group["lr"] = self.lr

    def update(self, model: Any, capture: Any) -> None:
        net = model.net
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)

        # Inverse model: recover the PA input from the PA output.

        # create tensors for the features (|x|^p, |x|, cos, sin), all with memory taps
        # dimensions are: 
        #   mag_feats dimensions    = num_samples x memory_depth x num_in_powers
        #   all the rest dimensions = num_samples x memory_depth
        mag_feats, mag_taps, cos_taps, sin_taps = model.features(capture.pa_out)
        # use PA inputas the target for the inverse model, 
        # convert to real-valued tensor with shape (num_samples, 2)
        target = torch.view_as_real(
            torch.as_tensor(capture.pa_in, dtype=torch.complex64)
        ).to(model.device)
        # total number of samples in the training set
        num_samples = mag_feats.shape[0]
        # put the model in training mode, so that dropout is applied if configured
        net.train()

        for _ in range(self.epochs_per_step):
            # shuffle the samples for each epoch, if configured
            # no problem to shuffle the samples, the model is non-recurrent
            # we already have the memory taps in the features, 
            # so no need to carry hidden state
            order = (torch.randperm(num_samples, device=mag_feats.device)
                     if self.shuffle else torch.arange(num_samples, device=mag_feats.device))
            # loop over the samples in batches, each iteration select a batch of samples
            for start in range(0, num_samples, self.batch_size):
                idx = order[start:start + self.batch_size]
                self._optimizer.zero_grad()
                pred = net(mag_feats[idx], mag_taps[idx], cos_taps[idx], sin_taps[idx])
                loss = self.criterion(pred, target[idx])
                loss.backward()
                self._optimizer.step()
                self.last_loss = float(loss.item())