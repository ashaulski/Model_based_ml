"""Training-curve plotting for the neural DPD paths."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt


def plot_training_curve(
    history: Sequence[dict],
    *,
    title: str = "Training curve",
    output_dir: Path | None = None,
    show: bool = True,
) -> None:
    """Plot train/validation NMSE (dB) against epoch."""
    if not history:
        raise ValueError("history is empty - nothing to plot")

    epochs = [h["epoch"] for h in history]
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, [h["train_nmse_db"] for h in history], label="Train NMSE", color="blue")
    plt.plot(epochs, [h["val_nmse_db"] for h in history], label="Validation NMSE", color="orange")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("NMSE (dB)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    # if output_dir is not None:
    #     plt.savefig(Path(output_dir) / "training_curve.png", dpi=160)

    if show:
        plt.show()
