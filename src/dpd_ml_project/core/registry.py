"""Factory that builds a (predistorter, adapter) pair from a spec dict.

Extension point: add new architectures here without touching the pipeline.
Currently supports the existing methods only: GMP + {RLS, LS, Kalman} and GRU + SGD.
"""
from __future__ import annotations

from typing import Any


def build_dpd(spec: dict[str, Any]) -> tuple[Any, Any]:
    """Return ``(model, adapter)`` for the requested architecture.

    Spec keys:
        arch:    "gmp" (default) | "gru"
        adapter: "rls" (default) | "ls" | "kal" | "gru_sgd"
        lambda_rls: RLS forgetting factor (optional)
        ls_num_samples: LS window size (optional; None = all samples)
        r_kal, q_kal: Kalman measurement/process noise (optional)
        nn_config: NnConfig for the GRU arch (optional)
        lr, batch_size, epochs_per_step: GRU adapter knobs (optional)
    """
    arch = spec.get("arch", "gmp")
    adapter_name = spec.get("adapter", "rls")

    if arch == "gmp":
        from dpd_ml_project.core.per_arc_config import GmpConfig
        from dpd_ml_project.predistorters.gmp import GmpPredistorter
        model: Any = GmpPredistorter(spec.get("gmp_config") or GmpConfig())
    elif arch == "gru":
        from dpd_ml_project.core.per_arc_config import NnConfig
        from dpd_ml_project.predistorters.gru import GruPredistorter
        model = GruPredistorter(spec.get("nn_config") or NnConfig())
    else:
        raise ValueError(f"Unknown arch: {arch!r}")

    if adapter_name == "rls":
        from dpd_ml_project.adapters.rls import RlsAdapter
        adapter: Any = RlsAdapter(lambda_rls=spec.get("lambda_rls", 0.999))
    elif adapter_name == "ls":
        from dpd_ml_project.adapters.ls import LsAdapter
        adapter = LsAdapter(num_samples=spec.get("ls_num_samples"))
    elif adapter_name == "kal":
        from dpd_ml_project.adapters.kalman import KalmanAdapter
        adapter = KalmanAdapter(r_kal=spec.get("r_kal", 1e-6), q_kal=spec.get("q_kal", 1e-6))
    elif adapter_name == "gru_sgd":
        from dpd_ml_project.adapters.gru_sgd import GruAdapter
        nn_cfg = getattr(model, "cfg", None)
        adapter = GruAdapter(
            lr=spec.get("lr", getattr(nn_cfg, "lr", 2e-4)),
            batch_size=spec.get("batch_size", getattr(nn_cfg, "batch_size", 64)),
            epochs_per_step=spec.get("epochs_per_step", 1),
        )
    else:
        raise ValueError(f"Unknown adapter: {adapter_name!r}")

    return model, adapter
