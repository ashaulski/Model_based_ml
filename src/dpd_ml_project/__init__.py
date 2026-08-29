"""DPD ML project package.

Architectures plug into one generic pipeline via ``core.registry.build_dpd``:
    * predistorters: GMP (linear), GRU (neural)
    * adapters:      LS / RLS / Kalman (linear), SGD (neural)

See ``dpd_ml_project.experiments`` for runnable drivers.
"""
from __future__ import annotations
