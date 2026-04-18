from __future__ import annotations

import numpy as np


def build_neighbor_average(
    module_activity: np.ndarray,
    cell_types: np.ndarray,
    target_type: str,
    source_type: str,
) -> np.ndarray:
    if module_activity.ndim != 2:
        raise ValueError("module_activity must be 2D")
    if len(cell_types) != len(module_activity):
        raise ValueError("cell_types length must match number of cells")

    source_mask = cell_types == source_type
    target_mask = cell_types == target_type
    if not np.any(source_mask):
        raise ValueError(f"source_type {source_type!r} not found")
    if not np.any(target_mask):
        raise ValueError(f"target_type {target_type!r} not found")

    source_mean = module_activity[source_mask].mean(axis=0, keepdims=True)
    external = np.zeros_like(module_activity)
    external[target_mask] = source_mean
    return external
