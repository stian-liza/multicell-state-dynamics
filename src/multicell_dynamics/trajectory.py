from __future__ import annotations

import numpy as np


def pca_embedding(x: np.ndarray, n_components: int = 2) -> np.ndarray:
    x_centered = x - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x_centered, full_matrices=False)
    n_components = min(n_components, u.shape[1])
    return u[:, :n_components] * s[:n_components]


def pseudotime_from_embedding(embedding: np.ndarray, axis: int = 0) -> np.ndarray:
    raw = embedding[:, axis]
    raw = raw - raw.min()
    denom = max(float(raw.max()), 1e-8)
    return raw / denom


def orient_pseudotime_by_labels(
    pseudotime: np.ndarray,
    labels: np.ndarray,
    low_label: str,
    high_label: str,
) -> np.ndarray:
    low_mask = labels == low_label
    high_mask = labels == high_label
    if np.any(low_mask) and np.any(high_mask):
        if float(np.mean(pseudotime[high_mask])) < float(np.mean(pseudotime[low_mask])):
            pseudotime = 1.0 - pseudotime
    return pseudotime


def local_direction_from_pseudotime(
    module_activity: np.ndarray,
    pseudotime: np.ndarray,
    k_neighbors: int = 15,
) -> np.ndarray:
    n_cells = len(pseudotime)
    velocity = np.zeros_like(module_activity, dtype=float)
    for idx in range(n_cells):
        delta_t = pseudotime - pseudotime[idx]
        spatial = np.abs(delta_t)
        forward = np.where(delta_t > 0)[0]
        if len(forward) == 0:
            nearest = np.argsort(spatial)[1 : k_neighbors + 1]
        else:
            nearest = forward[np.argsort(spatial[forward])[:k_neighbors]]
        if len(nearest) == 0:
            continue
        step = np.maximum((pseudotime[nearest] - pseudotime[idx])[:, None], 1e-4)
        delta = module_activity[nearest] - module_activity[idx]
        velocity[idx] = np.mean(delta / step, axis=0)
    return velocity
