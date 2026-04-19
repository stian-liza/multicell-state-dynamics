from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ModuleRepresentation:
    module_activity: np.ndarray
    module_weights: np.ndarray
    reconstruction_error: float


def top_genes_per_module(
    module_weights: np.ndarray,
    gene_names: np.ndarray,
    top_k: int = 10,
) -> list[list[tuple[str, float]]]:
    if module_weights.ndim != 2:
        raise ValueError("module_weights must be a 2D array")
    if len(gene_names) != module_weights.shape[0]:
        raise ValueError("gene_names length must match number of genes")

    top: list[list[tuple[str, float]]] = []
    for module_idx in range(module_weights.shape[1]):
        weights = module_weights[:, module_idx]
        order = np.argsort(weights)[::-1][: min(top_k, len(weights))]
        top.append([(str(gene_names[idx]), float(weights[idx])) for idx in order])
    return top


def fit_module_representation(
    x: np.ndarray,
    n_modules: int,
    random_state: int = 0,
    max_iter: int = 800,
) -> ModuleRepresentation:
    if x.ndim != 2:
        raise ValueError("x must be a 2D array of shape (cells, features)")
    if n_modules < 2:
        raise ValueError("n_modules must be at least 2")

    rng = np.random.default_rng(random_state)
    x_shifted = np.maximum(x - np.min(x), 0.0)
    n_cells, n_features = x_shifted.shape
    module_activity = rng.uniform(0.0, 1.0, size=(n_cells, n_modules)) + 1e-3
    module_weights_t = rng.uniform(0.0, 1.0, size=(n_modules, n_features)) + 1e-3

    eps = 1e-8
    for _ in range(max_iter):
        numerator_h = module_activity.T @ x_shifted
        denominator_h = (module_activity.T @ module_activity @ module_weights_t) + eps
        module_weights_t *= numerator_h / denominator_h

        numerator_w = x_shifted @ module_weights_t.T
        denominator_w = (module_activity @ module_weights_t @ module_weights_t.T) + eps
        module_activity *= numerator_w / denominator_w

    recon = module_activity @ module_weights_t
    error = float(np.mean((x_shifted - recon) ** 2))
    return ModuleRepresentation(
        module_activity=module_activity,
        module_weights=module_weights_t.T,
        reconstruction_error=error,
    )
