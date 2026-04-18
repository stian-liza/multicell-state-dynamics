from __future__ import annotations

import numpy as np


def generate_synthetic_multicell_data(
    n_cells: int = 240,
    n_features: int = 30,
    n_modules: int = 4,
    n_states: int = 2,
    random_state: int = 0,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(random_state)
    cell_types = np.array(["malignant"] * (n_cells // 2) + ["immune"] * (n_cells - n_cells // 2))

    base_modules = rng.normal(size=(n_cells, n_modules))
    base_modules[cell_types == "malignant", 0] += 1.5
    base_modules[cell_types == "immune", 1] += 1.5

    state_embedding = base_modules[:, :n_states] + 0.15 * rng.normal(size=(n_cells, n_states))
    genetics = np.zeros((n_cells, 1))
    genetics[cell_types == "malignant", 0] = 1.0

    source_mean = base_modules[cell_types == "immune"].mean(axis=0, keepdims=True)
    external_input = np.zeros_like(base_modules)
    external_input[cell_types == "malignant"] = source_mean

    a_true = np.array(
        [
            [0.60, -0.25, 0.00, 0.10],
            [0.00, 0.40, -0.20, 0.00],
            [0.20, 0.10, 0.35, -0.15],
            [0.00, 0.25, 0.15, 0.45],
        ]
    )
    b_true = np.array(
        [
            [0.30, 0.00],
            [0.00, 0.20],
            [0.15, -0.10],
            [0.05, 0.20],
        ]
    )
    c_true = np.array(
        [
            [0.25, 0.00, 0.00, 0.00],
            [0.00, 0.15, 0.00, 0.10],
            [0.00, 0.00, 0.00, 0.00],
            [0.10, 0.00, 0.00, 0.20],
        ]
    )
    d_true = np.array([[0.20], [0.00], [0.10], [0.05]])

    module_velocity = (
        base_modules @ a_true.T
        + state_embedding @ b_true.T
        + external_input @ c_true.T
        + genetics @ d_true.T
        + 0.08 * rng.normal(size=(n_cells, n_modules))
    )

    mixing = rng.uniform(0.0, 1.0, size=(n_modules, n_features))
    features = np.maximum(base_modules, 0.0) @ mixing + 0.05 * rng.normal(size=(n_cells, n_features))
    features = np.maximum(features, 0.0)

    return {
        "features": features,
        "module_activity": base_modules,
        "state_embedding": state_embedding,
        "module_velocity": module_velocity,
        "external_input": external_input,
        "genetics": genetics,
        "cell_types": cell_types,
    }
