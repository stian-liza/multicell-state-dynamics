from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def velocity_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual = np.sum((y_true - y_pred) ** 2)
    total = np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2)
    return float(1.0 - residual / total) if total > 0 else 0.0


def velocity_sign_agreement(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_pred) == np.sign(y_true)))


@dataclass
class SparseDynamicsModel:
    alpha: float
    coefficient_matrix: np.ndarray
    intercept: np.ndarray
    feature_names: list[str]
    target_names: list[str]
    train_r2: float

    def predict_velocity(
        self,
        module_activity: np.ndarray,
        state_embedding: np.ndarray,
        external_input: np.ndarray | None = None,
        genetics: np.ndarray | None = None,
    ) -> np.ndarray:
        design = _build_design_matrix(module_activity, state_embedding, external_input, genetics)
        return design @ self.coefficient_matrix.T + self.intercept

    def top_edges(self, top_k: int = 10) -> list[tuple[str, str, float]]:
        edges: list[tuple[str, str, float]] = []
        for target_idx, target_name in enumerate(self.target_names):
            for feature_idx, feature_name in enumerate(self.feature_names):
                weight = float(self.coefficient_matrix[target_idx, feature_idx])
                edges.append((feature_name, target_name, weight))
        edges.sort(key=lambda item: abs(item[2]), reverse=True)
        return edges[:top_k]


def _build_design_matrix(
    module_activity: np.ndarray,
    state_embedding: np.ndarray,
    external_input: np.ndarray | None = None,
    genetics: np.ndarray | None = None,
) -> np.ndarray:
    blocks = [module_activity, state_embedding]
    if external_input is not None:
        blocks.append(external_input)
    if genetics is not None:
        blocks.append(genetics)
    return np.concatenate(blocks, axis=1)


def fit_population_dynamics(
    module_activity: np.ndarray,
    state_embedding: np.ndarray,
    module_velocity: np.ndarray,
    external_input: np.ndarray | None = None,
    genetics: np.ndarray | None = None,
    alpha: float = 0.05,
) -> SparseDynamicsModel:
    if module_activity.shape != module_velocity.shape:
        raise ValueError("module_activity and module_velocity must have the same shape")

    design = _build_design_matrix(module_activity, state_embedding, external_input, genetics)
    centered_design = design - design.mean(axis=0, keepdims=True)
    centered_target = module_velocity - module_velocity.mean(axis=0, keepdims=True)
    gram = centered_design.T @ centered_design + alpha * np.eye(centered_design.shape[1])
    weights = np.linalg.solve(gram, centered_design.T @ centered_target).T
    weights[np.abs(weights) < alpha] = 0.0
    intercept = module_velocity.mean(axis=0) - design.mean(axis=0) @ weights.T
    prediction = design @ weights.T + intercept
    score = velocity_r2_score(module_velocity, prediction)

    n_modules = module_activity.shape[1]
    n_states = state_embedding.shape[1]
    feature_names = [f"m_{idx}" for idx in range(n_modules)] + [f"z_{idx}" for idx in range(n_states)]
    if external_input is not None:
        feature_names.extend([f"e_{idx}" for idx in range(external_input.shape[1])])
    if genetics is not None:
        feature_names.extend([f"g_{idx}" for idx in range(genetics.shape[1])])

    target_names = [f"dm_{idx}/dt" for idx in range(n_modules)]
    return SparseDynamicsModel(
        alpha=alpha,
        coefficient_matrix=weights,
        intercept=intercept,
        feature_names=feature_names,
        target_names=target_names,
        train_r2=score,
    )
