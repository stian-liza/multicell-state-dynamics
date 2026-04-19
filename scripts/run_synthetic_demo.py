from __future__ import annotations

import numpy as np

from multicell_dynamics import (
    build_neighbor_average,
    fit_module_representation,
    fit_population_dynamics,
    generate_synthetic_multicell_data,
    velocity_r2_score,
    velocity_sign_agreement,
)


def make_train_test_split(n_items: int, test_fraction: float = 0.25, random_state: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n_items)
    test_size = max(1, int(round(n_items * test_fraction)))
    test_idx = np.sort(indices[:test_size])
    train_idx = np.sort(indices[test_size:])
    return train_idx, test_idx


def main() -> None:
    data = generate_synthetic_multicell_data(random_state=7)
    rep = fit_module_representation(data["features"], n_modules=4, random_state=7)
    external = build_neighbor_average(
        rep.module_activity,
        data["cell_types"],
        target_type="malignant",
        source_type="immune",
    )
    train_idx, test_idx = make_train_test_split(len(data["features"]), test_fraction=0.25, random_state=7)
    model = fit_population_dynamics(
        module_activity=rep.module_activity[train_idx],
        state_embedding=data["state_embedding"][train_idx],
        module_velocity=data["module_velocity"][train_idx],
        external_input=external[train_idx],
        genetics=data["genetics"][train_idx],
        alpha=0.03,
    )

    print("Reconstruction error:", round(rep.reconstruction_error, 4))
    print("Train cells:", len(train_idx))
    print("Test cells:", len(test_idx))
    print("Training R^2:", round(model.train_r2, 4))
    print("Top inferred edges:")
    for source, target, weight in model.top_edges(top_k=8):
        print(f"  {source:>6} -> {target:<8} {weight:+.3f}")

    predicted_train = model.predict_velocity(
        module_activity=rep.module_activity[train_idx],
        state_embedding=data["state_embedding"][train_idx],
        external_input=external[train_idx],
        genetics=data["genetics"][train_idx],
    )
    predicted_test = model.predict_velocity(
        module_activity=rep.module_activity[test_idx],
        state_embedding=data["state_embedding"][test_idx],
        external_input=external[test_idx],
        genetics=data["genetics"][test_idx],
    )
    train_sign = velocity_sign_agreement(data["module_velocity"][train_idx], predicted_train)
    test_r2 = velocity_r2_score(data["module_velocity"][test_idx], predicted_test)
    test_sign = velocity_sign_agreement(data["module_velocity"][test_idx], predicted_test)
    print("Train sign agreement:", round(train_sign, 4))
    print("Test R^2:", round(test_r2, 4))
    print("Test sign agreement:", round(test_sign, 4))


if __name__ == "__main__":
    main()
