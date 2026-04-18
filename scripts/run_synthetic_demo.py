from __future__ import annotations

import numpy as np

from multicell_dynamics import (
    build_neighbor_average,
    fit_module_representation,
    fit_population_dynamics,
    generate_synthetic_multicell_data,
)


def main() -> None:
    data = generate_synthetic_multicell_data(random_state=7)
    rep = fit_module_representation(data["features"], n_modules=4, random_state=7)
    external = build_neighbor_average(
        rep.module_activity,
        data["cell_types"],
        target_type="malignant",
        source_type="immune",
    )
    model = fit_population_dynamics(
        module_activity=rep.module_activity,
        state_embedding=data["state_embedding"],
        module_velocity=data["module_velocity"],
        external_input=external,
        genetics=data["genetics"],
        alpha=0.03,
    )

    print("Reconstruction error:", round(rep.reconstruction_error, 4))
    print("Training R^2:", round(model.train_r2, 4))
    print("Top inferred edges:")
    for source, target, weight in model.top_edges(top_k=8):
        print(f"  {source:>6} -> {target:<8} {weight:+.3f}")

    predicted = model.predict_velocity(
        module_activity=rep.module_activity,
        state_embedding=data["state_embedding"],
        external_input=external,
        genetics=data["genetics"],
    )
    sign_agreement = np.mean(np.sign(predicted) == np.sign(data["module_velocity"]))
    print("Velocity sign agreement:", round(float(sign_agreement), 4))


if __name__ == "__main__":
    main()
