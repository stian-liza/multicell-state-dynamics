from __future__ import annotations

import unittest

import numpy as np

from multicell_dynamics import (
    build_neighbor_average,
    fit_module_representation,
    fit_population_dynamics,
    generate_synthetic_multicell_data,
)


class SyntheticPipelineTest(unittest.TestCase):
    def test_end_to_end_pipeline_runs(self) -> None:
        data = generate_synthetic_multicell_data(random_state=11)
        rep = fit_module_representation(data["features"], n_modules=4, random_state=11)
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
        predicted = model.predict_velocity(
            module_activity=rep.module_activity,
            state_embedding=data["state_embedding"],
            external_input=external,
            genetics=data["genetics"],
        )

        self.assertEqual(predicted.shape, data["module_velocity"].shape)
        self.assertGreater(model.train_r2, 0.5)
        self.assertLess(rep.reconstruction_error, 0.2)

        sign_agreement = float(np.mean(np.sign(predicted) == np.sign(data["module_velocity"])))
        self.assertGreater(sign_agreement, 0.65)


if __name__ == "__main__":
    unittest.main()
