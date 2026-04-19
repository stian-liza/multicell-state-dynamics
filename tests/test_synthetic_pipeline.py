from __future__ import annotations

import unittest

import numpy as np

from multicell_dynamics import (
    build_neighbor_average,
    fit_module_representation,
    fit_population_dynamics,
    generate_synthetic_multicell_data,
    velocity_r2_score,
    velocity_sign_agreement,
)


class SyntheticPipelineTest(unittest.TestCase):
    @staticmethod
    def _split_indices(n_items: int, test_fraction: float = 0.25, random_state: int = 11) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(n_items)
        test_size = max(1, int(round(n_items * test_fraction)))
        test_idx = np.sort(indices[:test_size])
        train_idx = np.sort(indices[test_size:])
        return train_idx, test_idx

    def test_end_to_end_pipeline_runs(self) -> None:
        data = generate_synthetic_multicell_data(random_state=11)
        rep = fit_module_representation(data["features"], n_modules=4, random_state=11)
        external = build_neighbor_average(
            rep.module_activity,
            data["cell_types"],
            target_type="malignant",
            source_type="immune",
        )
        train_idx, test_idx = self._split_indices(len(data["features"]), test_fraction=0.25, random_state=11)
        model = fit_population_dynamics(
            module_activity=rep.module_activity[train_idx],
            state_embedding=data["state_embedding"][train_idx],
            module_velocity=data["module_velocity"][train_idx],
            external_input=external[train_idx],
            genetics=data["genetics"][train_idx],
            alpha=0.03,
        )
        predicted = model.predict_velocity(
            module_activity=rep.module_activity[test_idx],
            state_embedding=data["state_embedding"][test_idx],
            external_input=external[test_idx],
            genetics=data["genetics"][test_idx],
        )

        self.assertEqual(predicted.shape, data["module_velocity"][test_idx].shape)
        self.assertGreater(model.train_r2, 0.5)
        self.assertLess(rep.reconstruction_error, 0.2)

        test_r2 = velocity_r2_score(data["module_velocity"][test_idx], predicted)
        sign_agreement = velocity_sign_agreement(data["module_velocity"][test_idx], predicted)
        self.assertGreater(test_r2, 0.45)
        self.assertGreater(sign_agreement, 0.6)


if __name__ == "__main__":
    unittest.main()
