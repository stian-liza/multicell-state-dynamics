from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_scp2154_chain_directionality import loo_prediction_test_multi


class ChainDirectionalityTest(unittest.TestCase):
    def test_multi_predictor_loo_prefers_true_signal(self) -> None:
        source = np.array([0.2, 0.5, 0.7, 1.0, 1.2, 1.5], dtype=float)
        mediator = source * 0.8 + np.array([0.0, 0.1, -0.05, 0.05, -0.02, 0.08], dtype=float)
        target = 1.3 * mediator + 0.2 * source + np.array([0.05, -0.02, 0.03, -0.01, 0.02, -0.03], dtype=float)

        source_only = loo_prediction_test_multi(source[:, None], target, alpha=0.01)
        joint = loo_prediction_test_multi(np.column_stack([source, mediator]), target, alpha=0.01)

        self.assertGreater(joint["loo_delta_r2"], source_only["loo_delta_r2"])
        self.assertEqual(joint["mean_coeffs"].shape, (2,))
        self.assertGreater(abs(float(joint["mean_coeffs"][1])), abs(float(joint["mean_coeffs"][0])))


if __name__ == "__main__":
    unittest.main()
