from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_scp2154_stagewise_velocity_coupling import build_velocity_edges


class StagewiseVelocityCouplingTest(unittest.TestCase):
    def test_build_velocity_edges_prefers_forward_direction(self) -> None:
        stage_score_values = {
            "Tumor": {
                "A.sig": {"d1": 0.0, "d2": 1.0, "d3": 2.0, "d4": 3.0, "d5": 4.0},
                "B.sig": {"d1": -1.0, "d2": 0.5, "d3": 0.9, "d4": -0.8, "d5": -0.1},
            }
        }
        stage_velocity_values = {
            "Tumor": {
                "A.sig": {"d1": -0.8, "d2": -0.6, "d3": -0.3, "d4": -0.01, "d5": 0.3},
                "B.sig": {"d1": 0.0, "d2": 1.0, "d3": 2.0, "d4": 3.0, "d5": 4.0},
            }
        }
        first_stage = {
            "A.sig": {"first_stage": "low_steatosis", "first_stage_order": 1},
            "B.sig": {"first_stage": "Tumor", "first_stage_order": 3},
        }

        edges = build_velocity_edges(
            stage_score_values,
            stage_velocity_values,
            first_stage,
            alpha=0.1,
            permutations=20,
            margin=0.01,
            random_state=0,
        )

        self.assertEqual(len(edges), 1)
        edge = edges[0]
        self.assertEqual(edge["direction_call"], "forward_stronger")
        self.assertEqual(edge["chain_source"], "A.sig")
        self.assertEqual(edge["chain_target"], "B.sig")
        self.assertGreater(edge["forward_loo_delta_r2"], edge["reverse_loo_delta_r2"])


if __name__ == "__main__":
    unittest.main()
