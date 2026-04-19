from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_scp2154_stagewise_bidirectional_network import attach_stage_consistency, select_stage_chain


class StagewiseNetworkTest(unittest.TestCase):
    def test_attach_stage_consistency_marks_recurrent_edges(self) -> None:
        edges = [
            {
                "stage": "low_steatosis",
                "source": "Myeloid.inflammatory_monocyte",
                "target": "Hepatocyte.secretory_stress",
                "direction_call": "forward_stronger",
            },
            {
                "stage": "cirrhotic",
                "source": "Myeloid.inflammatory_monocyte",
                "target": "Hepatocyte.secretory_stress",
                "direction_call": "forward_stronger",
            },
            {
                "stage": "Tumor",
                "source": "Stromal.ecm_matrix",
                "target": "Hepatocyte.malignant_like",
                "direction_call": "bidirectional_or_shared_stage",
            },
        ]
        attach_stage_consistency(edges)

        self.assertEqual(edges[0]["selected_edge"], "Myeloid.inflammatory_monocyte->Hepatocyte.secretory_stress")
        self.assertEqual(edges[0]["selected_edge_stage_count"], 2)
        self.assertEqual(edges[0]["stage_consistency"], "recurrent")
        self.assertEqual(edges[1]["selected_edge_stages"], "cirrhotic,low_steatosis")
        self.assertEqual(edges[2]["selected_edge"], "none")
        self.assertEqual(edges[2]["stage_consistency"], "shared_or_weak")

    def test_select_stage_chain_filters_target_earlier_edges(self) -> None:
        edges = [
            {
                "stage": "cirrhotic",
                "source": "A",
                "target": "B",
                "forward_edge": "A->B",
                "reverse_edge": "B->A",
                "forward_loo_delta_r2": 0.42,
                "reverse_loo_delta_r2": 0.05,
                "forward_minus_reverse": 0.37,
                "forward_coeff": 0.8,
                "reverse_coeff": -0.1,
                "forward_relation": "positive_predictive_relation",
                "reverse_relation": "no_heldout_gain",
                "forward_empirical_p": 0.04,
                "reverse_empirical_p": 0.6,
                "forward_pearson_r": 0.7,
                "reverse_pearson_r": -0.1,
                "direction_call": "forward_stronger",
                "n_donors": 8,
                "donors": "d1,d2,d3,d4,d5,d6,d7,d8",
                "source_first_stage": "low_steatosis",
                "target_first_stage": "cirrhotic",
                "temporal_relation": "source_earlier",
                "chain_source": "A",
                "chain_target": "B",
                "source_stage_appearance": "low_steatosis,cirrhotic",
                "target_stage_appearance": "cirrhotic",
                "selected_edge": "A->B",
                "selected_edge_stage_count": 2,
                "selected_edge_stages": "cirrhotic,low_steatosis",
                "stage_consistency": "recurrent",
            },
            {
                "stage": "cirrhotic",
                "source": "C",
                "target": "D",
                "forward_edge": "C->D",
                "reverse_edge": "D->C",
                "forward_loo_delta_r2": 0.5,
                "reverse_loo_delta_r2": 0.02,
                "forward_minus_reverse": 0.48,
                "forward_coeff": 0.9,
                "reverse_coeff": 0.0,
                "forward_relation": "positive_predictive_relation",
                "reverse_relation": "no_heldout_gain",
                "forward_empirical_p": 0.03,
                "reverse_empirical_p": 0.8,
                "forward_pearson_r": 0.8,
                "reverse_pearson_r": 0.0,
                "direction_call": "forward_stronger",
                "n_donors": 8,
                "donors": "d1,d2,d3,d4,d5,d6,d7,d8",
                "source_first_stage": "Tumor",
                "target_first_stage": "cirrhotic",
                "temporal_relation": "target_earlier",
                "chain_source": "C",
                "chain_target": "D",
                "source_stage_appearance": "Tumor",
                "target_stage_appearance": "cirrhotic",
                "selected_edge": "C->D",
                "selected_edge_stage_count": 1,
                "selected_edge_stages": "cirrhotic",
                "stage_consistency": "stage_limited",
            },
        ]

        chain = select_stage_chain(edges, min_delta=0.2, max_p=0.15)

        self.assertEqual(len(chain), 1)
        self.assertEqual(chain[0]["selected_edge"], "A->B")
        self.assertAlmostEqual(chain[0]["selected_delta_r2"], 0.42)

    def test_select_stage_chain_can_require_recurrence_and_min_donors(self) -> None:
        edges = [
            {
                "stage": "low_steatosis",
                "source": "A",
                "target": "B",
                "forward_edge": "A->B",
                "reverse_edge": "B->A",
                "forward_loo_delta_r2": 0.55,
                "reverse_loo_delta_r2": 0.10,
                "forward_minus_reverse": 0.45,
                "forward_coeff": 0.7,
                "reverse_coeff": 0.1,
                "forward_relation": "positive_predictive_relation",
                "reverse_relation": "no_heldout_gain",
                "forward_empirical_p": 0.03,
                "reverse_empirical_p": 0.8,
                "forward_pearson_r": 0.7,
                "reverse_pearson_r": 0.1,
                "direction_call": "forward_stronger",
                "n_donors": 7,
                "donors": "d1,d2,d3,d4,d5,d6,d7",
                "source_first_stage": "low_steatosis",
                "target_first_stage": "cirrhotic",
                "temporal_relation": "source_earlier",
                "chain_source": "A",
                "chain_target": "B",
                "source_stage_appearance": "low_steatosis,cirrhotic",
                "target_stage_appearance": "cirrhotic,Tumor",
                "selected_edge": "A->B",
                "selected_edge_stage_count": 2,
                "selected_edge_stages": "cirrhotic,low_steatosis",
                "stage_consistency": "recurrent",
            },
            {
                "stage": "Tumor",
                "source": "C",
                "target": "D",
                "forward_edge": "C->D",
                "reverse_edge": "D->C",
                "forward_loo_delta_r2": 0.70,
                "reverse_loo_delta_r2": 0.20,
                "forward_minus_reverse": 0.50,
                "forward_coeff": 0.8,
                "reverse_coeff": 0.1,
                "forward_relation": "positive_predictive_relation",
                "reverse_relation": "no_heldout_gain",
                "forward_empirical_p": 0.02,
                "reverse_empirical_p": 0.5,
                "forward_pearson_r": 0.8,
                "reverse_pearson_r": 0.2,
                "direction_call": "forward_stronger",
                "n_donors": 5,
                "donors": "d1,d2,d3,d4,d5",
                "source_first_stage": "cirrhotic",
                "target_first_stage": "Tumor",
                "temporal_relation": "source_earlier",
                "chain_source": "C",
                "chain_target": "D",
                "source_stage_appearance": "Tumor",
                "target_stage_appearance": "Tumor",
                "selected_edge": "C->D",
                "selected_edge_stage_count": 1,
                "selected_edge_stages": "Tumor",
                "stage_consistency": "stage_limited",
            },
        ]

        chain = select_stage_chain(edges, min_delta=0.2, max_p=0.05, min_edge_donors=6, require_recurrent=True)

        self.assertEqual(len(chain), 1)
        self.assertEqual(chain[0]["selected_edge"], "A->B")


if __name__ == "__main__":
    unittest.main()
