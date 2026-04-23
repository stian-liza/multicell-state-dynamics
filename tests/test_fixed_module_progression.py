from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_scp2154_fixed_module_progression import find_progression_lines, signature_gene_union


class FixedModuleProgressionTest(unittest.TestCase):
    def test_signature_gene_union_deduplicates_case_insensitive(self) -> None:
        signatures = {
            "A": {"sig1": ["Gene1", "Gene2"]},
            "B": {"sig2": ["gene2", "Gene3"]},
        }
        genes = signature_gene_union(signatures)
        self.assertEqual(genes, ["Gene1", "Gene2", "Gene3"])

    def test_find_progression_lines_prefers_secretory_stress_path(self) -> None:
        stages = [
            {"stage": "low_steatosis", "order": 1},
            {"stage": "cirrhotic", "order": 2},
            {"stage": "Tumor", "order": 3},
        ]
        chain_records = [
            {
                "stage": "cirrhotic",
                "source": "Myeloid.inflammatory_monocyte",
                "target": "Hepatocyte.secretory_stress",
                "direction_call": "forward_stronger",
                "forward_loo_delta_r2": 0.55,
                "reverse_loo_delta_r2": 0.20,
                "forward_empirical_p": 0.04,
                "reverse_empirical_p": 0.40,
                "forward_coeff": 0.70,
                "reverse_coeff": 0.10,
                "stage_consistency": "recurrent",
                "temporal_relation": "source_earlier",
            },
            {
                "stage": "Tumor",
                "source": "Hepatocyte.secretory_stress",
                "target": "Hepatocyte.malignant_like",
                "direction_call": "forward_stronger",
                "forward_loo_delta_r2": 0.80,
                "reverse_loo_delta_r2": 0.60,
                "forward_empirical_p": 0.02,
                "reverse_empirical_p": 0.10,
                "forward_coeff": 0.90,
                "reverse_coeff": 0.30,
                "stage_consistency": "stage_limited",
                "temporal_relation": "same_first_stage",
            },
            {
                "stage": "Tumor",
                "source": "Stromal.ecm_matrix",
                "target": "Hepatocyte.malignant_like",
                "direction_call": "forward_stronger",
                "forward_loo_delta_r2": 0.90,
                "reverse_loo_delta_r2": 0.10,
                "forward_empirical_p": 0.03,
                "reverse_empirical_p": 0.50,
                "forward_coeff": 0.80,
                "reverse_coeff": 0.00,
                "stage_consistency": "stage_limited",
                "temporal_relation": "same_first_stage",
            },
        ]

        lines = find_progression_lines(chain_records, stages, terminal_node="Hepatocyte.malignant_like", max_hops=3)

        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["contains_secretory_stress"], 1)
        self.assertEqual(
            lines[0]["path_edges"],
            [
                "Myeloid.inflammatory_monocyte->Hepatocyte.secretory_stress",
                "Hepatocyte.secretory_stress->Hepatocyte.malignant_like",
            ],
        )


if __name__ == "__main__":
    unittest.main()
