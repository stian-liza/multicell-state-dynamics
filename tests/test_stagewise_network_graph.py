from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from render_scp2154_stagewise_network_graph import choose_edges, choose_nodes


class StagewiseNetworkGraphTest(unittest.TestCase):
    def test_choose_edges_filters_large_or_weak_edges(self) -> None:
        rows = [
            {
                "stage": "low_steatosis",
                "chain_source": "A",
                "chain_target": "B",
                "selected_delta_r2": "0.8",
                "selected_empirical_p": "0.04",
                "selected_coeff": "0.2",
                "stage_consistency": "recurrent",
                "selected_edge_stage_count": "2",
            },
            {
                "stage": "low_steatosis",
                "chain_source": "A",
                "chain_target": "C",
                "selected_delta_r2": "8.0",
                "selected_empirical_p": "0.01",
                "selected_coeff": "0.3",
                "stage_consistency": "stage_limited",
                "selected_edge_stage_count": "1",
            },
            {
                "stage": "low_steatosis",
                "chain_source": "B",
                "chain_target": "C",
                "selected_delta_r2": "0.2",
                "selected_empirical_p": "0.5",
                "selected_coeff": "0.1",
                "stage_consistency": "stage_limited",
                "selected_edge_stage_count": "1",
            },
        ]
        edges = choose_edges(
            rows,
            stages=["low_steatosis"],
            edge_min_delta=0.3,
            edge_max_p=0.12,
            max_abs_delta=5.0,
            top_edges_per_stage=5,
        )
        self.assertEqual(len(edges["low_steatosis"]), 1)
        self.assertEqual(edges["low_steatosis"][0]["source"], "A")
        self.assertEqual(edges["low_steatosis"][0]["target"], "B")

    def test_choose_nodes_keeps_high_delta_nodes(self) -> None:
        rows = [
            {
                "stage": "Tumor",
                "node_id": "Hepatocyte.malignant_like",
                "cell_type": "Hepatocyte",
                "signature": "malignant_like",
                "delta": "0.45",
                "abs_delta": "0.45",
                "first_altered_stage": "alcohol",
            },
            {
                "stage": "Tumor",
                "node_id": "Myeloid.c1qc_macrophage",
                "cell_type": "Myeloid",
                "signature": "c1qc_macrophage",
                "delta": "0.05",
                "abs_delta": "0.05",
                "first_altered_stage": "none",
            },
        ]
        nodes_by_id, stage_nodes = choose_nodes(rows, ["Tumor"], node_abs_delta_min=0.2)
        self.assertIn("Hepatocyte.malignant_like", nodes_by_id)
        self.assertEqual(len(stage_nodes["Tumor"]), 2)
        filtered = [node for node in stage_nodes["Tumor"] if node["passes_node_filter"]]
        self.assertEqual(len(filtered), 1)


if __name__ == "__main__":
    unittest.main()
