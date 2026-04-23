from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from run_scp2154_coupling_driver_scan import DRIVER_SIGNATURES, unique_rows
from run_scp2154_fixed_signature_tumor_validation import (
    HEPATOCYTE_SIGNATURES,
    empirical_p_value,
    loo_prediction_test,
    safe_corr,
    signature_scores,
)
from run_scp2154_phenotype_baseline import read_metadata, stratified_sample
from run_scp2154_stromal_to_hepatocyte_coupling import save_text
from multicell_dynamics import log1p_library_normalize, read_10x_mtx_subset


STAGE_DEFS = [
    {"stage": "low_steatosis", "column": "health", "value": "low_steatosis", "order": 1},
    {"stage": "NAFLD", "column": "indication", "value": "NAFLD", "order": 1},
    {"stage": "alcohol", "column": "indication", "value": "alcohol", "order": 1},
    {"stage": "cirrhotic", "column": "health", "value": "cirrhotic", "order": 2},
    {"stage": "Tumor", "column": "indication", "value": "Tumor", "order": 3},
]


def all_signatures() -> dict[str, dict[str, list[str]]]:
    signatures = dict(DRIVER_SIGNATURES)
    signatures["Hepatocyte"] = HEPATOCYTE_SIGNATURES
    return signatures


def parse_stages(raw: str) -> list[dict]:
    wanted = {item.strip() for item in raw.split(",") if item.strip()}
    if not wanted:
        return list(STAGE_DEFS)
    stages = [stage for stage in STAGE_DEFS if stage["stage"] in wanted]
    missing = wanted - {stage["stage"] for stage in stages}
    if missing:
        raise ValueError(f"Unknown stages: {', '.join(sorted(missing))}")
    return stages


def sample_celltype_stage_rows(
    rows: list[dict[str, str]],
    cell_type: str,
    stages: list[dict],
    max_cells_per_donor_stage: int,
    max_cells_per_stage: int,
    random_state: int,
) -> list[dict[str, str]]:
    sampled = []
    for stage_idx, stage in enumerate(stages):
        column = stage["column"]
        value = stage["value"]
        candidates = []
        for row in rows:
            if row.get("organ__ontology_label") != "liver":
                continue
            if row.get("Cell_Type") != cell_type:
                continue
            if row.get(column) not in {"healthy", value}:
                continue
            copied = dict(row)
            copied["_stage_sample_label"] = row[column]
            candidates.append(copied)
        sampled.extend(
            stratified_sample(
                candidates,
                phenotype_col="_stage_sample_label",
                donor_col="donor_id",
                max_cells_per_donor_phenotype=max_cells_per_donor_stage,
                max_cells_per_phenotype=max_cells_per_stage,
                random_state=random_state + stage_idx * 101,
            )
        )
    return unique_rows(sampled)


def celltype_matrix(counts, rows: list[dict[str, str]]) -> tuple[np.ndarray, list[dict[str, str]]]:
    cell_to_index = {cell: idx for idx, cell in enumerate(counts.cell_barcodes)}
    kept = [row for row in rows if row["NAME"] in cell_to_index]
    idx = np.array([cell_to_index[row["NAME"]] for row in kept], dtype=int)
    return counts.matrix[idx], kept


def donor_score_means(
    rows: list[dict[str, str]],
    values: np.ndarray,
    stage: dict,
    label: str,
    min_cells: int,
) -> tuple[dict[str, float], dict[str, int]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    column = stage["column"]
    condition_value = "healthy" if label == "healthy" else stage["value"]
    for idx, row in enumerate(rows):
        if row.get(column) == condition_value:
            grouped[row["donor_id"]].append(float(values[idx]))
    means = {}
    counts = {}
    for donor, donor_values in grouped.items():
        if len(donor_values) < min_cells:
            continue
        means[donor] = float(np.mean(donor_values))
        counts[donor] = len(donor_values)
    return means, counts


def stage_node_tables(
    rows_by_cell_type: dict[str, list[dict[str, str]]],
    scores_by_node: dict[str, np.ndarray],
    stages: list[dict],
    min_cells: int,
) -> tuple[list[dict], dict[str, dict[str, dict[str, float]]]]:
    """Aggregate cell-level signature scores into donor-level stage tables.

    For each node (cell_type.signature) and each stage:
    - compute donor means in healthy cells
    - compute donor means in condition cells for that stage
    - store stage delta = mean(condition donors) - mean(healthy donors)

    These donor means are the basic unit for the later directional tests.
    """
    node_records = []
    stage_node_values: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for node_id, values in scores_by_node.items():
        cell_type, signature = node_id.split(".", 1)
        rows = rows_by_cell_type[cell_type]
        for stage in stages:
            condition_means, condition_counts = donor_score_means(rows, values, stage, "condition", min_cells)
            healthy_means, healthy_counts = donor_score_means(rows, values, stage, "healthy", min_cells)
            if not condition_means or not healthy_means:
                continue
            condition_values = np.array(list(condition_means.values()), dtype=float)
            healthy_values = np.array(list(healthy_means.values()), dtype=float)
            condition_mean = float(condition_values.mean())
            healthy_mean = float(healthy_values.mean())
            delta = condition_mean - healthy_mean
            stage_node_values[stage["stage"]][node_id] = condition_means
            node_records.append(
                {
                    "stage": stage["stage"],
                    "stage_order": stage["order"],
                    "node_id": node_id,
                    "cell_type": cell_type,
                    "signature": signature,
                    "condition_mean": condition_mean,
                    "healthy_mean": healthy_mean,
                    "delta": delta,
                    "abs_delta": abs(delta),
                    "condition_donors": len(condition_means),
                    "healthy_donors": len(healthy_means),
                    "condition_cells": int(sum(condition_counts.values())),
                    "healthy_cells": int(sum(healthy_counts.values())),
                }
            )
    return node_records, stage_node_values


def first_altered_stage(node_records: list[dict], delta_threshold: float) -> dict[str, dict]:
    """Assign the earliest stage where a node changes beyond |delta| threshold."""
    out = {}
    for record in sorted(node_records, key=lambda item: (item["stage_order"], -item["abs_delta"])):
        if record["abs_delta"] < delta_threshold:
            continue
        node_id = record["node_id"]
        if node_id not in out:
            out[node_id] = {
                "first_stage": record["stage"],
                "first_stage_order": record["stage_order"],
                "first_delta": record["delta"],
            }
    return out


def evaluate_pair(
    source_values: dict[str, float],
    target_values: dict[str, float],
    source_name: str,
    target_name: str,
    alpha: float,
    permutations: int,
    rng: np.random.Generator,
) -> dict | None:
    """Test whether source donor scores predict target donor scores.

    Input
    - source_values: donor -> source node score
    - target_values: donor -> target node score

    Output
    - held-out ridge metrics including loo_delta_r2
    - permutation null summary

    Formula summary
    - Fit ridge regression y ~ x in leave-one-donor-out fashion
    - Compare model R^2 to a baseline that predicts the training mean only
    - loo_delta_r2 = loo_r2 - loo_baseline_r2
    """
    donors = sorted(set(source_values) & set(target_values))
    if len(donors) < 5:
        return None
    x = np.array([[source_values[donor]] for donor in donors], dtype=float)
    y = np.array([target_values[donor] for donor in donors], dtype=float)
    observed = loo_prediction_test(x, y, alpha)
    null_values = []
    for _ in range(permutations):
        permuted = x[rng.permutation(len(x))]
        null_values.append(float(loo_prediction_test(permuted, y, alpha)["loo_delta_r2"]))
    null_array = np.array(null_values, dtype=float)
    return {
        "source": source_name,
        "target": target_name,
        "n_donors": len(donors),
        "loo_delta_r2": float(observed["loo_delta_r2"]),
        "loo_r2": float(observed["loo_r2"]),
        "loo_baseline_r2": float(observed["loo_baseline_r2"]),
        "mean_coeff": float(observed["mean_coeff"]),
        "pearson_r": safe_corr(x.ravel(), y),
        "empirical_p": empirical_p_value(float(observed["loo_delta_r2"]), null_array),
        "null_mean": float(null_array.mean()),
        "null_95pct": float(np.quantile(null_array, 0.95)),
        "donors": ",".join(donors),
    }


def relation_label(coeff: float, delta_r2: float) -> str:
    if delta_r2 <= 0:
        return "no_heldout_gain"
    if coeff > 0:
        return "positive_predictive_relation"
    if coeff < 0:
        return "negative_predictive_relation"
    return "zero_coefficient"


def direction_call(forward_delta: float, reverse_delta: float, margin: float) -> str:
    if forward_delta > margin and forward_delta > reverse_delta + margin:
        return "forward_stronger"
    if reverse_delta > margin and reverse_delta > forward_delta + margin:
        return "reverse_stronger"
    if forward_delta > margin and reverse_delta > margin:
        return "bidirectional_or_shared_stage"
    return "weak_or_inconclusive"


def temporal_relation(source_info: dict | None, target_info: dict | None) -> str:
    if source_info is None or target_info is None:
        return "unknown"
    if source_info["first_stage_order"] < target_info["first_stage_order"]:
        return "source_earlier"
    if source_info["first_stage_order"] == target_info["first_stage_order"]:
        return "same_first_stage"
    return "target_earlier"


def build_bidirectional_edges(
    stage_node_values: dict[str, dict[str, dict[str, float]]],
    first_stage: dict[str, dict],
    alpha: float,
    permutations: int,
    margin: float,
    random_state: int,
) -> list[dict]:
    rng = np.random.default_rng(random_state)
    edge_records = []
    for stage, node_values in stage_node_values.items():
        nodes = sorted(node_values)
        for source_idx, source in enumerate(nodes):
            for target in nodes[source_idx + 1 :]:
                forward = evaluate_pair(node_values[source], node_values[target], source, target, alpha, permutations, rng)
                reverse = evaluate_pair(node_values[target], node_values[source], target, source, alpha, permutations, rng)
                if forward is None or reverse is None:
                    continue
                source_first = first_stage.get(source)
                target_first = first_stage.get(target)
                call = direction_call(forward["loo_delta_r2"], reverse["loo_delta_r2"], margin)
                if call == "reverse_stronger":
                    chain_source = target
                    chain_target = source
                    chain_source_first = target_first
                    chain_target_first = source_first
                else:
                    chain_source = source
                    chain_target = target
                    chain_source_first = source_first
                    chain_target_first = target_first
                edge_records.append(
                    {
                        "stage": stage,
                        "source": source,
                        "target": target,
                        "forward_edge": f"{source}->{target}",
                        "reverse_edge": f"{target}->{source}",
                        "forward_loo_delta_r2": forward["loo_delta_r2"],
                        "reverse_loo_delta_r2": reverse["loo_delta_r2"],
                        "forward_minus_reverse": forward["loo_delta_r2"] - reverse["loo_delta_r2"],
                        "forward_coeff": forward["mean_coeff"],
                        "reverse_coeff": reverse["mean_coeff"],
                        "forward_relation": relation_label(forward["mean_coeff"], forward["loo_delta_r2"]),
                        "reverse_relation": relation_label(reverse["mean_coeff"], reverse["loo_delta_r2"]),
                        "forward_empirical_p": forward["empirical_p"],
                        "reverse_empirical_p": reverse["empirical_p"],
                        "forward_pearson_r": forward["pearson_r"],
                        "reverse_pearson_r": reverse["pearson_r"],
                        "direction_call": call,
                        "n_donors": forward["n_donors"],
                        "donors": forward["donors"],
                        "source_first_stage": source_first["first_stage"] if source_first else "none",
                        "target_first_stage": target_first["first_stage"] if target_first else "none",
                        "temporal_relation": temporal_relation(chain_source_first, chain_target_first),
                        "chain_source": chain_source,
                        "chain_target": chain_target,
                    }
                )
    edge_records.sort(
        key=lambda item: (
            item["direction_call"] in {"forward_stronger", "reverse_stronger"},
            max(item["forward_loo_delta_r2"], item["reverse_loo_delta_r2"]),
            abs(item["forward_minus_reverse"]),
        ),
        reverse=True,
    )
    return edge_records


def select_stage_chain(
    edges: list[dict],
    min_delta: float,
    max_p: float,
    min_edge_donors: int = 5,
    require_recurrent: bool = False,
) -> list[dict]:
    chain = []
    for edge in edges:
        if edge["direction_call"] == "forward_stronger":
            selected_delta = edge["forward_loo_delta_r2"]
            selected_p = edge["forward_empirical_p"]
            selected_coeff = edge["forward_coeff"]
        elif edge["direction_call"] == "reverse_stronger":
            selected_delta = edge["reverse_loo_delta_r2"]
            selected_p = edge["reverse_empirical_p"]
            selected_coeff = edge["reverse_coeff"]
        else:
            continue
        if selected_delta < min_delta or selected_p > max_p:
            continue
        if edge["n_donors"] < min_edge_donors:
            continue
        if edge["temporal_relation"] == "target_earlier":
            continue
        if require_recurrent and edge.get("stage_consistency") != "recurrent":
            continue
        item = dict(edge)
        item["selected_delta_r2"] = selected_delta
        item["selected_empirical_p"] = selected_p
        item["selected_coeff"] = selected_coeff
        chain.append(item)
    chain.sort(
        key=lambda item: (
            item["temporal_relation"] == "source_earlier",
            item["selected_delta_r2"],
            abs(item["selected_coeff"]),
        ),
        reverse=True,
    )
    return chain


def attach_stage_consistency(edge_records: list[dict]) -> None:
    """Annotate whether the selected direction repeats across multiple stages."""
    selected_stage_map: dict[tuple[str, str], list[str]] = defaultdict(list)
    source_stage_map: dict[str, list[str]] = defaultdict(list)
    target_stage_map: dict[str, list[str]] = defaultdict(list)
    for edge in edge_records:
        source_stage_map[edge["source"]].append(edge["stage"])
        target_stage_map[edge["target"]].append(edge["stage"])
        if edge["direction_call"] == "forward_stronger":
            selected_stage_map[(edge["source"], edge["target"])].append(edge["stage"])
        elif edge["direction_call"] == "reverse_stronger":
            selected_stage_map[(edge["target"], edge["source"])].append(edge["stage"])

    for edge in edge_records:
        edge["source_stage_appearance"] = ",".join(sorted(set(source_stage_map[edge["source"]])))
        edge["target_stage_appearance"] = ",".join(sorted(set(target_stage_map[edge["target"]])))
        if edge["direction_call"] == "forward_stronger":
            selected_key = (edge["source"], edge["target"])
            edge["selected_edge"] = f"{edge['source']}->{edge['target']}"
        elif edge["direction_call"] == "reverse_stronger":
            selected_key = (edge["target"], edge["source"])
            edge["selected_edge"] = f"{edge['target']}->{edge['source']}"
        else:
            selected_key = None
            edge["selected_edge"] = "none"

        if selected_key is None:
            edge["selected_edge_stage_count"] = 0
            edge["selected_edge_stages"] = "none"
            edge["stage_consistency"] = "shared_or_weak"
            continue

        stages = sorted(set(selected_stage_map[selected_key]))
        edge["selected_edge_stage_count"] = len(stages)
        edge["selected_edge_stages"] = ",".join(stages)
        edge["stage_consistency"] = "recurrent" if len(stages) >= 2 else "stage_limited"


def write_outputs(
    output_dir: Path,
    node_records: list[dict],
    first_stage: dict[str, dict],
    edge_records: list[dict],
    chain_records: list[dict],
    signature_lines: list[str],
    summary_lines: list[str],
) -> None:
    node_lines = [
        "stage\tstage_order\tnode_id\tcell_type\tsignature\tcondition_mean\thealthy_mean\t"
        "delta\tabs_delta\tcondition_donors\thealthy_donors\tcondition_cells\thealthy_cells\t"
        "first_altered_stage\tfirst_altered_delta"
    ]
    for record in sorted(node_records, key=lambda item: (item["stage_order"], item["stage"], -item["abs_delta"])):
        first = first_stage.get(record["node_id"])
        node_lines.append(
            "\t".join(
                [
                    record["stage"],
                    str(record["stage_order"]),
                    record["node_id"],
                    record["cell_type"],
                    record["signature"],
                    f"{record['condition_mean']:.6f}",
                    f"{record['healthy_mean']:.6f}",
                    f"{record['delta']:.6f}",
                    f"{record['abs_delta']:.6f}",
                    str(record["condition_donors"]),
                    str(record["healthy_donors"]),
                    str(record["condition_cells"]),
                    str(record["healthy_cells"]),
                    first["first_stage"] if first else "none",
                    f"{first['first_delta']:.6f}" if first else "nan",
                ]
            )
        )

    edge_header = (
        "rank\tstage\tsource\ttarget\tforward_edge\treverse_edge\tforward_loo_delta_r2\t"
        "reverse_loo_delta_r2\tforward_minus_reverse\tforward_coeff\treverse_coeff\t"
        "forward_relation\treverse_relation\tforward_empirical_p\treverse_empirical_p\t"
        "forward_pearson_r\treverse_pearson_r\tdirection_call\tn_donors\tdonors\t"
        "source_first_stage\ttarget_first_stage\ttemporal_relation\tchain_source\tchain_target\t"
        "source_stage_appearance\ttarget_stage_appearance\tselected_edge\tselected_edge_stage_count\t"
        "selected_edge_stages\tstage_consistency"
    )
    edge_lines = [edge_header]
    for rank, record in enumerate(edge_records, start=1):
        edge_lines.append(
            "\t".join(
                [
                    str(rank),
                    record["stage"],
                    record["source"],
                    record["target"],
                    record["forward_edge"],
                    record["reverse_edge"],
                    f"{record['forward_loo_delta_r2']:.6f}",
                    f"{record['reverse_loo_delta_r2']:.6f}",
                    f"{record['forward_minus_reverse']:.6f}",
                    f"{record['forward_coeff']:.6f}",
                    f"{record['reverse_coeff']:.6f}",
                    record["forward_relation"],
                    record["reverse_relation"],
                    f"{record['forward_empirical_p']:.6f}",
                    f"{record['reverse_empirical_p']:.6f}",
                    f"{record['forward_pearson_r']:.6f}",
                    f"{record['reverse_pearson_r']:.6f}",
                    record["direction_call"],
                    str(record["n_donors"]),
                    record["donors"],
                    record["source_first_stage"],
                    record["target_first_stage"],
                    record["temporal_relation"],
                    record["chain_source"],
                    record["chain_target"],
                    record["source_stage_appearance"],
                    record["target_stage_appearance"],
                    record["selected_edge"],
                    str(record["selected_edge_stage_count"]),
                    record["selected_edge_stages"],
                    record["stage_consistency"],
                ]
            )
        )

    chain_lines = [
        edge_header
        + "\tselected_delta_r2\tselected_empirical_p\tselected_coeff"
    ]
    for rank, record in enumerate(chain_records, start=1):
        chain_lines.append(
            "\t".join(
                [
                    str(rank),
                    record["stage"],
                    record["source"],
                    record["target"],
                    record["forward_edge"],
                    record["reverse_edge"],
                    f"{record['forward_loo_delta_r2']:.6f}",
                    f"{record['reverse_loo_delta_r2']:.6f}",
                    f"{record['forward_minus_reverse']:.6f}",
                    f"{record['forward_coeff']:.6f}",
                    f"{record['reverse_coeff']:.6f}",
                    record["forward_relation"],
                    record["reverse_relation"],
                    f"{record['forward_empirical_p']:.6f}",
                    f"{record['reverse_empirical_p']:.6f}",
                    f"{record['forward_pearson_r']:.6f}",
                    f"{record['reverse_pearson_r']:.6f}",
                    record["direction_call"],
                    str(record["n_donors"]),
                    record["donors"],
                    record["source_first_stage"],
                    record["target_first_stage"],
                    record["temporal_relation"],
                    record["chain_source"],
                    record["chain_target"],
                    record["source_stage_appearance"],
                    record["target_stage_appearance"],
                    record["selected_edge"],
                    str(record["selected_edge_stage_count"]),
                    record["selected_edge_stages"],
                    record["stage_consistency"],
                    f"{record['selected_delta_r2']:.6f}",
                    f"{record['selected_empirical_p']:.6f}",
                    f"{record['selected_coeff']:.6f}",
                ]
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_text(output_dir / "nodes.tsv", "\n".join(node_lines) + "\n")
    save_text(output_dir / "bidirectional_edges.tsv", "\n".join(edge_lines) + "\n")
    save_text(output_dir / "stage_chain.tsv", "\n".join(chain_lines) + "\n")
    save_text(output_dir / "signature_reference.tsv", "\n".join(signature_lines) + "\n")
    save_text(output_dir / "network_summary.tsv", "\n".join(summary_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-wise bidirectional disease-chain network for SCP2154")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--barcodes", type=Path, required=True)
    parser.add_argument("--stages", default="low_steatosis,NAFLD,alcohol,cirrhotic,Tumor")
    parser.add_argument("--max-cells-per-donor-stage", type=int, default=80)
    parser.add_argument("--max-cells-per-stage", type=int, default=700)
    parser.add_argument("--min-cells-per-donor-node", type=int, default=5)
    parser.add_argument("--node-delta-threshold", type=float, default=0.25)
    parser.add_argument("--edge-min-delta-r2", type=float, default=0.2)
    parser.add_argument("--edge-max-p", type=float, default=0.15)
    parser.add_argument("--min-edge-donors", type=int, default=5)
    parser.add_argument("--direction-margin", type=float, default=0.05)
    parser.add_argument("--require-recurrent", action="store_true")
    parser.add_argument("--permutations", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("results/scp2154_stagewise_network"))
    args = parser.parse_args()

    stages = parse_stages(args.stages)
    metadata = read_metadata(args.metadata)
    signatures = all_signatures()
    rows_by_cell_type = {}
    sampled_rows = []
    for idx, cell_type in enumerate(signatures):
        rows = sample_celltype_stage_rows(
            metadata,
            cell_type,
            stages,
            args.max_cells_per_donor_stage,
            args.max_cells_per_stage,
            args.random_state + idx * 1000,
        )
        rows_by_cell_type[cell_type] = rows
        sampled_rows.extend(rows)
    sampled_rows = unique_rows(sampled_rows)
    print(f"Reading one shared matrix subset for {len(sampled_rows)} sampled cells...", flush=True)
    counts = read_10x_mtx_subset(args.matrix, args.features, args.barcodes, [row["NAME"] for row in sampled_rows])

    scores_by_node = {}
    kept_rows_by_cell_type = {}
    signature_lines = ["cell_type\tsignature\tmatched_genes"]
    for cell_type, rows in rows_by_cell_type.items():
        matrix, kept_rows = celltype_matrix(counts, rows)
        kept_rows_by_cell_type[cell_type] = kept_rows
        normalized = log1p_library_normalize(matrix)
        scores, used = signature_scores(normalized, counts.gene_names, signatures[cell_type])
        for signature, genes in used.items():
            node_id = f"{cell_type}.{signature}"
            scores_by_node[node_id] = scores[signature]
            signature_lines.append(f"{cell_type}\t{signature}\t{','.join(genes)}")

    node_records, stage_node_values = stage_node_tables(
        kept_rows_by_cell_type,
        scores_by_node,
        stages,
        args.min_cells_per_donor_node,
    )
    first_stage = first_altered_stage(node_records, args.node_delta_threshold)
    edge_records = build_bidirectional_edges(
        stage_node_values,
        first_stage,
        args.alpha,
        args.permutations,
        args.direction_margin,
        args.random_state + 999,
    )
    attach_stage_consistency(edge_records)
    chain_records = select_stage_chain(
        edge_records,
        args.edge_min_delta_r2,
        args.edge_max_p,
        min_edge_donors=args.min_edge_donors,
        require_recurrent=args.require_recurrent,
    )

    stage_counts = Counter(record["stage"] for record in chain_records)
    top_edges = chain_records[:8]
    summary_lines = [
        "metric\tvalue",
        f"sampled_cells\t{len(sampled_rows)}",
        f"nodes_scored\t{len(scores_by_node)}",
        f"stage_node_records\t{len(node_records)}",
        f"first_altered_nodes\t{len(first_stage)}",
        f"bidirectional_edges\t{len(edge_records)}",
        f"chain_edges\t{len(chain_records)}",
        f"chain_edges_by_stage\t{';'.join(f'{stage}:{count}' for stage, count in sorted(stage_counts.items()))}",
    ]
    for idx, edge in enumerate(top_edges, start=1):
        summary_lines.append(
            f"top_chain_{idx}\t{edge['stage']}:{edge['chain_source']}->{edge['chain_target']}:"
            f"delta_r2={edge['selected_delta_r2']:.4f}:p={edge['selected_empirical_p']:.4f}:"
            f"temporal={edge['temporal_relation']}"
        )

    write_outputs(args.output_dir, node_records, first_stage, edge_records, chain_records, signature_lines, summary_lines)
    print("SCP2154 stage-wise bidirectional network", flush=True)
    print(f"Nodes scored: {len(scores_by_node)}", flush=True)
    print(f"Bidirectional edges tested: {len(edge_records)}", flush=True)
    print(f"Chain edges retained: {len(chain_records)}", flush=True)
    for edge in top_edges[:5]:
        print(
            f"  {edge['stage']}: {edge['chain_source']} -> {edge['chain_target']} "
            f"delta_r2={edge['selected_delta_r2']:+.4f}, p={edge['selected_empirical_p']:.4f}, "
            f"{edge['temporal_relation']}",
            flush=True,
        )
    print("Saved:", flush=True)
    print(f"  {args.output_dir / 'nodes.tsv'}", flush=True)
    print(f"  {args.output_dir / 'bidirectional_edges.tsv'}", flush=True)
    print(f"  {args.output_dir / 'stage_chain.tsv'}", flush=True)
    print(f"  {args.output_dir / 'network_summary.tsv'}", flush=True)
    print(f"  {args.output_dir / 'signature_reference.tsv'}", flush=True)


if __name__ == "__main__":
    main()
