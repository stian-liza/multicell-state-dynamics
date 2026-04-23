from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from run_scp2154_coupling_driver_scan import unique_rows
from run_scp2154_fixed_module_progression import (
    find_progression_lines,
    hepatocyte_focus_records,
    microenvironment_progression_lines,
    sample_overview,
    signature_gene_union,
    suspicious_edge_records,
    write_table,
)
from run_scp2154_fixed_signature_tumor_validation import (
    empirical_p_value,
    loo_prediction_test,
    safe_corr,
    signature_scores,
)
from run_scp2154_phenotype_baseline import read_metadata
from run_scp2154_stagewise_bidirectional_network import (
    all_signatures,
    attach_stage_consistency,
    celltype_matrix,
    first_altered_stage,
    parse_stages,
    sample_celltype_stage_rows,
    select_stage_chain,
    write_outputs,
)
from multicell_dynamics import (
    local_direction_from_pseudotime,
    log1p_library_normalize,
    orient_pseudotime_by_labels,
    pca_embedding,
    pseudotime_from_embedding,
    read_10x_mtx_gene_cell_subset,
)


DEFAULT_STAGES = "low_steatosis,cirrhotic,Tumor"
DEFAULT_OUTPUT = Path("results/scp2154_stagewise_velocity_coupling")


def donor_metric_means(
    rows: list[dict[str, str]],
    values: np.ndarray,
    stage: dict,
    label: str,
    min_cells: int,
) -> tuple[dict[str, float], dict[str, int]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    column = stage["column"]
    wanted = "healthy" if label == "healthy" else stage["value"]
    for idx, row in enumerate(rows):
        if row.get(column) == wanted:
            grouped[row["donor_id"]].append(float(values[idx]))
    means = {}
    counts = {}
    for donor, donor_values in grouped.items():
        if len(donor_values) < min_cells:
            continue
        means[donor] = float(np.mean(donor_values))
        counts[donor] = len(donor_values)
    return means, counts


def score_and_velocity_tables(
    rows_by_cell_type: dict[str, list[dict[str, str]]],
    scores_by_node: dict[str, np.ndarray],
    velocity_by_node: dict[str, np.ndarray],
    stages: list[dict],
    min_cells: int,
) -> tuple[list[dict], dict[str, dict[str, dict[str, float]]], dict[str, dict[str, dict[str, float]]], list[str]]:
    node_records = []
    stage_score_values: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    stage_velocity_values: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    velocity_lines = [
        "stage\tstage_order\tnode_id\tcell_type\tsignature\tcondition_velocity_mean\thealthy_velocity_mean\t"
        "velocity_delta\tabs_velocity_delta\tcondition_velocity_donors\thealthy_velocity_donors\t"
        "condition_velocity_cells\thealthy_velocity_cells"
    ]
    for node_id, score_values in scores_by_node.items():
        cell_type, signature = node_id.split(".", 1)
        rows = rows_by_cell_type[cell_type]
        velocity_values = velocity_by_node[node_id]
        for stage in stages:
            cond_scores, cond_score_counts = donor_metric_means(rows, score_values, stage, "condition", min_cells)
            healthy_scores, healthy_score_counts = donor_metric_means(rows, score_values, stage, "healthy", min_cells)
            cond_vel, cond_vel_counts = donor_metric_means(rows, velocity_values, stage, "condition", min_cells)
            healthy_vel, healthy_vel_counts = donor_metric_means(rows, velocity_values, stage, "healthy", min_cells)
            if not cond_scores or not healthy_scores or not cond_vel:
                continue
            cond_score_mean = float(np.mean(list(cond_scores.values())))
            healthy_score_mean = float(np.mean(list(healthy_scores.values())))
            cond_vel_mean = float(np.mean(list(cond_vel.values())))
            healthy_vel_mean = float(np.mean(list(healthy_vel.values()))) if healthy_vel else 0.0
            score_delta = cond_score_mean - healthy_score_mean
            vel_delta = cond_vel_mean - healthy_vel_mean
            stage_score_values[stage["stage"]][node_id] = cond_scores
            stage_velocity_values[stage["stage"]][node_id] = cond_vel
            node_records.append(
                {
                    "stage": stage["stage"],
                    "stage_order": stage["order"],
                    "node_id": node_id,
                    "cell_type": cell_type,
                    "signature": signature,
                    "condition_mean": cond_score_mean,
                    "healthy_mean": healthy_score_mean,
                    "delta": score_delta,
                    "abs_delta": abs(score_delta),
                    "condition_donors": len(cond_scores),
                    "healthy_donors": len(healthy_scores),
                    "condition_cells": int(sum(cond_score_counts.values())),
                    "healthy_cells": int(sum(healthy_score_counts.values())),
                }
            )
            velocity_lines.append(
                "\t".join(
                    [
                        stage["stage"],
                        str(stage["order"]),
                        node_id,
                        cell_type,
                        signature,
                        f"{cond_vel_mean:.6f}",
                        f"{healthy_vel_mean:.6f}",
                        f"{vel_delta:.6f}",
                        f"{abs(vel_delta):.6f}",
                        str(len(cond_vel)),
                        str(len(healthy_vel)),
                        str(int(sum(cond_vel_counts.values()))),
                        str(int(sum(healthy_vel_counts.values()))),
                    ]
                )
            )
    return node_records, stage_score_values, stage_velocity_values, velocity_lines


def evaluate_score_to_velocity(
    source_values: dict[str, float],
    target_velocity: dict[str, float],
    alpha: float,
    permutations: int,
    rng: np.random.Generator,
) -> dict | None:
    donors = sorted(set(source_values) & set(target_velocity))
    if len(donors) < 5:
        return None
    x = np.array([[source_values[donor]] for donor in donors], dtype=float)
    y = np.array([target_velocity[donor] for donor in donors], dtype=float)
    observed = loo_prediction_test(x, y, alpha)
    null_values = []
    for _ in range(permutations):
        permuted = x[rng.permutation(len(x))]
        null_values.append(float(loo_prediction_test(permuted, y, alpha)["loo_delta_r2"]))
    null_array = np.array(null_values, dtype=float)
    return {
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
        return "positive_velocity_coupling"
    if coeff < 0:
        return "negative_velocity_coupling"
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


def build_velocity_edges(
    stage_score_values: dict[str, dict[str, dict[str, float]]],
    stage_velocity_values: dict[str, dict[str, dict[str, float]]],
    first_stage: dict[str, dict],
    alpha: float,
    permutations: int,
    margin: float,
    random_state: int,
) -> list[dict]:
    rng = np.random.default_rng(random_state)
    edge_records = []
    for stage, node_scores in stage_score_values.items():
        node_velocities = stage_velocity_values.get(stage, {})
        nodes = sorted(set(node_scores) & set(node_velocities))
        altered_nodes = [node for node in nodes if node in first_stage]
        for source_idx, source in enumerate(altered_nodes):
            for target in altered_nodes[source_idx + 1 :]:
                forward = evaluate_score_to_velocity(
                    node_scores[source],
                    node_velocities[target],
                    alpha,
                    permutations,
                    rng,
                )
                reverse = evaluate_score_to_velocity(
                    node_scores[target],
                    node_velocities[source],
                    alpha,
                    permutations,
                    rng,
                )
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
                        "forward_edge": f"{source}->{target}_velocity",
                        "reverse_edge": f"{target}->{source}_velocity",
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


def compute_celltype_scores_and_velocity(
    matrix: np.ndarray,
    rows: list[dict[str, str]],
    gene_names: np.ndarray,
    signatures: dict[str, list[str]],
    stages: list[dict],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, list[str]]]:
    normalized = log1p_library_normalize(matrix)
    scores, used = signature_scores(normalized, gene_names, signatures)
    score_matrix = np.column_stack([scores[name] for name in signatures]) if signatures else np.zeros((len(rows), 0))
    velocity_matrix = np.zeros_like(score_matrix, dtype=float)

    for stage in stages:
        labels = np.array(
            [
                "healthy" if row.get(stage["column"]) == "healthy" else stage["stage"]
                for row in rows
                if row.get(stage["column"]) in {"healthy", stage["value"]}
            ],
            dtype=object,
        )
        stage_idx = np.array(
            [idx for idx, row in enumerate(rows) if row.get(stage["column"]) in {"healthy", stage["value"]}],
            dtype=int,
        )
        if len(stage_idx) < 10:
            continue
        stage_scores = score_matrix[stage_idx]
        embedding = pca_embedding(stage_scores, n_components=min(2, max(1, stage_scores.shape[1])))
        pseudotime = pseudotime_from_embedding(embedding, axis=0)
        pseudotime = orient_pseudotime_by_labels(pseudotime, labels, "healthy", stage["stage"])
        stage_velocity = local_direction_from_pseudotime(stage_scores, pseudotime, k_neighbors=min(15, max(3, len(stage_idx) // 8)))
        velocity_matrix[stage_idx] = stage_velocity

    velocity_by_signature = {
        signature_name: velocity_matrix[:, idx]
        for idx, signature_name in enumerate(signatures)
    }
    return scores, velocity_by_signature, used


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-wise velocity-coupling network for SCP2154")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--barcodes", type=Path, required=True)
    parser.add_argument("--stages", default=DEFAULT_STAGES)
    parser.add_argument("--max-cells-per-donor-stage", type=int, default=120)
    parser.add_argument("--max-cells-per-stage", type=int, default=1200)
    parser.add_argument("--min-cells-per-donor-node", type=int, default=8)
    parser.add_argument("--node-delta-threshold", type=float, default=0.25)
    parser.add_argument("--edge-min-delta-r2", type=float, default=0.15)
    parser.add_argument("--edge-max-p", type=float, default=0.15)
    parser.add_argument("--min-edge-donors", type=int, default=5)
    parser.add_argument("--direction-margin", type=float, default=0.05)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=29)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
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

    genes_to_read = signature_gene_union(signatures)
    print(
        f"Reading velocity-coupling subset for {len(sampled_rows)} sampled cells and {len(genes_to_read)} signature genes...",
        flush=True,
    )
    counts = read_10x_mtx_gene_cell_subset(
        args.matrix,
        args.features,
        args.barcodes,
        [row["NAME"] for row in sampled_rows],
        genes_to_read,
    )

    scores_by_node = {}
    velocity_by_node = {}
    kept_rows_by_cell_type = {}
    signature_lines = ["cell_type\tsignature\tmatched_genes"]
    for cell_type, rows in rows_by_cell_type.items():
        matrix, kept_rows = celltype_matrix(counts, rows)
        kept_rows_by_cell_type[cell_type] = kept_rows
        scores, velocity_scores, used = compute_celltype_scores_and_velocity(
            matrix,
            kept_rows,
            counts.gene_names,
            signatures[cell_type],
            stages,
        )
        for signature, genes in used.items():
            node_id = f"{cell_type}.{signature}"
            scores_by_node[node_id] = scores[signature]
            velocity_by_node[node_id] = velocity_scores[signature]
            signature_lines.append(f"{cell_type}\t{signature}\t{','.join(genes)}")

    node_records, stage_score_values, stage_velocity_values, velocity_lines = score_and_velocity_tables(
        kept_rows_by_cell_type,
        scores_by_node,
        velocity_by_node,
        stages,
        args.min_cells_per_donor_node,
    )
    first_stage = first_altered_stage(node_records, args.node_delta_threshold)
    edge_records = build_velocity_edges(
        stage_score_values,
        stage_velocity_values,
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
        require_recurrent=False,
    )

    focus_records = hepatocyte_focus_records(edge_records)
    progression_lines = find_progression_lines(chain_records, stages)
    micro_lines = microenvironment_progression_lines(progression_lines)
    suspicious_records = suspicious_edge_records(edge_records, micro_lines or progression_lines)
    stage_counts = Counter(record["stage"] for record in chain_records)
    sample_lines = sample_overview(rows_by_cell_type, stages)

    summary_lines = [
        "metric\tvalue",
        f"sampled_cells\t{len(sampled_rows)}",
        f"signature_gene_count\t{len(genes_to_read)}",
        f"nodes_scored\t{len(scores_by_node)}",
        f"stage_node_records\t{len(node_records)}",
        f"first_altered_nodes\t{len(first_stage)}",
        f"bidirectional_edges\t{len(edge_records)}",
        f"chain_edges\t{len(chain_records)}",
        f"hepatocyte_focus_edges\t{len(focus_records)}",
        f"progression_lines\t{len(progression_lines)}",
        f"microenvironment_progression_lines\t{len(micro_lines)}",
        f"suspicious_edges\t{len(suspicious_records)}",
        f"chain_edges_by_stage\t{';'.join(f'{stage}:{count}' for stage, count in sorted(stage_counts.items()))}",
    ]
    for idx, record in enumerate((micro_lines or progression_lines)[:5], start=1):
        summary_lines.append(
            f"top_progression_line_{idx}\t{' | '.join(record['path_edges'])}:"
            f"score={record['path_score']:.4f}:delta_r2={record['total_delta_r2']:.4f}"
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(output_dir, node_records, first_stage, edge_records, chain_records, signature_lines, summary_lines)
    save_text = output_dir.joinpath  # local alias for cleaner writes below
    write_table(output_dir / "sample_overview.tsv", ["stage", "condition_label", "cell_type", "cells", "donors"], [line.split("\t") for line in sample_lines[1:]])
    output_dir.joinpath("velocity_nodes.tsv").write_text("\n".join(velocity_lines) + "\n")

    focus_header = [
        "rank",
        "stage",
        "source",
        "target",
        "selected_source",
        "selected_target",
        "forward_loo_delta_r2",
        "reverse_loo_delta_r2",
        "selected_delta_r2",
        "selected_empirical_p",
        "selected_coeff",
        "direction_call",
        "temporal_relation",
        "stage_consistency",
        "n_donors",
        "donors",
    ]
    focus_rows = []
    for rank, record in enumerate(focus_records, start=1):
        focus_rows.append(
            [
                str(rank),
                record["stage"],
                record["source"],
                record["target"],
                record["selected_source"],
                record["selected_target"],
                f"{record['forward_loo_delta_r2']:.6f}",
                f"{record['reverse_loo_delta_r2']:.6f}",
                f"{record['selected_delta_r2']:.6f}",
                f"{record['selected_empirical_p']:.6f}",
                f"{record['selected_coeff']:.6f}",
                record["direction_call"],
                record["temporal_relation"],
                record["stage_consistency"],
                str(record["n_donors"]),
                record["donors"],
            ]
        )
    write_table(output_dir / "hepatocyte_focus_edges.tsv", focus_header, focus_rows)

    suspicious_header = focus_header + ["selected_edge"]
    suspicious_rows = []
    for rank, record in enumerate(suspicious_records, start=1):
        suspicious_rows.append(
            [
                str(rank),
                record["stage"],
                record["source"],
                record["target"],
                record["selected_source"],
                record["selected_target"],
                f"{record['forward_loo_delta_r2']:.6f}",
                f"{record['reverse_loo_delta_r2']:.6f}",
                f"{record['selected_delta_r2']:.6f}",
                f"{record['selected_empirical_p']:.6f}",
                f"{record['selected_coeff']:.6f}",
                record["direction_call"],
                record["temporal_relation"],
                record["stage_consistency"],
                str(record["n_donors"]),
                record["donors"],
                f"{record['selected_source']}->{record['selected_target']}",
            ]
        )
    write_table(output_dir / "suspicious_bidirectional_edges.tsv", suspicious_header, suspicious_rows)

    line_header = [
        "rank",
        "n_hops",
        "path_score",
        "total_delta_r2",
        "recurrent_hops",
        "source_earlier_hops",
        "contains_secretory_stress",
        "path_nodes",
        "path_stages",
        "path_edges",
    ]
    line_rows = []
    for rank, record in enumerate(progression_lines, start=1):
        line_rows.append(
            [
                str(rank),
                str(record["n_hops"]),
                f"{record['path_score']:.6f}",
                f"{record['total_delta_r2']:.6f}",
                str(record["recurrent_hops"]),
                str(record["source_earlier_hops"]),
                str(record["contains_secretory_stress"]),
                " | ".join(record["path_nodes"]),
                " | ".join(record["path_stages"]),
                " | ".join(record["path_edges"]),
            ]
        )
    write_table(output_dir / "candidate_progression_lines.tsv", line_header, line_rows)

    micro_rows = []
    for rank, record in enumerate(micro_lines, start=1):
        micro_rows.append(
            [
                str(rank),
                str(record["n_hops"]),
                f"{record['path_score']:.6f}",
                f"{record['total_delta_r2']:.6f}",
                str(record["recurrent_hops"]),
                str(record["source_earlier_hops"]),
                str(record["contains_secretory_stress"]),
                " | ".join(record["path_nodes"]),
                " | ".join(record["path_stages"]),
                " | ".join(record["path_edges"]),
            ]
        )
    write_table(output_dir / "candidate_microenvironment_lines.tsv", line_header, micro_rows)

    print("SCP2154 stage-wise velocity coupling", flush=True)
    print(f"Sampled cells: {len(sampled_rows)}", flush=True)
    print(f"Signature genes read: {len(genes_to_read)}", flush=True)
    print(f"Bidirectional edges tested: {len(edge_records)}", flush=True)
    print(f"Chain edges retained: {len(chain_records)}", flush=True)
    print(f"Hepatocyte-focus edges: {len(focus_records)}", flush=True)
    if micro_lines or progression_lines:
        best = (micro_lines or progression_lines)[0]
        print(f"Top progression line: {' | '.join(best['path_edges'])}", flush=True)
        print(f"  score={best['path_score']:.4f}, total_delta_r2={best['total_delta_r2']:.4f}", flush=True)
    print("Saved:", flush=True)
    for name in [
        "sample_overview.tsv",
        "nodes.tsv",
        "velocity_nodes.tsv",
        "bidirectional_edges.tsv",
        "stage_chain.tsv",
        "hepatocyte_focus_edges.tsv",
        "suspicious_bidirectional_edges.tsv",
        "candidate_progression_lines.tsv",
        "candidate_microenvironment_lines.tsv",
        "network_summary.tsv",
    ]:
        print(f"  {output_dir / name}", flush=True)


if __name__ == "__main__":
    main()
