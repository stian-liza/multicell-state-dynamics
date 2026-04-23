from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from run_scp2154_coupling_driver_scan import unique_rows
from run_scp2154_fixed_signature_tumor_validation import signature_scores
from run_scp2154_phenotype_baseline import read_metadata
from run_scp2154_stagewise_bidirectional_network import (
    all_signatures,
    attach_stage_consistency,
    build_bidirectional_edges,
    celltype_matrix,
    first_altered_stage,
    parse_stages,
    sample_celltype_stage_rows,
    select_stage_chain,
    stage_node_tables,
)
from run_scp2154_stromal_to_hepatocyte_coupling import save_text
from multicell_dynamics import log1p_library_normalize, read_10x_mtx_gene_cell_subset


DEFAULT_STAGES = "low_steatosis,cirrhotic,Tumor"
DEFAULT_OUTPUT = Path("results/scp2154_fixed_module_progression")
FOCUS_TARGETS = {"Hepatocyte.secretory_stress", "Hepatocyte.malignant_like"}


def signature_gene_union(signatures: dict[str, dict[str, list[str]]]) -> list[str]:
    seen = set()
    genes = []
    for by_signature in signatures.values():
        for signature_genes in by_signature.values():
            for gene in signature_genes:
                gene_upper = gene.upper()
                if gene_upper in seen:
                    continue
                seen.add(gene_upper)
                genes.append(gene)
    return genes


def sample_overview(rows_by_cell_type: dict[str, list[dict[str, str]]], stages: list[dict]) -> list[str]:
    stage_by_column_value = {(stage["column"], stage["value"]): stage["stage"] for stage in stages}
    lines = ["stage\tcondition_label\tcell_type\tcells\tdonors"]
    for cell_type, rows in sorted(rows_by_cell_type.items()):
        grouped_counts: dict[tuple[str, str], int] = Counter()
        grouped_donors: dict[tuple[str, str], set[str]] = defaultdict(set)
        for row in rows:
            for stage in stages:
                stage_value = row.get(stage["column"])
                if stage_value not in {"healthy", stage["value"]}:
                    continue
                label = "healthy" if stage_value == "healthy" else stage["stage"]
                key = (stage["stage"], label)
                grouped_counts[key] += 1
                grouped_donors[key].add(row["donor_id"])
        for stage in stages:
            for label in ("healthy", stage["stage"]):
                key = (stage["stage"], label)
                lines.append(
                    f"{stage['stage']}\t{label}\t{cell_type}\t{grouped_counts[key]}\t{len(grouped_donors[key])}"
                )
    return lines


def selected_direction(edge: dict) -> tuple[str, str, float, float, float] | None:
    if edge["direction_call"] == "forward_stronger":
        return (
            edge["source"],
            edge["target"],
            edge["forward_loo_delta_r2"],
            edge["forward_empirical_p"],
            edge["forward_coeff"],
        )
    if edge["direction_call"] == "reverse_stronger":
        return (
            edge["target"],
            edge["source"],
            edge["reverse_loo_delta_r2"],
            edge["reverse_empirical_p"],
            edge["reverse_coeff"],
        )
    return None


def hepatocyte_focus_records(edge_records: list[dict]) -> list[dict]:
    focus = []
    for edge in edge_records:
        directed = selected_direction(edge)
        if directed is None:
            continue
        selected_source, selected_target, selected_delta, selected_p, selected_coeff = directed
        nodes = {edge["source"], edge["target"], selected_source, selected_target}
        if not any(node in FOCUS_TARGETS or node.startswith("Hepatocyte.") for node in nodes):
            continue
        item = dict(edge)
        item["selected_source"] = selected_source
        item["selected_target"] = selected_target
        item["selected_delta_r2"] = selected_delta
        item["selected_empirical_p"] = selected_p
        item["selected_coeff"] = selected_coeff
        focus.append(item)
    focus.sort(
        key=lambda item: (
            item["selected_target"] in FOCUS_TARGETS,
            item["selected_delta_r2"],
            abs(item["selected_coeff"]),
        ),
        reverse=True,
    )
    return focus


def _path_score(path_edges: list[dict]) -> float:
    base = sum(edge["selected_delta_r2"] for edge in path_edges)
    recurrent_bonus = 0.05 * sum(edge.get("stage_consistency") == "recurrent" for edge in path_edges)
    temporal_bonus = 0.03 * sum(edge.get("temporal_relation") == "source_earlier" for edge in path_edges)
    return float(base + recurrent_bonus + temporal_bonus)


def find_progression_lines(
    chain_records: list[dict],
    stages: list[dict],
    terminal_node: str = "Hepatocyte.malignant_like",
    max_hops: int = 4,
) -> list[dict]:
    stage_order = {stage["stage"]: stage["order"] for stage in stages}
    directed_edges = []
    for edge in chain_records:
        directed = selected_direction(edge)
        if directed is None:
            continue
        selected_source, selected_target, selected_delta, selected_p, selected_coeff = directed
        item = dict(edge)
        item["selected_source"] = selected_source
        item["selected_target"] = selected_target
        item["selected_delta_r2"] = selected_delta
        item["selected_empirical_p"] = selected_p
        item["selected_coeff"] = selected_coeff
        item["stage_order_numeric"] = stage_order[edge["stage"]]
        directed_edges.append(item)

    by_source: dict[str, list[dict]] = defaultdict(list)
    for edge in directed_edges:
        by_source[edge["selected_source"]].append(edge)
    for source in by_source:
        by_source[source].sort(key=lambda item: (item["stage_order_numeric"], -item["selected_delta_r2"]))

    candidates: list[dict] = []

    def dfs(current_node: str, current_stage_order: int, visited: set[str], path_edges: list[dict]) -> None:
        if len(path_edges) >= max_hops:
            return
        for edge in by_source.get(current_node, []):
            if edge["stage_order_numeric"] < current_stage_order:
                continue
            next_node = edge["selected_target"]
            if next_node in visited:
                continue
            new_path = path_edges + [edge]
            new_visited = set(visited)
            new_visited.add(next_node)
            if next_node == terminal_node:
                node_sequence = [new_path[0]["selected_source"]] + [item["selected_target"] for item in new_path]
                cell_types = {node.split(".", 1)[0] for node in node_sequence}
                if len(cell_types) >= 2:
                    candidates.append(
                        {
                            "path_nodes": node_sequence,
                            "path_stages": [item["stage"] for item in new_path],
                            "path_edges": [f"{item['selected_source']}->{item['selected_target']}" for item in new_path],
                            "n_hops": len(new_path),
                            "total_delta_r2": float(sum(item["selected_delta_r2"] for item in new_path)),
                            "path_score": _path_score(new_path),
                            "recurrent_hops": int(sum(item.get("stage_consistency") == "recurrent" for item in new_path)),
                            "source_earlier_hops": int(sum(item.get("temporal_relation") == "source_earlier" for item in new_path)),
                            "contains_secretory_stress": int("Hepatocyte.secretory_stress" in node_sequence),
                            "starts_in_hepatocyte": int(node_sequence[0].startswith("Hepatocyte.")),
                        }
                    )
            dfs(next_node, edge["stage_order_numeric"], new_visited, new_path)

    for source in sorted(by_source):
        if source == terminal_node:
            continue
        dfs(source, 0, {source}, [])

    unique_candidates = {}
    for item in candidates:
        key = tuple(item["path_edges"])
        previous = unique_candidates.get(key)
        if previous is None or item["path_score"] > previous["path_score"]:
            unique_candidates[key] = item
    out = list(unique_candidates.values())
    out.sort(
        key=lambda item: (
            1 - item["starts_in_hepatocyte"],
            item["contains_secretory_stress"],
            item["path_score"],
            item["total_delta_r2"],
            -item["n_hops"],
        ),
        reverse=True,
    )
    return out


def microenvironment_progression_lines(progression_lines: list[dict]) -> list[dict]:
    return [item for item in progression_lines if not item["starts_in_hepatocyte"]]


def suspicious_edge_records(edge_records: list[dict], progression_lines: list[dict], keep_top_paths: int = 5) -> list[dict]:
    suspicious_pairs = set()
    for path in progression_lines[:keep_top_paths]:
        suspicious_pairs.update(path["path_edges"])
    suspicious = []
    for edge in edge_records:
        directed = selected_direction(edge)
        if directed is None:
            continue
        selected_source, selected_target, selected_delta, selected_p, selected_coeff = directed
        edge_name = f"{selected_source}->{selected_target}"
        if edge_name not in suspicious_pairs:
            continue
        item = dict(edge)
        item["selected_source"] = selected_source
        item["selected_target"] = selected_target
        item["selected_delta_r2"] = selected_delta
        item["selected_empirical_p"] = selected_p
        item["selected_coeff"] = selected_coeff
        suspicious.append(item)
    suspicious.sort(
        key=lambda item: (
            item["selected_target"] == "Hepatocyte.malignant_like",
            item["selected_target"] == "Hepatocyte.secretory_stress",
            item["selected_delta_r2"],
        ),
        reverse=True,
    )
    return suspicious


def write_table(path: Path, header: list[str], rows: list[list[str]]) -> None:
    lines = ["\t".join(header)]
    lines.extend("\t".join(row) for row in rows)
    save_text(path, "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed cell-type/module progression scan for SCP2154")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--barcodes", type=Path, required=True)
    parser.add_argument("--stages", default=DEFAULT_STAGES)
    parser.add_argument("--max-cells-per-donor-stage", type=int, default=160)
    parser.add_argument("--max-cells-per-stage", type=int, default=1800)
    parser.add_argument("--min-cells-per-donor-node", type=int, default=8)
    parser.add_argument("--node-delta-threshold", type=float, default=0.25)
    parser.add_argument("--edge-min-delta-r2", type=float, default=0.2)
    parser.add_argument("--edge-max-p", type=float, default=0.15)
    parser.add_argument("--min-edge-donors", type=int, default=5)
    parser.add_argument("--direction-margin", type=float, default=0.05)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=17)
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
        f"Reading fixed-signature subset for {len(sampled_rows)} sampled cells and {len(genes_to_read)} signature genes...",
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
        require_recurrent=False,
    )

    focus_records = hepatocyte_focus_records(edge_records)
    progression_lines = find_progression_lines(chain_records, stages)
    micro_lines = microenvironment_progression_lines(progression_lines)
    suspicious_records = suspicious_edge_records(edge_records, micro_lines or progression_lines)
    stage_counts = Counter(record["stage"] for record in chain_records)
    sample_lines = sample_overview(rows_by_cell_type, stages)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the standard stage-wise outputs for compatibility.
    from run_scp2154_stagewise_bidirectional_network import write_outputs

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

    write_outputs(output_dir, node_records, first_stage, edge_records, chain_records, signature_lines, summary_lines)
    save_text(output_dir / "sample_overview.tsv", "\n".join(sample_lines) + "\n")

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

    print("SCP2154 fixed-module progression scan", flush=True)
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
