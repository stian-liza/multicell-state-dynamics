from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from run_scp2154_phenotype_baseline import read_metadata
from run_scp2154_pretumor_coupling import (
    build_sender_input,
    condition_relation,
    coupling_direction,
    fit_condition_representation,
    make_condition_rows,
    module_shift,
    parse_comparisons,
    sender_scores_by_donor_condition,
    split_indices_by_donor,
)
from run_scp2154_sender_comparison import fit_baseline, module_delta
from run_scp2154_stromal_to_hepatocyte_coupling import (
    annotate_program,
    per_module_r2,
    save_text,
    top_gene_names,
    zscore_by_train,
)
from multicell_dynamics import fit_population_dynamics, read_10x_mtx_subset


DEFAULT_PARTNERS = ["Stromal", "Myeloid", "Endothelial", "TNKcell", "Bcell"]
DEFAULT_COMPARISONS = "health:low_steatosis,health:cirrhotic,indication:alcohol,indication:NAFLD,indication:Tumor"
RECEIVER = "Hepatocyte"


def label_counts(labels: np.ndarray) -> str:
    counts = Counter(str(label) for label in labels)
    return ";".join(f"{key}:{value}" for key, value in sorted(counts.items()))


def direction_call(forward_delta: float, reverse_delta: float, margin: float) -> str:
    if forward_delta > margin and forward_delta > reverse_delta + margin:
        return "partner_to_hepatocyte_stronger"
    if reverse_delta > margin and reverse_delta > forward_delta + margin:
        return "hepatocyte_to_partner_stronger"
    if forward_delta > margin and reverse_delta > margin:
        return "bidirectional_or_shared_stage"
    return "weak_or_inconclusive"


def fit_directed_edges(
    condition_value: str,
    sender_cell_type: str,
    receiver_cell_type: str,
    sender: dict,
    receiver: dict,
    receiver_rows: list[dict[str, str]],
    sender_shift: list[tuple[int, float, float, float]],
    receiver_shift: list[tuple[int, float, float, float]],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    alpha: float,
) -> list[dict]:
    baseline = fit_baseline(receiver, train_idx, test_idx, alpha)
    sender_scores = sender_scores_by_donor_condition(sender["rows"], sender["rep"].module_activity)
    raw_sender_input = build_sender_input(receiver_rows, sender_scores)
    sender_input = zscore_by_train(raw_sender_input, train_idx)
    records = []

    for sender_module_idx in range(sender_input.shape[1]):
        single_input = sender_input[:, [sender_module_idx]]
        model = fit_population_dynamics(
            module_activity=receiver["rep"].module_activity[train_idx],
            state_embedding=receiver["embedding"][train_idx],
            module_velocity=receiver["local_velocity"][train_idx],
            external_input=single_input[train_idx],
            alpha=alpha,
        )
        prediction = model.predict_velocity(
            receiver["rep"].module_activity[test_idx],
            receiver["embedding"][test_idx],
            external_input=single_input[test_idx],
        )
        coupled_r2 = per_module_r2(receiver["local_velocity"][test_idx], prediction)
        coeff_idx = model.feature_names.index("e_0")
        sender_delta = module_delta(sender_shift, sender_module_idx)
        sender_genes = top_gene_names(sender["top_genes"][sender_module_idx])
        for receiver_module_idx in range(receiver["rep"].module_activity.shape[1]):
            receiver_delta = module_delta(receiver_shift, receiver_module_idx)
            receiver_genes = top_gene_names(receiver["top_genes"][receiver_module_idx])
            delta_r2 = float(coupled_r2[receiver_module_idx] - baseline["per_module_r2"][receiver_module_idx])
            coeff = float(model.coefficient_matrix[receiver_module_idx, coeff_idx])
            records.append(
                {
                    "condition": condition_value,
                    "sender_cell_type": sender_cell_type,
                    "receiver_cell_type": receiver_cell_type,
                    "sender_module": sender_module_idx,
                    "sender_delta": sender_delta,
                    "sender_relation": condition_relation(sender_delta),
                    "sender_program": annotate_program(sender_cell_type, sender_genes),
                    "sender_top_genes": ",".join(sender_genes),
                    "receiver_module": receiver_module_idx,
                    "receiver_delta": receiver_delta,
                    "receiver_relation": condition_relation(receiver_delta),
                    "receiver_program": annotate_program(receiver_cell_type, receiver_genes),
                    "receiver_top_genes": ",".join(receiver_genes),
                    "baseline_r2": float(baseline["per_module_r2"][receiver_module_idx]),
                    "coupled_r2": float(coupled_r2[receiver_module_idx]),
                    "delta_r2": delta_r2,
                    "coeff": coeff,
                    "direction": coupling_direction(coeff, receiver_delta, delta_r2),
                }
            )
    return records


def pair_bidirectional_records(
    condition_col: str,
    condition_value: str,
    partner_cell_type: str,
    forward_edges: list[dict],
    reverse_edges: list[dict],
    margin: float,
) -> list[dict]:
    reverse_by_pair = {
        (item["receiver_module"], item["sender_module"]): item
        for item in reverse_edges
    }
    records = []
    for forward in forward_edges:
        pair_key = (forward["sender_module"], forward["receiver_module"])
        reverse = reverse_by_pair[pair_key]
        forward_delta = forward["delta_r2"]
        reverse_delta = reverse["delta_r2"]
        records.append(
            {
                "condition_col": condition_col,
                "condition": condition_value,
                "partner_cell_type": partner_cell_type,
                "partner_module": forward["sender_module"],
                "hep_module": forward["receiver_module"],
                "forward_edge": f"{partner_cell_type}_m{forward['sender_module']}->Hepatocyte_m{forward['receiver_module']}",
                "reverse_edge": f"Hepatocyte_m{reverse['sender_module']}->{partner_cell_type}_m{reverse['receiver_module']}",
                "forward_delta_r2": forward_delta,
                "reverse_delta_r2": reverse_delta,
                "forward_minus_reverse": forward_delta - reverse_delta,
                "direction_call": direction_call(forward_delta, reverse_delta, margin),
                "forward_coeff": forward["coeff"],
                "reverse_coeff": reverse["coeff"],
                "partner_condition_delta": forward["sender_delta"],
                "partner_relation": forward["sender_relation"],
                "partner_program": forward["sender_program"],
                "partner_top_genes": forward["sender_top_genes"],
                "hep_condition_delta": forward["receiver_delta"],
                "hep_relation": forward["receiver_relation"],
                "hep_program": forward["receiver_program"],
                "hep_top_genes": forward["receiver_top_genes"],
                "forward_direction": forward["direction"],
                "reverse_direction": reverse["direction"],
            }
        )
    return records


def module_reference_lines(
    condition_col: str,
    condition_value: str,
    cell_type: str,
    rep: dict,
    shifts: list[tuple[int, float, float, float]],
) -> list[str]:
    lines = []
    for module_idx, condition_mean, healthy_mean, delta in shifts:
        genes = top_gene_names(rep["top_genes"][module_idx])
        lines.append(
            "\t".join(
                [
                    condition_col,
                    condition_value,
                    cell_type,
                    f"m_{module_idx}",
                    f"{condition_mean:.6f}",
                    f"{healthy_mean:.6f}",
                    f"{delta:.6f}",
                    condition_relation(delta),
                    annotate_program(cell_type, genes),
                    ",".join(genes),
                ]
            )
        )
    return lines


def write_outputs(output_dir: Path, summaries: list[dict], pairs: list[dict], directed_edges: list[dict], modules: list[str]) -> None:
    pairs.sort(
        key=lambda item: (
            item["direction_call"] == "partner_to_hepatocyte_stronger",
            item["forward_delta_r2"],
            item["forward_minus_reverse"],
        ),
        reverse=True,
    )
    directed_edges.sort(key=lambda item: item["delta_r2"], reverse=True)
    summaries.sort(key=lambda item: item["best_forward_delta_r2"], reverse=True)

    pair_lines = [
        "rank\tcondition_col\tcondition\tpartner_cell_type\tpartner_module\thep_module\tforward_edge\treverse_edge\t"
        "forward_delta_r2\treverse_delta_r2\tforward_minus_reverse\tdirection_call\tforward_coeff\treverse_coeff\t"
        "partner_condition_delta\tpartner_relation\tpartner_program\thep_condition_delta\thep_relation\thep_program\t"
        "forward_direction\treverse_direction\tpartner_top_genes\thep_top_genes"
    ]
    for rank, item in enumerate(pairs, start=1):
        pair_lines.append(
            "\t".join(
                [
                    str(rank),
                    item["condition_col"],
                    item["condition"],
                    item["partner_cell_type"],
                    f"m_{item['partner_module']}",
                    f"m_{item['hep_module']}",
                    item["forward_edge"],
                    item["reverse_edge"],
                    f"{item['forward_delta_r2']:.6f}",
                    f"{item['reverse_delta_r2']:.6f}",
                    f"{item['forward_minus_reverse']:.6f}",
                    item["direction_call"],
                    f"{item['forward_coeff']:.6f}",
                    f"{item['reverse_coeff']:.6f}",
                    f"{item['partner_condition_delta']:.6f}",
                    item["partner_relation"],
                    item["partner_program"],
                    f"{item['hep_condition_delta']:.6f}",
                    item["hep_relation"],
                    item["hep_program"],
                    item["forward_direction"],
                    item["reverse_direction"],
                    item["partner_top_genes"],
                    item["hep_top_genes"],
                ]
            )
        )

    edge_lines = [
        "rank\tcondition\tsender_cell_type\tsender_module\treceiver_cell_type\treceiver_module\t"
        "delta_r2\tcoeff\tbaseline_r2\tcoupled_r2\tdirection\tsender_relation\treceiver_relation\t"
        "sender_program\treceiver_program\tsender_top_genes\treceiver_top_genes"
    ]
    for rank, item in enumerate(directed_edges, start=1):
        edge_lines.append(
            "\t".join(
                [
                    str(rank),
                    item["condition"],
                    item["sender_cell_type"],
                    f"m_{item['sender_module']}",
                    item["receiver_cell_type"],
                    f"m_{item['receiver_module']}",
                    f"{item['delta_r2']:.6f}",
                    f"{item['coeff']:.6f}",
                    f"{item['baseline_r2']:.6f}",
                    f"{item['coupled_r2']:.6f}",
                    item["direction"],
                    item["sender_relation"],
                    item["receiver_relation"],
                    item["sender_program"],
                    item["receiver_program"],
                    item["sender_top_genes"],
                    item["receiver_top_genes"],
                ]
            )
        )

    summary_lines = [
        "condition_col\tcondition\tpartner_cell_type\thep_cells\thep_label_counts\tpartner_cells\tpartner_label_counts\t"
        "best_forward_edge\tbest_reverse_edge\tbest_forward_delta_r2\tbest_reverse_delta_r2\t"
        "best_forward_minus_reverse\tbest_direction_call\tbest_partner_program\tbest_hep_program"
    ]
    for item in summaries:
        best = item["best_pair"]
        summary_lines.append(
            "\t".join(
                [
                    item["condition_col"],
                    item["condition"],
                    item["partner_cell_type"],
                    str(item["hep_cells"]),
                    item["hep_label_counts"],
                    str(item["partner_cells"]),
                    item["partner_label_counts"],
                    best["forward_edge"],
                    best["reverse_edge"],
                    f"{best['forward_delta_r2']:.6f}",
                    f"{best['reverse_delta_r2']:.6f}",
                    f"{best['forward_minus_reverse']:.6f}",
                    best["direction_call"],
                    best["partner_program"],
                    best["hep_program"],
                ]
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_text(output_dir / "bidirectional_pairs.tsv", "\n".join(pair_lines) + "\n")
    save_text(output_dir / "directed_edges.tsv", "\n".join(edge_lines) + "\n")
    save_text(output_dir / "summary.tsv", "\n".join(summary_lines) + "\n")
    save_text(output_dir / "module_reference.tsv", "\n".join(modules) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bidirectional SCP2154 module-coupling tests across key liver disease states")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--barcodes", type=Path, required=True)
    parser.add_argument("--comparisons", default=DEFAULT_COMPARISONS)
    parser.add_argument("--partner-cell-types", default=",".join(DEFAULT_PARTNERS))
    parser.add_argument("--max-cells-per-donor-condition", type=int, default=70)
    parser.add_argument("--max-cells-per-condition", type=int, default=700)
    parser.add_argument("--hvg", type=int, default=700)
    parser.add_argument("--modules", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.03)
    parser.add_argument("--margin", type=float, default=0.01)
    parser.add_argument("--min-cells", type=int, default=120)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("results/scp2154_bidirectional_key_tests"))
    args = parser.parse_args()

    metadata = read_metadata(args.metadata)
    comparisons = parse_comparisons(args.comparisons)
    partner_cell_types = [item.strip() for item in args.partner_cell_types.split(",") if item.strip()]
    all_pairs: list[dict] = []
    all_directed_edges: list[dict] = []
    summaries: list[dict] = []
    module_lines = [
        "condition_col\tcondition\tcell_type\tmodule\tcondition_mean\thealthy_mean\t"
        "delta_condition_minus_healthy\tcondition_relation\tputative_program\ttop_genes"
    ]

    print("SCP2154 bidirectional key tests", flush=True)
    for comparison_idx, (condition_col, condition_value) in enumerate(comparisons):
        hep_rows = make_condition_rows(
            metadata,
            RECEIVER,
            condition_col,
            condition_value,
            args.max_cells_per_donor_condition,
            args.max_cells_per_condition,
            args.random_state + comparison_idx * 1000 + 101,
        )
        if len(hep_rows) < args.min_cells:
            print(f"  {condition_value}: skipped Hepatocyte, only {len(hep_rows)} cells", flush=True)
            continue

        partner_rows_by_type = {}
        for partner_idx, partner_cell_type in enumerate(partner_cell_types):
            rows = make_condition_rows(
                metadata,
                partner_cell_type,
                condition_col,
                condition_value,
                args.max_cells_per_donor_condition,
                args.max_cells_per_condition,
                args.random_state + comparison_idx * 1000 + partner_idx,
            )
            if len(rows) >= args.min_cells:
                partner_rows_by_type[partner_cell_type] = rows
            else:
                print(f"  {condition_value}/{partner_cell_type}: skipped, only {len(rows)} cells", flush=True)

        sampled_rows = hep_rows + [row for rows in partner_rows_by_type.values() for row in rows]
        print(f"  {condition_value}: reading {len(sampled_rows)} sampled cells...", flush=True)
        counts = read_10x_mtx_subset(args.matrix, args.features, args.barcodes, [row["NAME"] for row in sampled_rows])
        hep = fit_condition_representation(
            counts,
            hep_rows,
            condition_value,
            args.modules,
            args.hvg,
            args.random_state + comparison_idx * 1000 + 201,
        )
        hep_shift = module_shift(hep["rep"].module_activity, hep["labels"], condition_value)
        hep_train_idx, hep_test_idx = split_indices_by_donor(hep_rows, random_state=args.random_state)
        module_lines.extend(module_reference_lines(condition_col, condition_value, RECEIVER, hep, hep_shift))

        for partner_idx, (partner_cell_type, partner_rows) in enumerate(partner_rows_by_type.items()):
            print(f"    testing {partner_cell_type} <-> {RECEIVER}...", flush=True)
            partner = fit_condition_representation(
                counts,
                partner_rows,
                condition_value,
                args.modules,
                args.hvg,
                args.random_state + comparison_idx * 1000 + 301 + partner_idx,
            )
            partner_shift = module_shift(partner["rep"].module_activity, partner["labels"], condition_value)
            partner_train_idx, partner_test_idx = split_indices_by_donor(partner_rows, random_state=args.random_state)
            module_lines.extend(module_reference_lines(condition_col, condition_value, partner_cell_type, partner, partner_shift))

            forward_edges = fit_directed_edges(
                condition_value,
                partner_cell_type,
                RECEIVER,
                partner,
                hep,
                hep_rows,
                partner_shift,
                hep_shift,
                hep_train_idx,
                hep_test_idx,
                args.alpha,
            )
            reverse_edges = fit_directed_edges(
                condition_value,
                RECEIVER,
                partner_cell_type,
                hep,
                partner,
                partner_rows,
                hep_shift,
                partner_shift,
                partner_train_idx,
                partner_test_idx,
                args.alpha,
            )
            pair_records = pair_bidirectional_records(
                condition_col,
                condition_value,
                partner_cell_type,
                forward_edges,
                reverse_edges,
                args.margin,
            )
            best_pair = max(pair_records, key=lambda item: item["forward_delta_r2"])
            summaries.append(
                {
                    "condition_col": condition_col,
                    "condition": condition_value,
                    "partner_cell_type": partner_cell_type,
                    "hep_cells": len(hep_rows),
                    "hep_label_counts": label_counts(hep["labels"]),
                    "partner_cells": len(partner_rows),
                    "partner_label_counts": label_counts(partner["labels"]),
                    "best_forward_delta_r2": best_pair["forward_delta_r2"],
                    "best_pair": best_pair,
                }
            )
            all_pairs.extend(pair_records)
            all_directed_edges.extend(forward_edges)
            all_directed_edges.extend(reverse_edges)
            print(
                f"      best forward {best_pair['forward_edge']}: "
                f"forward_delta_r2={best_pair['forward_delta_r2']:+.4f}, "
                f"reverse_delta_r2={best_pair['reverse_delta_r2']:+.4f}, "
                f"call={best_pair['direction_call']}",
                flush=True,
            )

    write_outputs(args.output_dir, summaries, all_pairs, all_directed_edges, module_lines)
    print("Saved:", flush=True)
    print(f"  {args.output_dir / 'bidirectional_pairs.tsv'}", flush=True)
    print(f"  {args.output_dir / 'directed_edges.tsv'}", flush=True)
    print(f"  {args.output_dir / 'summary.tsv'}", flush=True)
    print(f"  {args.output_dir / 'module_reference.tsv'}", flush=True)


if __name__ == "__main__":
    main()
