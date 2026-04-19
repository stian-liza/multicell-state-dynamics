from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from run_scp2154_phenotype_baseline import read_metadata
from run_scp2154_stromal_to_hepatocyte_coupling import (
    annotate_program,
    build_stromal_input,
    coupling_direction,
    fit_celltype_representation,
    module_shift,
    patient_phenotype_module_scores,
    per_module_r2,
    sample_celltype_rows,
    save_text,
    split_indices_by_donor,
    top_gene_names,
    tumor_relation,
    write_module_reference,
    zscore_by_train,
)
from multicell_dynamics import (
    fit_population_dynamics,
    read_10x_mtx_subset,
    velocity_r2_score,
    velocity_sign_agreement,
)


DEFAULT_SENDERS = ["Stromal", "Myeloid", "Endothelial", "TNKcell", "Bcell"]
RECEIVER = "Hepatocyte"


def label_counts(labels: np.ndarray) -> str:
    counts = Counter(str(label) for label in labels)
    return ";".join(f"{key}:{value}" for key, value in sorted(counts.items()))


def module_delta(shifts: list[tuple[int, float, float, float]], module_idx: int) -> float:
    for idx, _, _, delta in shifts:
        if idx == module_idx:
            return delta
    raise KeyError(module_idx)


def fit_baseline(hep: dict, train_idx: np.ndarray, test_idx: np.ndarray, alpha: float) -> dict:
    model = fit_population_dynamics(
        module_activity=hep["rep"].module_activity[train_idx],
        state_embedding=hep["embedding"][train_idx],
        module_velocity=hep["local_velocity"][train_idx],
        alpha=alpha,
    )
    prediction = model.predict_velocity(hep["rep"].module_activity[test_idx], hep["embedding"][test_idx])
    true = hep["local_velocity"][test_idx]
    return {
        "model": model,
        "prediction": prediction,
        "true": true,
        "per_module_r2": per_module_r2(true, prediction),
        "overall_r2": velocity_r2_score(true, prediction),
        "overall_sign": velocity_sign_agreement(true, prediction),
    }


def evaluate_sender(
    sender_cell_type: str,
    sender: dict,
    hep: dict,
    hep_rows: list[dict[str, str]],
    hep_shift: list[tuple[int, float, float, float]],
    baseline: dict,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    alpha: float,
) -> tuple[list[dict], dict]:
    sender_shift = module_shift(sender["rep"].module_activity, sender["labels"])
    selected_sender_module = max(sender_shift, key=lambda item: item[3])[0]
    sender_scores = patient_phenotype_module_scores(sender["rows"], sender["rep"].module_activity)
    raw_sender_input = build_stromal_input(hep_rows, sender_scores)
    sender_input = zscore_by_train(raw_sender_input, train_idx)
    true = baseline["true"]
    base_r2 = baseline["per_module_r2"]
    pair_records = []

    all_model = fit_population_dynamics(
        module_activity=hep["rep"].module_activity[train_idx],
        state_embedding=hep["embedding"][train_idx],
        module_velocity=hep["local_velocity"][train_idx],
        external_input=sender_input[train_idx],
        alpha=alpha,
    )
    all_prediction = all_model.predict_velocity(
        hep["rep"].module_activity[test_idx],
        hep["embedding"][test_idx],
        external_input=sender_input[test_idx],
    )
    all_r2 = per_module_r2(true, all_prediction)

    for source_module_idx in range(sender_input.shape[1]):
        single_input = sender_input[:, [source_module_idx]]
        single_model = fit_population_dynamics(
            module_activity=hep["rep"].module_activity[train_idx],
            state_embedding=hep["embedding"][train_idx],
            module_velocity=hep["local_velocity"][train_idx],
            external_input=single_input[train_idx],
            alpha=alpha,
        )
        single_prediction = single_model.predict_velocity(
            hep["rep"].module_activity[test_idx],
            hep["embedding"][test_idx],
            external_input=single_input[test_idx],
        )
        single_r2 = per_module_r2(true, single_prediction)
        single_e_idx = single_model.feature_names.index("e_0")
        all_e_idx = all_model.feature_names.index(f"e_{source_module_idx}")
        sender_delta = module_delta(sender_shift, source_module_idx)
        sender_genes = top_gene_names(sender["top_genes"][source_module_idx])
        for target_module_idx in range(hep["rep"].module_activity.shape[1]):
            hep_delta = module_delta(hep_shift, target_module_idx)
            hep_genes = top_gene_names(hep["top_genes"][target_module_idx])
            coeff = float(single_model.coefficient_matrix[target_module_idx, single_e_idx])
            delta_r2 = float(single_r2[target_module_idx] - base_r2[target_module_idx])
            pair_records.append(
                {
                    "sender_cell_type": sender_cell_type,
                    "sender_module": source_module_idx,
                    "sender_delta": sender_delta,
                    "sender_relation": tumor_relation(sender_delta),
                    "sender_program": annotate_program(sender_cell_type, sender_genes),
                    "sender_top_genes": ",".join(sender_genes),
                    "hep_module": target_module_idx,
                    "hep_delta": hep_delta,
                    "hep_relation": tumor_relation(hep_delta),
                    "hep_program": annotate_program(RECEIVER, hep_genes),
                    "hep_top_genes": ",".join(hep_genes),
                    "single_input_coeff": coeff,
                    "single_input_delta_r2": delta_r2,
                    "all_input_coeff": float(all_model.coefficient_matrix[target_module_idx, all_e_idx]),
                    "all_input_delta_r2": float(all_r2[target_module_idx] - base_r2[target_module_idx]),
                    "direction": coupling_direction(coeff, hep_delta, delta_r2),
                }
            )

    selected_pairs = [item for item in pair_records if item["sender_module"] == selected_sender_module]
    selected_score = sum(max(0.0, item["single_input_delta_r2"]) for item in selected_pairs)
    tumor_up_pairs = [item for item in pair_records if item["hep_relation"] == "Tumor-up"]
    best_pair = max(tumor_up_pairs or pair_records, key=lambda item: item["single_input_delta_r2"])
    summary = {
        "sender_cell_type": sender_cell_type,
        "sender_cells": len(sender["rows"]),
        "sender_label_counts": label_counts(sender["labels"]),
        "selected_sender_module": selected_sender_module,
        "selected_sender_delta": module_delta(sender_shift, selected_sender_module),
        "selected_sender_program": annotate_program(
            sender_cell_type,
            top_gene_names(sender["top_genes"][selected_sender_module]),
        ),
        "selected_positive_delta_r2_sum": selected_score,
        "best_sender_module": best_pair["sender_module"],
        "best_hep_module": best_pair["hep_module"],
        "best_hep_program": best_pair["hep_program"],
        "best_single_delta_r2": best_pair["single_input_delta_r2"],
        "best_single_coeff": best_pair["single_input_coeff"],
        "best_direction": best_pair["direction"],
        "all_input_overall_r2": velocity_r2_score(true, all_prediction),
        "all_input_overall_sign": velocity_sign_agreement(true, all_prediction),
    }
    return pair_records, summary


def save_outputs(
    output_dir: Path,
    summaries: list[dict],
    pair_records: list[dict],
    module_lines: list[str],
    baseline: dict,
) -> None:
    pair_records.sort(
        key=lambda item: (
            item["direction"] == "candidate_supports_tumor_up_module",
            item["single_input_delta_r2"],
            abs(item["single_input_coeff"]),
        ),
        reverse=True,
    )
    summaries.sort(key=lambda item: item["best_single_delta_r2"], reverse=True)

    summary_lines = [
        "rank\tsender_cell_type\tsender_cells\tsender_label_counts\tselected_sender_module\t"
        "selected_sender_delta\tselected_sender_program\tselected_positive_delta_r2_sum\t"
        "best_sender_module\tbest_hep_module\tbest_hep_program\tbest_single_delta_r2\t"
        "best_single_coeff\tbest_direction\tall_input_overall_r2\tall_input_overall_sign"
    ]
    for rank, item in enumerate(summaries, start=1):
        summary_lines.append(
            "\t".join(
                [
                    str(rank),
                    item["sender_cell_type"],
                    str(item["sender_cells"]),
                    item["sender_label_counts"],
                    f"m_{item['selected_sender_module']}",
                    f"{item['selected_sender_delta']:.6f}",
                    item["selected_sender_program"],
                    f"{item['selected_positive_delta_r2_sum']:.6f}",
                    f"m_{item['best_sender_module']}",
                    f"m_{item['best_hep_module']}",
                    item["best_hep_program"],
                    f"{item['best_single_delta_r2']:.6f}",
                    f"{item['best_single_coeff']:.6f}",
                    item["best_direction"],
                    f"{item['all_input_overall_r2']:.6f}",
                    f"{item['all_input_overall_sign']:.6f}",
                ]
            )
        )

    pair_lines = [
        "rank\tsender_cell_type\tsender_module\tsender_tumor_delta\tsender_relation\t"
        "sender_program\thep_module\thep_tumor_delta\thep_relation\thep_program\t"
        "single_input_coeff\tsingle_input_delta_r2\tall_input_coeff\tall_input_delta_r2\t"
        "direction\tsender_top_genes\thep_top_genes"
    ]
    for rank, item in enumerate(pair_records, start=1):
        pair_lines.append(
            "\t".join(
                [
                    str(rank),
                    item["sender_cell_type"],
                    f"m_{item['sender_module']}",
                    f"{item['sender_delta']:.6f}",
                    item["sender_relation"],
                    item["sender_program"],
                    f"m_{item['hep_module']}",
                    f"{item['hep_delta']:.6f}",
                    item["hep_relation"],
                    item["hep_program"],
                    f"{item['single_input_coeff']:.6f}",
                    f"{item['single_input_delta_r2']:.6f}",
                    f"{item['all_input_coeff']:.6f}",
                    f"{item['all_input_delta_r2']:.6f}",
                    item["direction"],
                    item["sender_top_genes"],
                    item["hep_top_genes"],
                ]
            )
        )

    baseline_lines = [
        "metric\tvalue",
        f"baseline_overall_r2\t{baseline['overall_r2']:.6f}",
        f"baseline_overall_sign\t{baseline['overall_sign']:.6f}",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    save_text(output_dir / "sender_summary.tsv", "\n".join(summary_lines) + "\n")
    save_text(output_dir / "coupling_pairs.tsv", "\n".join(pair_lines) + "\n")
    save_text(output_dir / "module_reference.tsv", "\n".join(module_lines) + "\n")
    save_text(output_dir / "baseline_metrics.tsv", "\n".join(baseline_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SCP2154 sender cell types for hepatocyte coupling")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--barcodes", type=Path, required=True)
    parser.add_argument("--sender-cell-types", default=",".join(DEFAULT_SENDERS))
    parser.add_argument("--max-cells-per-donor-phenotype", type=int, default=90)
    parser.add_argument("--max-cells-per-phenotype", type=int, default=1000)
    parser.add_argument("--hvg", type=int, default=800)
    parser.add_argument("--modules", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.03)
    parser.add_argument("--min-cells", type=int, default=150)
    parser.add_argument("--random-state", type=int, default=7)
    args = parser.parse_args()

    metadata = read_metadata(args.metadata)
    sender_cell_types = [item.strip() for item in args.sender_cell_types.split(",") if item.strip()]
    output_dir = Path("results/scp2154_sender_comparison")
    hep_rows = sample_celltype_rows(
        metadata,
        RECEIVER,
        args.max_cells_per_donor_phenotype,
        args.max_cells_per_phenotype,
        args.random_state + 101,
    )
    sampled_senders = {}
    for sender_idx, sender_cell_type in enumerate(sender_cell_types):
        sender_rows = sample_celltype_rows(
            metadata,
            sender_cell_type,
            args.max_cells_per_donor_phenotype,
            args.max_cells_per_phenotype,
            args.random_state + sender_idx,
        )
        if len(sender_rows) >= args.min_cells:
            sampled_senders[sender_cell_type] = sender_rows
        else:
            print(f"  {sender_cell_type}: skipped, only {len(sender_rows)} cells", flush=True)
    all_sampled_rows = hep_rows + [row for rows in sampled_senders.values() for row in rows]
    print(f"Reading one shared matrix subset for {len(all_sampled_rows)} cells...", flush=True)
    counts = read_10x_mtx_subset(args.matrix, args.features, args.barcodes, [row["NAME"] for row in all_sampled_rows])
    hep = fit_celltype_representation(counts, hep_rows, args.modules, args.hvg, args.random_state + 201)
    hep_shift = module_shift(hep["rep"].module_activity, hep["labels"])
    train_idx, test_idx = split_indices_by_donor(hep_rows, random_state=args.random_state)
    baseline = fit_baseline(hep, train_idx, test_idx, args.alpha)

    module_lines = [
        "cell_type\tmodule\ttumor_mean\thealthy_mean\tdelta_tumor_minus_healthy\t"
        "tumor_relation\tputative_program\tselected_sender_module\ttop_genes"
    ]
    module_lines.extend(write_module_reference(output_dir, RECEIVER, hep_shift, hep["top_genes"]))
    all_pairs: list[dict] = []
    summaries: list[dict] = []

    print("SCP2154 sender comparison", flush=True)
    print(f"{RECEIVER} cells: {len(hep_rows)} {dict(Counter(hep['labels']))}", flush=True)
    print("Baseline R2:", round(baseline["overall_r2"], 4), flush=True)
    print("Baseline sign:", round(baseline["overall_sign"], 4), flush=True)
    for sender_idx, sender_cell_type in enumerate(sender_cell_types):
        sender_rows = sampled_senders.get(sender_cell_type)
        if not sender_rows:
            continue
        print(f"  Fitting {sender_cell_type} sender modules for {len(sender_rows)} cells...", flush=True)
        sender = fit_celltype_representation(
            counts,
            sender_rows,
            args.modules,
            args.hvg,
            args.random_state + 301 + sender_idx,
        )
        pairs, summary = evaluate_sender(
            sender_cell_type,
            sender,
            hep,
            hep_rows,
            hep_shift,
            baseline,
            train_idx,
            test_idx,
            args.alpha,
        )
        sender_shift = module_shift(sender["rep"].module_activity, sender["labels"])
        module_lines.extend(
            write_module_reference(
                output_dir,
                sender_cell_type,
                sender_shift,
                sender["top_genes"],
                selected_module=summary["selected_sender_module"],
            )
        )
        all_pairs.extend(pairs)
        summaries.append(summary)
        print(
            f"  {sender_cell_type}: best_delta_r2={summary['best_single_delta_r2']:+.4f}, "
            f"best_pair={sender_cell_type}_m{summary['best_sender_module']} -> "
            f"{RECEIVER}_m{summary['best_hep_module']}, "
            f"selected_program={summary['selected_sender_program']}",
            flush=True,
        )

    save_outputs(output_dir, summaries, all_pairs, module_lines, baseline)
    print("Top sender summaries:", flush=True)
    for item in sorted(summaries, key=lambda row: row["best_single_delta_r2"], reverse=True)[:5]:
        print(
            f"  {item['sender_cell_type']}: best_delta_r2={item['best_single_delta_r2']:+.4f}, "
            f"best_pair=m{item['best_sender_module']} -> hep_m{item['best_hep_module']}, "
            f"selected_sender_m{item['selected_sender_module']}, "
            f"{item['selected_sender_program']}",
            flush=True,
        )
    print("Saved:", flush=True)
    print(f"  {output_dir / 'sender_summary.tsv'}", flush=True)
    print(f"  {output_dir / 'coupling_pairs.tsv'}", flush=True)
    print(f"  {output_dir / 'module_reference.tsv'}", flush=True)
    print(f"  {output_dir / 'baseline_metrics.tsv'}", flush=True)


if __name__ == "__main__":
    main()
