from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from run_scp2154_phenotype_baseline import read_metadata
from run_scp2154_sender_comparison import RECEIVER, evaluate_sender, fit_baseline, module_delta
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
    zscore_by_train,
)
from multicell_dynamics import fit_population_dynamics, read_10x_mtx_subset


def select_observed_pair(pair_records: list[dict]) -> dict:
    candidates = [
        item
        for item in pair_records
        if item["direction"] == "candidate_supports_tumor_up_module"
    ]
    if not candidates:
        candidates = pair_records
    return max(candidates, key=lambda item: item["single_input_delta_r2"])


def permute_sender_scores(
    sender_scores: dict[tuple[str, str], np.ndarray],
    rng: np.random.Generator,
) -> dict[tuple[str, str], np.ndarray]:
    keys = list(sender_scores)
    values = [sender_scores[key] for key in keys]
    order = rng.permutation(len(values))
    return {key: values[int(order[idx])] for idx, key in enumerate(keys)}


def fit_single_input_delta(
    hep: dict,
    sender_input: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    baseline_per_module_r2: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    model = fit_population_dynamics(
        module_activity=hep["rep"].module_activity[train_idx],
        state_embedding=hep["embedding"][train_idx],
        module_velocity=hep["local_velocity"][train_idx],
        external_input=sender_input[train_idx],
        alpha=alpha,
    )
    prediction = model.predict_velocity(
        hep["rep"].module_activity[test_idx],
        hep["embedding"][test_idx],
        external_input=sender_input[test_idx],
    )
    delta_r2 = per_module_r2(hep["local_velocity"][test_idx], prediction) - baseline_per_module_r2
    coeff = model.coefficient_matrix[:, model.feature_names.index("e_0")]
    return delta_r2, coeff


def permutation_record(
    perm_idx: int,
    permuted_input: np.ndarray,
    hep: dict,
    hep_shift: list[tuple[int, float, float, float]],
    observed_pair: dict,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    baseline_per_module_r2: np.ndarray,
    alpha: float,
) -> dict:
    selected_source = int(observed_pair["sender_module"])
    selected_target = int(observed_pair["hep_module"])
    selected_delta, selected_coeff = fit_single_input_delta(
        hep,
        permuted_input[:, [selected_source]],
        train_idx,
        test_idx,
        baseline_per_module_r2,
        alpha,
    )

    max_tumor_up_delta = -np.inf
    max_any_delta = -np.inf
    max_tumor_up_source = -1
    max_tumor_up_target = -1
    for source_module_idx in range(permuted_input.shape[1]):
        delta_r2, coeff = fit_single_input_delta(
            hep,
            permuted_input[:, [source_module_idx]],
            train_idx,
            test_idx,
            baseline_per_module_r2,
            alpha,
        )
        max_any_delta = max(max_any_delta, float(delta_r2.max()))
        for target_module_idx in range(delta_r2.shape[0]):
            hep_delta = module_delta(hep_shift, target_module_idx)
            direction = coupling_direction(float(coeff[target_module_idx]), hep_delta, float(delta_r2[target_module_idx]))
            if direction != "candidate_supports_tumor_up_module":
                continue
            if float(delta_r2[target_module_idx]) > max_tumor_up_delta:
                max_tumor_up_delta = float(delta_r2[target_module_idx])
                max_tumor_up_source = source_module_idx
                max_tumor_up_target = target_module_idx
    if not np.isfinite(max_tumor_up_delta):
        max_tumor_up_delta = max_any_delta

    return {
        "permutation": perm_idx,
        "selected_pair_delta_r2": float(selected_delta[selected_target]),
        "selected_pair_coeff": float(selected_coeff[selected_target]),
        "max_tumor_up_delta_r2": float(max_tumor_up_delta),
        "max_tumor_up_sender_module": max_tumor_up_source,
        "max_tumor_up_hep_module": max_tumor_up_target,
        "max_any_delta_r2": float(max_any_delta),
    }


def empirical_p_value(observed: float, null_values: np.ndarray) -> float:
    return float((1 + np.sum(null_values >= observed)) / (len(null_values) + 1))


def save_permutation_outputs(
    output_dir: Path,
    observed_pair: dict,
    permutation_records: list[dict],
    sender: dict,
    hep: dict,
    baseline: dict,
    summary: dict,
) -> None:
    selected_null = np.array([item["selected_pair_delta_r2"] for item in permutation_records], dtype=float)
    max_tumor_up_null = np.array([item["max_tumor_up_delta_r2"] for item in permutation_records], dtype=float)
    max_any_null = np.array([item["max_any_delta_r2"] for item in permutation_records], dtype=float)
    observed_delta = float(observed_pair["single_input_delta_r2"])

    selected_sender_genes = top_gene_names(sender["top_genes"][int(observed_pair["sender_module"])])
    selected_hep_genes = top_gene_names(hep["top_genes"][int(observed_pair["hep_module"])])
    observed_lines = [
        "field\tvalue",
        f"sender_cell_type\t{observed_pair['sender_cell_type']}",
        f"sender_module\tm_{observed_pair['sender_module']}",
        f"sender_program\t{observed_pair['sender_program']}",
        f"sender_tumor_delta\t{observed_pair['sender_delta']:.6f}",
        f"sender_top_genes\t{','.join(selected_sender_genes)}",
        f"hep_module\tm_{observed_pair['hep_module']}",
        f"hep_program\t{observed_pair['hep_program']}",
        f"hep_tumor_delta\t{observed_pair['hep_delta']:.6f}",
        f"hep_top_genes\t{','.join(selected_hep_genes)}",
        f"observed_delta_r2\t{observed_delta:.6f}",
        f"observed_coeff\t{observed_pair['single_input_coeff']:.6f}",
        f"observed_direction\t{observed_pair['direction']}",
        f"baseline_overall_r2\t{baseline['overall_r2']:.6f}",
        f"baseline_overall_sign\t{baseline['overall_sign']:.6f}",
        f"selected_pair_empirical_p\t{empirical_p_value(observed_delta, selected_null):.6f}",
        f"max_tumor_up_empirical_p\t{empirical_p_value(observed_delta, max_tumor_up_null):.6f}",
        f"max_any_empirical_p\t{empirical_p_value(observed_delta, max_any_null):.6f}",
        f"selected_pair_null_mean\t{selected_null.mean():.6f}",
        f"selected_pair_null_95pct\t{np.quantile(selected_null, 0.95):.6f}",
        f"max_tumor_up_null_95pct\t{np.quantile(max_tumor_up_null, 0.95):.6f}",
        f"max_any_null_95pct\t{np.quantile(max_any_null, 0.95):.6f}",
        f"sender_selected_module\tm_{summary['selected_sender_module']}",
        f"sender_selected_program\t{summary['selected_sender_program']}",
    ]

    null_lines = [
        "permutation\tselected_pair_delta_r2\tselected_pair_coeff\tmax_tumor_up_delta_r2\t"
        "max_tumor_up_sender_module\tmax_tumor_up_hep_module\tmax_any_delta_r2"
    ]
    for item in permutation_records:
        null_lines.append(
            "\t".join(
                [
                    str(item["permutation"]),
                    f"{item['selected_pair_delta_r2']:.6f}",
                    f"{item['selected_pair_coeff']:.6f}",
                    f"{item['max_tumor_up_delta_r2']:.6f}",
                    f"m_{item['max_tumor_up_sender_module']}",
                    f"m_{item['max_tumor_up_hep_module']}",
                    f"{item['max_any_delta_r2']:.6f}",
                ]
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_text(output_dir / "observed_pair.tsv", "\n".join(observed_lines) + "\n")
    save_text(output_dir / "null_distribution.tsv", "\n".join(null_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Permutation control for SCP2154 sender-to-hepatocyte coupling")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--barcodes", type=Path, required=True)
    parser.add_argument("--sender-cell-type", default="Stromal")
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--max-cells-per-donor-phenotype", type=int, default=90)
    parser.add_argument("--max-cells-per-phenotype", type=int, default=1000)
    parser.add_argument("--hvg", type=int, default=800)
    parser.add_argument("--modules", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.03)
    parser.add_argument("--random-state", type=int, default=7)
    args = parser.parse_args()

    metadata = read_metadata(args.metadata)
    hep_rows = sample_celltype_rows(
        metadata,
        RECEIVER,
        args.max_cells_per_donor_phenotype,
        args.max_cells_per_phenotype,
        args.random_state + 101,
    )
    sender_rows = sample_celltype_rows(
        metadata,
        args.sender_cell_type,
        args.max_cells_per_donor_phenotype,
        args.max_cells_per_phenotype,
        args.random_state,
    )
    all_rows = hep_rows + sender_rows
    print(f"Reading one shared matrix subset for {len(all_rows)} cells...", flush=True)
    counts = read_10x_mtx_subset(args.matrix, args.features, args.barcodes, [row["NAME"] for row in all_rows])

    hep = fit_celltype_representation(counts, hep_rows, args.modules, args.hvg, args.random_state + 201)
    sender = fit_celltype_representation(counts, sender_rows, args.modules, args.hvg, args.random_state + 301)
    hep_shift = module_shift(hep["rep"].module_activity, hep["labels"])
    train_idx, test_idx = split_indices_by_donor(hep_rows, random_state=args.random_state)
    baseline = fit_baseline(hep, train_idx, test_idx, args.alpha)
    pair_records, summary = evaluate_sender(
        args.sender_cell_type,
        sender,
        hep,
        hep_rows,
        hep_shift,
        baseline,
        train_idx,
        test_idx,
        args.alpha,
    )
    observed_pair = select_observed_pair(pair_records)
    sender_scores = patient_phenotype_module_scores(sender_rows, sender["rep"].module_activity)
    rng = np.random.default_rng(args.random_state + 999)
    permutation_records = []

    print("SCP2154 coupling permutation control", flush=True)
    print(f"{RECEIVER} cells: {len(hep_rows)} {dict(Counter(hep['labels']))}", flush=True)
    print(f"{args.sender_cell_type} cells: {len(sender_rows)} {dict(Counter(sender['labels']))}", flush=True)
    print(
        f"Observed pair: {args.sender_cell_type}_m{observed_pair['sender_module']} -> "
        f"{RECEIVER}_m{observed_pair['hep_module']}, delta_r2={observed_pair['single_input_delta_r2']:+.4f}",
        flush=True,
    )
    for perm_idx in range(args.permutations):
        permuted_scores = permute_sender_scores(sender_scores, rng)
        raw_input = build_stromal_input(hep_rows, permuted_scores)
        permuted_input = zscore_by_train(raw_input, train_idx)
        permutation_records.append(
            permutation_record(
                perm_idx,
                permuted_input,
                hep,
                hep_shift,
                observed_pair,
                train_idx,
                test_idx,
                baseline["per_module_r2"],
                args.alpha,
            )
        )
        if (perm_idx + 1) % max(1, args.permutations // 5) == 0:
            print(f"  permutations: {perm_idx + 1}/{args.permutations}", flush=True)

    output_dir = Path("results/scp2154_coupling_permutation")
    save_permutation_outputs(output_dir, observed_pair, permutation_records, sender, hep, baseline, summary)
    selected_null = np.array([item["selected_pair_delta_r2"] for item in permutation_records], dtype=float)
    max_tumor_up_null = np.array([item["max_tumor_up_delta_r2"] for item in permutation_records], dtype=float)
    observed_delta = float(observed_pair["single_input_delta_r2"])
    print("Permutation result:", flush=True)
    print(f"  selected_pair_empirical_p={empirical_p_value(observed_delta, selected_null):.4f}", flush=True)
    print(f"  max_tumor_up_empirical_p={empirical_p_value(observed_delta, max_tumor_up_null):.4f}", flush=True)
    print(f"  selected_pair_null_95pct={np.quantile(selected_null, 0.95):.4f}", flush=True)
    print(f"  max_tumor_up_null_95pct={np.quantile(max_tumor_up_null, 0.95):.4f}", flush=True)
    print("Saved:", flush=True)
    print(f"  {output_dir / 'observed_pair.tsv'}", flush=True)
    print(f"  {output_dir / 'null_distribution.tsv'}", flush=True)


if __name__ == "__main__":
    main()
