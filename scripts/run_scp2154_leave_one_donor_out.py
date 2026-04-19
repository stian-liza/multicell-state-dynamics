from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from run_scp2154_phenotype_baseline import read_metadata
from run_scp2154_sender_comparison import RECEIVER, fit_baseline, module_delta
from run_scp2154_stromal_to_hepatocyte_coupling import (
    annotate_program,
    build_stromal_input,
    fit_celltype_representation,
    module_shift,
    patient_phenotype_module_scores,
    per_module_r2,
    sample_celltype_rows,
    save_text,
    top_gene_names,
    tumor_relation,
    zscore_by_train,
)
from multicell_dynamics import fit_population_dynamics, read_10x_mtx_subset


def fit_fold(
    hep: dict,
    sender_input: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    sender_module: int,
    hep_module: int,
    alpha: float,
) -> dict:
    baseline = fit_baseline(hep, train_idx, test_idx, alpha)
    single_input = sender_input[:, [sender_module]]
    coupled = fit_population_dynamics(
        module_activity=hep["rep"].module_activity[train_idx],
        state_embedding=hep["embedding"][train_idx],
        module_velocity=hep["local_velocity"][train_idx],
        external_input=single_input[train_idx],
        alpha=alpha,
    )
    prediction = coupled.predict_velocity(
        hep["rep"].module_activity[test_idx],
        hep["embedding"][test_idx],
        external_input=single_input[test_idx],
    )
    coupled_r2 = per_module_r2(hep["local_velocity"][test_idx], prediction)
    coeff = float(coupled.coefficient_matrix[hep_module, coupled.feature_names.index("e_0")])
    return {
        "baseline_r2": float(baseline["per_module_r2"][hep_module]),
        "coupled_r2": float(coupled_r2[hep_module]),
        "delta_r2": float(coupled_r2[hep_module] - baseline["per_module_r2"][hep_module]),
        "coeff": coeff,
        "baseline_overall_r2": float(baseline["overall_r2"]),
    }


def donor_indices(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    out = {}
    for donor in sorted({row["donor_id"] for row in rows}):
        idx = [row_idx for row_idx, row in enumerate(rows) if row["donor_id"] == donor]
        out[donor] = np.array(idx, dtype=int)
    return out


def summarize_fold_records(records: list[dict]) -> list[str]:
    deltas = np.array([item["delta_r2"] for item in records], dtype=float)
    coeffs = np.array([item["coeff"] for item in records], dtype=float)
    test_cells = np.array([item["test_cells"] for item in records], dtype=int)
    positive = deltas > 0
    positive_coeff = coeffs > 0
    large_test_mask = test_cells >= 60
    large_test_positive = positive[large_test_mask] if large_test_mask.any() else np.array([], dtype=bool)
    return [
        "metric\tvalue",
        f"n_folds\t{len(records)}",
        f"positive_delta_folds\t{int(positive.sum())}",
        f"positive_delta_fraction\t{positive.mean():.6f}",
        f"positive_coeff_folds\t{int(positive_coeff.sum())}",
        f"positive_coeff_fraction\t{positive_coeff.mean():.6f}",
        f"n_large_test_folds\t{int(large_test_mask.sum())}",
        f"large_test_positive_delta_folds\t{int(large_test_positive.sum()) if len(large_test_positive) else 0}",
        f"large_test_positive_delta_fraction\t{large_test_positive.mean():.6f}" if len(large_test_positive) else "large_test_positive_delta_fraction\tnan",
        f"mean_delta_r2\t{deltas.mean():.6f}",
        f"median_delta_r2\t{np.median(deltas):.6f}",
        f"min_delta_r2\t{deltas.min():.6f}",
        f"max_delta_r2\t{deltas.max():.6f}",
        f"mean_coeff\t{coeffs.mean():.6f}",
        f"median_coeff\t{np.median(coeffs):.6f}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Leave-one-donor-out validation for SCP2154 stromal-hepatocyte coupling")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--barcodes", type=Path, required=True)
    parser.add_argument("--sender-cell-type", default="Stromal")
    parser.add_argument("--sender-module", type=int, default=0)
    parser.add_argument("--hep-module", type=int, default=2)
    parser.add_argument("--max-cells-per-donor-phenotype", type=int, default=90)
    parser.add_argument("--max-cells-per-phenotype", type=int, default=1000)
    parser.add_argument("--hvg", type=int, default=800)
    parser.add_argument("--modules", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.03)
    parser.add_argument("--min-test-cells", type=int, default=20)
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
    sender_shift = module_shift(sender["rep"].module_activity, sender["labels"])
    sender_scores = patient_phenotype_module_scores(sender_rows, sender["rep"].module_activity)
    raw_sender_input = build_stromal_input(hep_rows, sender_scores)

    donor_to_indices = donor_indices(hep_rows)
    fold_records = []
    print("SCP2154 leave-one-donor-out validation", flush=True)
    print(f"{RECEIVER} cells: {len(hep_rows)} {dict(Counter(hep['labels']))}", flush=True)
    print(f"{args.sender_cell_type} cells: {len(sender_rows)} {dict(Counter(sender['labels']))}", flush=True)
    print(
        f"Testing {args.sender_cell_type}_m{args.sender_module} -> {RECEIVER}_m{args.hep_module}",
        flush=True,
    )
    for donor, test_idx in donor_to_indices.items():
        if len(test_idx) < args.min_test_cells:
            continue
        train_idx = np.array([idx for idx in range(len(hep_rows)) if idx not in set(test_idx)], dtype=int)
        sender_input = zscore_by_train(raw_sender_input, train_idx)
        result = fit_fold(
            hep,
            sender_input,
            train_idx,
            test_idx,
            args.sender_module,
            args.hep_module,
            args.alpha,
        )
        test_labels = Counter(str(hep["labels"][idx]) for idx in test_idx)
        fold_records.append(
            {
                "donor_id": donor,
                "test_cells": len(test_idx),
                "test_label_counts": ";".join(f"{key}:{value}" for key, value in sorted(test_labels.items())),
                **result,
            }
        )
        print(
            f"  {donor}: cells={len(test_idx)}, delta_r2={result['delta_r2']:+.4f}, coeff={result['coeff']:+.4f}",
            flush=True,
        )

    sender_genes = top_gene_names(sender["top_genes"][args.sender_module])
    hep_genes = top_gene_names(hep["top_genes"][args.hep_module])
    fold_lines = [
        "donor_id\ttest_cells\ttest_label_counts\tbaseline_r2\tcoupled_r2\tdelta_r2\tcoeff\tbaseline_overall_r2"
    ]
    for item in fold_records:
        fold_lines.append(
            "\t".join(
                [
                    item["donor_id"],
                    str(item["test_cells"]),
                    item["test_label_counts"],
                    f"{item['baseline_r2']:.6f}",
                    f"{item['coupled_r2']:.6f}",
                    f"{item['delta_r2']:.6f}",
                    f"{item['coeff']:.6f}",
                    f"{item['baseline_overall_r2']:.6f}",
                ]
            )
        )
    summary_lines = summarize_fold_records(fold_records)
    summary_lines.extend(
        [
            f"sender_cell_type\t{args.sender_cell_type}",
            f"sender_module\tm_{args.sender_module}",
            f"sender_program\t{annotate_program(args.sender_cell_type, sender_genes)}",
            f"sender_tumor_delta\t{module_delta(sender_shift, args.sender_module):.6f}",
            f"sender_relation\t{tumor_relation(module_delta(sender_shift, args.sender_module))}",
            f"sender_top_genes\t{','.join(sender_genes)}",
            f"hep_module\tm_{args.hep_module}",
            f"hep_program\t{annotate_program(RECEIVER, hep_genes)}",
            f"hep_tumor_delta\t{module_delta(hep_shift, args.hep_module):.6f}",
            f"hep_relation\t{tumor_relation(module_delta(hep_shift, args.hep_module))}",
            f"hep_top_genes\t{','.join(hep_genes)}",
        ]
    )

    output_dir = Path("results/scp2154_leave_one_donor_out")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_text(output_dir / "fold_results.tsv", "\n".join(fold_lines) + "\n")
    save_text(output_dir / "summary.tsv", "\n".join(summary_lines) + "\n")
    print("Summary:", flush=True)
    for line in summary_lines[1:7]:
        print(f"  {line}", flush=True)
    print("Saved:", flush=True)
    print(f"  {output_dir / 'fold_results.tsv'}", flush=True)
    print(f"  {output_dir / 'summary.tsv'}", flush=True)


if __name__ == "__main__":
    main()
