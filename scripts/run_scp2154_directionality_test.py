from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from run_scp2154_phenotype_baseline import read_metadata
from run_scp2154_sender_comparison import module_delta
from run_scp2154_stromal_to_hepatocyte_coupling import (
    annotate_program,
    build_stromal_input,
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


def fit_direction(
    receiver: dict,
    receiver_rows: list[dict[str, str]],
    sender: dict,
    receiver_module: int,
    sender_module: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    alpha: float,
) -> dict:
    sender_scores = patient_phenotype_module_scores(sender["rows"], sender["rep"].module_activity)
    raw_sender_input = build_stromal_input(receiver_rows, sender_scores)
    sender_input = zscore_by_train(raw_sender_input, train_idx)[:, [sender_module]]

    baseline = fit_population_dynamics(
        module_activity=receiver["rep"].module_activity[train_idx],
        state_embedding=receiver["embedding"][train_idx],
        module_velocity=receiver["local_velocity"][train_idx],
        alpha=alpha,
    )
    base_pred = baseline.predict_velocity(
        receiver["rep"].module_activity[test_idx],
        receiver["embedding"][test_idx],
    )
    base_r2 = per_module_r2(receiver["local_velocity"][test_idx], base_pred)

    coupled = fit_population_dynamics(
        module_activity=receiver["rep"].module_activity[train_idx],
        state_embedding=receiver["embedding"][train_idx],
        module_velocity=receiver["local_velocity"][train_idx],
        external_input=sender_input[train_idx],
        alpha=alpha,
    )
    coupled_pred = coupled.predict_velocity(
        receiver["rep"].module_activity[test_idx],
        receiver["embedding"][test_idx],
        external_input=sender_input[test_idx],
    )
    coupled_r2 = per_module_r2(receiver["local_velocity"][test_idx], coupled_pred)
    coeff = float(coupled.coefficient_matrix[receiver_module, coupled.feature_names.index("e_0")])
    return {
        "baseline_r2": float(base_r2[receiver_module]),
        "coupled_r2": float(coupled_r2[receiver_module]),
        "delta_r2": float(coupled_r2[receiver_module] - base_r2[receiver_module]),
        "coeff": coeff,
    }


def direction_call(forward_delta: float, reverse_delta: float, margin: float) -> str:
    if forward_delta > margin and forward_delta > reverse_delta + margin:
        return "forward_stronger"
    if reverse_delta > margin and reverse_delta > forward_delta + margin:
        return "reverse_stronger"
    if forward_delta > margin and reverse_delta > margin:
        return "bidirectional_or_shared_stage"
    return "weak_or_inconclusive"


def main() -> None:
    parser = argparse.ArgumentParser(description="Directionality test for SCP2154 stromal-hepatocyte coupling")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--barcodes", type=Path, required=True)
    parser.add_argument("--stromal-module", type=int, default=0)
    parser.add_argument("--hep-module", type=int, default=2)
    parser.add_argument("--max-cells-per-donor-phenotype", type=int, default=90)
    parser.add_argument("--max-cells-per-phenotype", type=int, default=1000)
    parser.add_argument("--hvg", type=int, default=800)
    parser.add_argument("--modules", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.03)
    parser.add_argument("--margin", type=float, default=0.01)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("results/scp2154_directionality_test"))
    args = parser.parse_args()

    metadata = read_metadata(args.metadata)
    hep_rows = sample_celltype_rows(
        metadata,
        "Hepatocyte",
        args.max_cells_per_donor_phenotype,
        args.max_cells_per_phenotype,
        args.random_state + 101,
    )
    stromal_rows = sample_celltype_rows(
        metadata,
        "Stromal",
        args.max_cells_per_donor_phenotype,
        args.max_cells_per_phenotype,
        args.random_state,
    )
    all_rows = hep_rows + stromal_rows
    print(f"Reading one shared matrix subset for {len(all_rows)} cells...", flush=True)
    counts = read_10x_mtx_subset(args.matrix, args.features, args.barcodes, [row["NAME"] for row in all_rows])

    hep = fit_celltype_representation(counts, hep_rows, args.modules, args.hvg, args.random_state + 201)
    stromal = fit_celltype_representation(counts, stromal_rows, args.modules, args.hvg, args.random_state + 301)
    hep_train_idx, hep_test_idx = split_indices_by_donor(hep_rows, random_state=args.random_state)
    stromal_train_idx, stromal_test_idx = split_indices_by_donor(stromal_rows, random_state=args.random_state)

    forward = fit_direction(
        receiver=hep,
        receiver_rows=hep_rows,
        sender=stromal,
        receiver_module=args.hep_module,
        sender_module=args.stromal_module,
        train_idx=hep_train_idx,
        test_idx=hep_test_idx,
        alpha=args.alpha,
    )
    reverse = fit_direction(
        receiver=stromal,
        receiver_rows=stromal_rows,
        sender=hep,
        receiver_module=args.stromal_module,
        sender_module=args.hep_module,
        train_idx=stromal_train_idx,
        test_idx=stromal_test_idx,
        alpha=args.alpha,
    )

    hep_shift = module_shift(hep["rep"].module_activity, hep["labels"])
    stromal_shift = module_shift(stromal["rep"].module_activity, stromal["labels"])
    hep_genes = top_gene_names(hep["top_genes"][args.hep_module])
    stromal_genes = top_gene_names(stromal["top_genes"][args.stromal_module])
    call = direction_call(forward["delta_r2"], reverse["delta_r2"], args.margin)
    ratio = forward["delta_r2"] / reverse["delta_r2"] if abs(reverse["delta_r2"]) > 1e-8 else float("inf")

    summary_lines = [
        "metric\tvalue",
        f"forward_edge\tStromal_m{args.stromal_module}->dHepatocyte_m{args.hep_module}/dt",
        f"reverse_edge\tHepatocyte_m{args.hep_module}->dStromal_m{args.stromal_module}/dt",
        f"forward_delta_r2\t{forward['delta_r2']:.6f}",
        f"reverse_delta_r2\t{reverse['delta_r2']:.6f}",
        f"forward_minus_reverse\t{forward['delta_r2'] - reverse['delta_r2']:.6f}",
        f"forward_reverse_ratio\t{ratio:.6f}",
        f"forward_coeff\t{forward['coeff']:.6f}",
        f"reverse_coeff\t{reverse['coeff']:.6f}",
        f"direction_call\t{call}",
        f"hep_cells\t{len(hep_rows)}",
        f"hep_label_counts\t{';'.join(f'{k}:{v}' for k, v in sorted(Counter(hep['labels']).items()))}",
        f"stromal_cells\t{len(stromal_rows)}",
        f"stromal_label_counts\t{';'.join(f'{k}:{v}' for k, v in sorted(Counter(stromal['labels']).items()))}",
        f"stromal_module\tm_{args.stromal_module}",
        f"stromal_tumor_delta\t{module_delta(stromal_shift, args.stromal_module):.6f}",
        f"stromal_relation\t{tumor_relation(module_delta(stromal_shift, args.stromal_module))}",
        f"stromal_program\t{annotate_program('Stromal', stromal_genes)}",
        f"stromal_top_genes\t{','.join(stromal_genes)}",
        f"hep_module\tm_{args.hep_module}",
        f"hep_tumor_delta\t{module_delta(hep_shift, args.hep_module):.6f}",
        f"hep_relation\t{tumor_relation(module_delta(hep_shift, args.hep_module))}",
        f"hep_program\t{annotate_program('Hepatocyte', hep_genes)}",
        f"hep_top_genes\t{','.join(hep_genes)}",
    ]
    details = [
        "direction\tbaseline_r2\tcoupled_r2\tdelta_r2\tcoeff",
        f"forward\t{forward['baseline_r2']:.6f}\t{forward['coupled_r2']:.6f}\t{forward['delta_r2']:.6f}\t{forward['coeff']:.6f}",
        f"reverse\t{reverse['baseline_r2']:.6f}\t{reverse['coupled_r2']:.6f}\t{reverse['delta_r2']:.6f}\t{reverse['coeff']:.6f}",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_text(args.output_dir / "summary.tsv", "\n".join(summary_lines) + "\n")
    save_text(args.output_dir / "direction_details.tsv", "\n".join(details) + "\n")

    print("SCP2154 directionality test", flush=True)
    print(f"Forward delta_r2: {forward['delta_r2']:+.4f}", flush=True)
    print(f"Reverse delta_r2: {reverse['delta_r2']:+.4f}", flush=True)
    print(f"Direction call: {call}", flush=True)
    print("Saved:", flush=True)
    print(f"  {args.output_dir / 'summary.tsv'}", flush=True)
    print(f"  {args.output_dir / 'direction_details.tsv'}", flush=True)


if __name__ == "__main__":
    main()
