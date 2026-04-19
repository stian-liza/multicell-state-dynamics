from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from run_scp2154_phenotype_baseline import read_metadata
from run_scp2154_sender_comparison import RECEIVER, module_delta
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
from multicell_dynamics import fit_population_dynamics, read_10x_mtx_subset, velocity_r2_score


SENDER = "Stromal"


def ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_mean = x_train.mean(axis=0, keepdims=True)
    y_mean = y_train.mean(axis=0, keepdims=True)
    x_centered = x_train - x_mean
    y_centered = y_train - y_mean
    gram = x_centered.T @ x_centered + alpha * np.eye(x_centered.shape[1])
    weights = np.linalg.solve(gram, x_centered.T @ y_centered)
    intercept = y_mean.ravel() - x_mean.ravel() @ weights
    prediction = x_test @ weights + intercept
    return prediction, weights, intercept


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std <= 1e-8 or y_std <= 1e-8:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def donor_condition_means(
    rows: list[dict[str, str]],
    values: np.ndarray,
) -> dict[tuple[str, str], np.ndarray]:
    grouped: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[(row["donor_id"], row["indication"])].append(values[idx])
    return {key: np.mean(np.vstack(items), axis=0) for key, items in grouped.items()}


def shared_sender_receiver_scores(
    sender_rows: list[dict[str, str]],
    receiver_rows: list[dict[str, str]],
    sender_activity: np.ndarray,
    receiver_activity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sender_scores = donor_condition_means(sender_rows, sender_activity)
    receiver_scores = donor_condition_means(receiver_rows, receiver_activity)
    keys = sorted(set(sender_scores) & set(receiver_scores))
    sender_matrix = np.vstack([sender_scores[key] for key in keys]) if keys else np.zeros((0, sender_activity.shape[1]))
    receiver_matrix = np.vstack([receiver_scores[key] for key in keys]) if keys else np.zeros((0, receiver_activity.shape[1]))
    return sender_matrix, receiver_matrix


def fit_static_models(
    receiver,
    single_sender_input: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target = receiver["rep"].module_activity
    base_x = receiver["embedding"]
    base_pred, _, _ = ridge_predict(base_x[train_idx], target[train_idx], base_x[test_idx], alpha)
    base_r2 = per_module_r2(target[test_idx], base_pred)

    coupled_x = np.concatenate([receiver["embedding"], single_sender_input], axis=1)
    coupled_pred, weights, _ = ridge_predict(coupled_x[train_idx], target[train_idx], coupled_x[test_idx], alpha)
    coupled_r2 = per_module_r2(target[test_idx], coupled_pred)
    sender_weights = weights[-single_sender_input.shape[1] :, :].T
    return base_r2, coupled_r2, coupled_r2 - base_r2, sender_weights


def fit_dynamic_models(
    receiver,
    single_sender_input: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    baseline = fit_population_dynamics(
        module_activity=receiver["rep"].module_activity[train_idx],
        state_embedding=receiver["embedding"][train_idx],
        module_velocity=receiver["local_velocity"][train_idx],
        alpha=alpha,
    )
    pred_base = baseline.predict_velocity(receiver["rep"].module_activity[test_idx], receiver["embedding"][test_idx])
    base_r2 = per_module_r2(receiver["local_velocity"][test_idx], pred_base)

    coupled = fit_population_dynamics(
        module_activity=receiver["rep"].module_activity[train_idx],
        state_embedding=receiver["embedding"][train_idx],
        module_velocity=receiver["local_velocity"][train_idx],
        external_input=single_sender_input[train_idx],
        alpha=alpha,
    )
    pred_coupled = coupled.predict_velocity(
        receiver["rep"].module_activity[test_idx],
        receiver["embedding"][test_idx],
        external_input=single_sender_input[test_idx],
    )
    coupled_r2 = per_module_r2(receiver["local_velocity"][test_idx], pred_coupled)
    sender_start = coupled.feature_names.index("e_0")
    sender_coefficients = coupled.coefficient_matrix[:, sender_start : sender_start + single_sender_input.shape[1]]
    return base_r2, coupled_r2, coupled_r2 - base_r2, sender_coefficients


def write_outputs(
    output_dir: Path,
    records: list[dict],
    summary_lines: list[str],
    module_lines: list[str],
) -> None:
    records.sort(
        key=lambda item: (
            item["dynamic_advantage"],
            item["dynamic_delta_r2"],
            abs(item["dynamic_coeff"]),
        ),
        reverse=True,
    )
    comparison_lines = [
        "rank\tsender_module\thep_module\tstatic_corr\tstatic_delta_r2\tstatic_coeff\t"
        "dynamic_delta_r2\tdynamic_coeff\tdynamic_advantage\thep_tumor_delta\thep_relation\t"
        "hep_program\tsender_tumor_delta\tsender_relation\tsender_program\tinterpretation\t"
        "sender_top_genes\thep_top_genes"
    ]
    for rank, item in enumerate(records, start=1):
        comparison_lines.append(
            "\t".join(
                [
                    str(rank),
                    f"m_{item['sender_module']}",
                    f"m_{item['hep_module']}",
                    f"{item['static_corr']:.6f}",
                    f"{item['static_delta_r2']:.6f}",
                    f"{item['static_coeff']:.6f}",
                    f"{item['dynamic_delta_r2']:.6f}",
                    f"{item['dynamic_coeff']:.6f}",
                    f"{item['dynamic_advantage']:.6f}",
                    f"{item['hep_tumor_delta']:.6f}",
                    item["hep_relation"],
                    item["hep_program"],
                    f"{item['sender_tumor_delta']:.6f}",
                    item["sender_relation"],
                    item["sender_program"],
                    item["interpretation"],
                    item["sender_top_genes"],
                    item["hep_top_genes"],
                ]
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_text(output_dir / "comparison_pairs.tsv", "\n".join(comparison_lines) + "\n")
    save_text(output_dir / "summary.tsv", "\n".join(summary_lines) + "\n")
    save_text(output_dir / "module_reference.tsv", "\n".join(module_lines) + "\n")


def interpretation(static_delta: float, dynamic_delta: float, corr: float) -> str:
    if dynamic_delta > 0 and static_delta <= 0:
        return "dynamic_only_gain"
    if dynamic_delta > static_delta and dynamic_delta > 0:
        return "dynamic_stronger_than_static"
    if static_delta > dynamic_delta and static_delta > 0:
        return "static_stronger_than_dynamic"
    if abs(corr) >= 0.5 and dynamic_delta <= 0:
        return "static_correlation_without_dynamic_gain"
    return "weak_or_mixed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare static association and dynamic coupling on SCP2154")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--barcodes", type=Path, required=True)
    parser.add_argument("--max-cells-per-donor-phenotype", type=int, default=90)
    parser.add_argument("--max-cells-per-phenotype", type=int, default=1000)
    parser.add_argument("--hvg", type=int, default=800)
    parser.add_argument("--modules", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.03)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("results/scp2154_static_vs_dynamic"))
    args = parser.parse_args()

    metadata = read_metadata(args.metadata)
    receiver_rows = sample_celltype_rows(
        metadata,
        RECEIVER,
        args.max_cells_per_donor_phenotype,
        args.max_cells_per_phenotype,
        args.random_state + 101,
    )
    sender_rows = sample_celltype_rows(
        metadata,
        SENDER,
        args.max_cells_per_donor_phenotype,
        args.max_cells_per_phenotype,
        args.random_state,
    )
    all_rows = receiver_rows + sender_rows
    print(f"Reading one shared matrix subset for {len(all_rows)} cells...", flush=True)
    counts = read_10x_mtx_subset(args.matrix, args.features, args.barcodes, [row["NAME"] for row in all_rows])

    receiver = fit_celltype_representation(counts, receiver_rows, args.modules, args.hvg, args.random_state + 201)
    sender = fit_celltype_representation(counts, sender_rows, args.modules, args.hvg, args.random_state + 301)
    receiver_shift = module_shift(receiver["rep"].module_activity, receiver["labels"])
    sender_shift = module_shift(sender["rep"].module_activity, sender["labels"])
    sender_scores = patient_phenotype_module_scores(sender_rows, sender["rep"].module_activity)
    raw_sender_input = build_stromal_input(receiver_rows, sender_scores)
    train_idx, test_idx = split_indices_by_donor(receiver_rows, random_state=args.random_state)
    sender_input = zscore_by_train(raw_sender_input, train_idx)

    sender_means, receiver_means = shared_sender_receiver_scores(
        sender_rows,
        receiver_rows,
        sender["rep"].module_activity,
        receiver["rep"].module_activity,
    )

    records = []
    static_delta_values = []
    dynamic_delta_values = []
    for sender_module in range(sender_input.shape[1]):
        sender_genes = top_gene_names(sender["top_genes"][sender_module])
        sender_delta = module_delta(sender_shift, sender_module)
        single_sender_input = sender_input[:, [sender_module]]
        _, _, static_delta_r2, static_coeff = fit_static_models(
            receiver,
            single_sender_input,
            train_idx,
            test_idx,
            args.alpha,
        )
        _, _, dynamic_delta_r2, dynamic_coeff = fit_dynamic_models(
            receiver,
            single_sender_input,
            train_idx,
            test_idx,
            args.alpha,
        )
        static_delta_values.extend(float(value) for value in static_delta_r2)
        dynamic_delta_values.extend(float(value) for value in dynamic_delta_r2)
        for hep_module in range(receiver["rep"].module_activity.shape[1]):
            hep_genes = top_gene_names(receiver["top_genes"][hep_module])
            hep_delta = module_delta(receiver_shift, hep_module)
            corr = safe_corr(sender_means[:, sender_module], receiver_means[:, hep_module])
            static_delta = float(static_delta_r2[hep_module])
            dynamic_delta = float(dynamic_delta_r2[hep_module])
            item_interpretation = interpretation(static_delta, dynamic_delta, corr)
            records.append(
                {
                    "sender_module": sender_module,
                    "hep_module": hep_module,
                    "static_corr": corr,
                    "static_delta_r2": static_delta,
                    "static_coeff": float(static_coeff[hep_module, 0]),
                    "dynamic_delta_r2": dynamic_delta,
                    "dynamic_coeff": float(dynamic_coeff[hep_module, 0]),
                    "dynamic_advantage": dynamic_delta - static_delta,
                    "hep_tumor_delta": hep_delta,
                    "hep_relation": tumor_relation(hep_delta),
                    "hep_program": annotate_program(RECEIVER, hep_genes),
                    "sender_tumor_delta": sender_delta,
                    "sender_relation": tumor_relation(sender_delta),
                    "sender_program": annotate_program(SENDER, sender_genes),
                    "interpretation": item_interpretation,
                    "sender_top_genes": ",".join(sender_genes),
                    "hep_top_genes": ",".join(hep_genes),
                }
            )

    dynamic_only = sum(item["interpretation"] == "dynamic_only_gain" for item in records)
    dynamic_stronger = sum(item["interpretation"] == "dynamic_stronger_than_static" for item in records)
    static_stronger = sum(item["interpretation"] == "static_stronger_than_dynamic" for item in records)
    top_dynamic = max(records, key=lambda item: item["dynamic_delta_r2"])
    top_advantage = max(records, key=lambda item: item["dynamic_advantage"])
    summary_lines = [
        "metric\tvalue",
        f"receiver_cells\t{len(receiver_rows)}",
        f"receiver_label_counts\t{';'.join(f'{k}:{v}' for k, v in sorted(Counter(receiver['labels']).items()))}",
        f"sender_cells\t{len(sender_rows)}",
        f"sender_label_counts\t{';'.join(f'{k}:{v}' for k, v in sorted(Counter(sender['labels']).items()))}",
        f"n_shared_donor_phenotype_points\t{len(sender_means)}",
        f"static_baseline_overall_r2\t{velocity_r2_score(receiver['rep'].module_activity[test_idx], np.zeros_like(receiver['rep'].module_activity[test_idx]) + receiver['rep'].module_activity[train_idx].mean(axis=0)):.6f}",
        f"mean_static_delta_r2\t{float(np.mean(static_delta_values)):.6f}",
        f"mean_dynamic_delta_r2\t{float(np.mean(dynamic_delta_values)):.6f}",
        f"dynamic_only_gain_pairs\t{dynamic_only}",
        f"dynamic_stronger_than_static_pairs\t{dynamic_stronger}",
        f"static_stronger_than_dynamic_pairs\t{static_stronger}",
        f"top_dynamic_pair\tStromal_m{top_dynamic['sender_module']}->Hepatocyte_m{top_dynamic['hep_module']}",
        f"top_dynamic_delta_r2\t{top_dynamic['dynamic_delta_r2']:.6f}",
        f"top_dynamic_static_delta_r2\t{top_dynamic['static_delta_r2']:.6f}",
        f"top_dynamic_static_corr\t{top_dynamic['static_corr']:.6f}",
        f"top_advantage_pair\tStromal_m{top_advantage['sender_module']}->Hepatocyte_m{top_advantage['hep_module']}",
        f"top_advantage_dynamic_minus_static\t{top_advantage['dynamic_advantage']:.6f}",
    ]

    module_lines = [
        "cell_type\tmodule\ttumor_mean\thealthy_mean\tdelta_tumor_minus_healthy\t"
        "tumor_relation\tputative_program\ttop_genes"
    ]
    for cell_type, shifts, rep in [(RECEIVER, receiver_shift, receiver), (SENDER, sender_shift, sender)]:
        for module_idx, tumor_mean, healthy_mean, delta in shifts:
            genes = top_gene_names(rep["top_genes"][module_idx])
            module_lines.append(
                "\t".join(
                    [
                        cell_type,
                        f"m_{module_idx}",
                        f"{tumor_mean:.6f}",
                        f"{healthy_mean:.6f}",
                        f"{delta:.6f}",
                        tumor_relation(delta),
                        annotate_program(cell_type, genes),
                        ",".join(genes),
                    ]
                )
            )

    write_outputs(args.output_dir, records, summary_lines, module_lines)
    print("SCP2154 static vs dynamic comparison", flush=True)
    print(f"Top dynamic pair: Stromal_m{top_dynamic['sender_module']} -> Hepatocyte_m{top_dynamic['hep_module']}", flush=True)
    print(
        f"  dynamic_delta_r2={top_dynamic['dynamic_delta_r2']:+.4f}, "
        f"static_delta_r2={top_dynamic['static_delta_r2']:+.4f}, "
        f"static_corr={top_dynamic['static_corr']:+.4f}",
        flush=True,
    )
    print(f"Dynamic-only gain pairs: {dynamic_only}", flush=True)
    print(f"Dynamic stronger than static pairs: {dynamic_stronger}", flush=True)
    print("Saved:", flush=True)
    print(f"  {args.output_dir / 'comparison_pairs.tsv'}", flush=True)
    print(f"  {args.output_dir / 'summary.tsv'}", flush=True)
    print(f"  {args.output_dir / 'module_reference.tsv'}", flush=True)


if __name__ == "__main__":
    main()
