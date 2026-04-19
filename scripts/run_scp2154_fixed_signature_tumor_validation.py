from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from run_scp2154_phenotype_baseline import read_metadata, stratified_sample
from run_scp2154_stromal_to_hepatocyte_coupling import save_text
from multicell_dynamics import log1p_library_normalize, read_10x_mtx_subset


STROMAL_SIGNATURES = {
    "caf_contractile": [
        "ACTA2",
        "TAGLN",
        "MYL9",
        "TPM2",
        "CALD1",
        "ADIRF",
        "SPARCL1",
        "VIM",
        "TIMP1",
        "IGFBP7",
        "LGALS1",
        "RGS5",
    ],
    "ecm_matrix": [
        "COL1A1",
        "COL1A2",
        "COL3A1",
        "DCN",
        "LUM",
        "BGN",
        "SPARC",
        "MGP",
        "C1R",
        "TIMP1",
        "IGFBP7",
        "LGALS1",
    ],
    "acute_phase_secretory": [
        "SAA1",
        "SAA2",
        "HP",
        "ORM1",
        "SERPINA1",
        "FGA",
        "FGB",
        "FGG",
        "APOA1",
        "APOC3",
        "ALB",
        "VTN",
    ],
}

HEPATOCYTE_SIGNATURES = {
    "malignant_like": [
        "IFI27",
        "SPINK1",
        "RARRES2",
        "GPC3",
        "EPCAM",
        "KRT19",
        "AFP",
        "MDK",
        "SPP1",
        "TACSTD2",
        "KRT8",
        "KRT18",
        "MALAT1",
    ],
    "secretory_stress": [
        "SAA1",
        "SAA2",
        "CRP",
        "HP",
        "ORM1",
        "SERPINA1",
        "FGA",
        "FGB",
        "FGG",
        "VTN",
        "APOA1",
        "APOC3",
    ],
    "immune_ambient": [
        "PTPRC",
        "CD3D",
        "CD3E",
        "NKG7",
        "CCL4",
        "CCL5",
        "CD74",
        "HLA-DRA",
        "SRGN",
        "TYROBP",
        "LYZ",
        "S100A9",
    ],
}


def sample_liver_rows(
    rows: list[dict[str, str]],
    cell_type: str,
    max_cells_per_donor_phenotype: int,
    max_cells_per_phenotype: int,
    random_state: int,
) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if row.get("organ__ontology_label") == "liver"
        and row.get("Cell_Type") == cell_type
        and row.get("indication") in {"healthy", "Tumor"}
    ]
    return stratified_sample(
        selected,
        phenotype_col="indication",
        donor_col="donor_id",
        max_cells_per_donor_phenotype=max_cells_per_donor_phenotype,
        max_cells_per_phenotype=max_cells_per_phenotype,
        random_state=random_state,
    )


def signature_scores(
    matrix: np.ndarray,
    gene_names: np.ndarray,
    signatures: dict[str, list[str]],
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    upper_to_index = {str(gene).upper(): idx for idx, gene in enumerate(gene_names)}
    mean = matrix.mean(axis=0, keepdims=True)
    std = np.maximum(matrix.std(axis=0, keepdims=True), 1e-6)
    z = (matrix - mean) / std
    scores: dict[str, np.ndarray] = {}
    used_genes: dict[str, list[str]] = {}
    for name, genes in signatures.items():
        matched = [gene for gene in genes if gene.upper() in upper_to_index]
        used_genes[name] = matched
        if not matched:
            scores[name] = np.zeros(matrix.shape[0], dtype=float)
            continue
        idx = np.array([upper_to_index[gene.upper()] for gene in matched], dtype=int)
        scores[name] = z[:, idx].mean(axis=1)
    return scores, used_genes


def celltype_matrix(counts, rows: list[dict[str, str]]) -> tuple[np.ndarray, list[dict[str, str]]]:
    cell_to_index = {cell: idx for idx, cell in enumerate(counts.cell_barcodes)}
    kept = [row for row in rows if row["NAME"] in cell_to_index]
    idx = np.array([cell_to_index[row["NAME"]] for row in kept], dtype=int)
    return counts.matrix[idx], kept


def mean_by_group(rows: list[dict[str, str]], values: np.ndarray, field: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[row[field]].append(float(values[idx]))
    return {key: float(np.mean(items)) for key, items in grouped.items()}


def donor_condition_means(
    rows: list[dict[str, str]],
    scores: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for idx, row in enumerate(rows):
        if not mask[idx]:
            continue
        donor = row["donor_id"]
        for score_name, values in scores.items():
            grouped[donor][score_name].append(float(values[idx]))
    out: dict[str, dict[str, float]] = {}
    for donor, by_score in grouped.items():
        out[donor] = {score_name: float(np.mean(values)) for score_name, values in by_score.items()}
        out[donor]["n_cells"] = float(len(next(iter(by_score.values())))) if by_score else 0.0
    return out


def tumor_hepatocyte_selection(
    rows: list[dict[str, str]],
    hep_scores: dict[str, np.ndarray],
    malignant_percentile: float,
    immune_percentile: float,
    selection_scope: str,
) -> np.ndarray:
    tumor_mask = np.array([row["indication"] == "Tumor" for row in rows], dtype=bool)
    malignant = hep_scores["malignant_like"]
    immune = hep_scores["immune_ambient"]
    selected = np.zeros(len(rows), dtype=bool)
    if selection_scope == "global":
        tumor_malignant = malignant[tumor_mask]
        tumor_immune = immune[tumor_mask]
        malignant_cutoff = float(np.percentile(tumor_malignant, malignant_percentile))
        immune_cutoff = float(np.percentile(tumor_immune, immune_percentile))
        return tumor_mask & (malignant >= malignant_cutoff) & (immune <= immune_cutoff)

    for donor in sorted({row["donor_id"] for row in rows if row["indication"] == "Tumor"}):
        donor_mask = np.array(
            [row["indication"] == "Tumor" and row["donor_id"] == donor for row in rows],
            dtype=bool,
        )
        if not donor_mask.any():
            continue
        malignant_cutoff = float(np.percentile(malignant[donor_mask], malignant_percentile))
        immune_cutoff = float(np.percentile(immune[donor_mask], immune_percentile))
        selected |= donor_mask & (malignant >= malignant_cutoff) & (immune <= immune_cutoff)
    return selected


def ridge_fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, float, float]:
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = np.maximum(x_train.std(axis=0, keepdims=True), 1e-8)
    y_mean = float(y_train.mean())
    x_train_z = (x_train - x_mean) / x_std
    x_test_z = (x_test - x_mean) / x_std
    y_centered = y_train - y_mean
    gram = x_train_z.T @ x_train_z + alpha * np.eye(x_train_z.shape[1])
    weights = np.linalg.solve(gram, x_train_z.T @ y_centered)
    pred = x_test_z @ weights + y_mean
    return pred, float(weights[0]), y_mean


def loo_prediction_test(x: np.ndarray, y: np.ndarray, alpha: float) -> dict:
    pred = np.zeros_like(y, dtype=float)
    baseline_pred = np.zeros_like(y, dtype=float)
    coeffs = np.zeros_like(y, dtype=float)
    for heldout_idx in range(len(y)):
        train_idx = np.array([idx for idx in range(len(y)) if idx != heldout_idx], dtype=int)
        test_idx = np.array([heldout_idx], dtype=int)
        pred[test_idx], coeff, baseline = ridge_fit_predict(x[train_idx], y[train_idx], x[test_idx], alpha)
        baseline_pred[test_idx] = baseline
        coeffs[heldout_idx] = coeff
    denom = float(np.sum((y - y.mean()) ** 2))
    if denom <= 1e-8:
        model_r2 = float("nan")
        baseline_r2 = float("nan")
    else:
        model_r2 = 1.0 - float(np.sum((y - pred) ** 2)) / denom
        baseline_r2 = 1.0 - float(np.sum((y - baseline_pred) ** 2)) / denom
    return {
        "loo_r2": model_r2,
        "loo_baseline_r2": baseline_r2,
        "loo_delta_r2": model_r2 - baseline_r2,
        "mean_coeff": float(coeffs.mean()),
        "positive_coeff_fraction": float(np.mean(coeffs > 0)),
        "pred": pred,
        "baseline_pred": baseline_pred,
    }


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) <= 1e-8 or np.std(y) <= 1e-8:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def empirical_p_value(observed: float, null_values: np.ndarray) -> float:
    return float((1 + np.sum(null_values >= observed)) / (len(null_values) + 1))


def build_donor_table(
    stromal_rows: list[dict[str, str]],
    hep_rows: list[dict[str, str]],
    stromal_scores: dict[str, np.ndarray],
    hep_scores: dict[str, np.ndarray],
    hep_selected: np.ndarray,
    min_stromal_cells: int,
    min_hep_cells: int,
) -> list[dict]:
    stromal_tumor_mask = np.array([row["indication"] == "Tumor" for row in stromal_rows], dtype=bool)
    stromal_by_donor = donor_condition_means(stromal_rows, stromal_scores, stromal_tumor_mask)
    hep_by_donor = donor_condition_means(hep_rows, hep_scores, hep_selected)
    donors = sorted(set(stromal_by_donor) & set(hep_by_donor))
    table = []
    for donor in donors:
        stromal_n = int(stromal_by_donor[donor]["n_cells"])
        hep_n = int(hep_by_donor[donor]["n_cells"])
        if stromal_n < min_stromal_cells or hep_n < min_hep_cells:
            continue
        row = {
            "donor_id": donor,
            "stromal_tumor_cells": stromal_n,
            "selected_tumor_hep_cells": hep_n,
        }
        for score_name in STROMAL_SIGNATURES:
            row[f"stromal_{score_name}"] = stromal_by_donor[donor][score_name]
        for score_name in HEPATOCYTE_SIGNATURES:
            row[f"hep_{score_name}"] = hep_by_donor[donor][score_name]
        table.append(row)
    return table


def evaluate_pairs(
    donor_table: list[dict],
    predictors: list[str],
    targets: list[str],
    alpha: float,
    permutations: int,
    random_state: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    rng = np.random.default_rng(random_state)
    pair_records = []
    fold_records = []
    null_records = []
    donors = [row["donor_id"] for row in donor_table]
    for predictor in predictors:
        for target in targets:
            x = np.array([[row[predictor]] for row in donor_table], dtype=float)
            y = np.array([row[target] for row in donor_table], dtype=float)
            observed = loo_prediction_test(x, y, alpha)
            null_values = []
            for perm_idx in range(permutations):
                permuted = x[rng.permutation(len(x))]
                permuted_result = loo_prediction_test(permuted, y, alpha)
                null_values.append(float(permuted_result["loo_delta_r2"]))
                null_records.append(
                    {
                        "predictor": predictor,
                        "target": target,
                        "permutation": perm_idx,
                        "loo_delta_r2": float(permuted_result["loo_delta_r2"]),
                    }
                )
            null_array = np.array(null_values, dtype=float)
            pair_records.append(
                {
                    "predictor": predictor,
                    "target": target,
                    "n_donors": len(donor_table),
                    "loo_r2": observed["loo_r2"],
                    "loo_baseline_r2": observed["loo_baseline_r2"],
                    "loo_delta_r2": observed["loo_delta_r2"],
                    "mean_coeff": observed["mean_coeff"],
                    "positive_coeff_fraction": observed["positive_coeff_fraction"],
                    "pearson_r": safe_corr(x.ravel(), y),
                    "empirical_p": empirical_p_value(float(observed["loo_delta_r2"]), null_array),
                    "null_mean": float(null_array.mean()) if len(null_array) else float("nan"),
                    "null_95pct": float(np.quantile(null_array, 0.95)) if len(null_array) else float("nan"),
                }
            )
            for idx, donor in enumerate(donors):
                fold_records.append(
                    {
                        "predictor": predictor,
                        "target": target,
                        "donor_id": donor,
                        "observed": float(y[idx]),
                        "predicted": float(observed["pred"][idx]),
                        "baseline_predicted": float(observed["baseline_pred"][idx]),
                        "prediction_error": float(y[idx] - observed["pred"][idx]),
                    }
                )
    pair_records.sort(key=lambda item: (item["loo_delta_r2"], item["mean_coeff"]), reverse=True)
    return pair_records, fold_records, null_records


def effect_label(coeff: float, delta_r2: float) -> str:
    if delta_r2 <= 0:
        return "no_heldout_gain"
    if coeff > 0:
        return "positive_predictive_relation"
    if coeff < 0:
        return "negative_predictive_relation"
    return "zero_coefficient"


def bidirectional_call(forward_delta: float, reverse_delta: float, margin: float = 0.05) -> str:
    if forward_delta > margin and forward_delta > reverse_delta + margin:
        return "stromal_to_hepatocyte_stronger"
    if reverse_delta > margin and reverse_delta > forward_delta + margin:
        return "hepatocyte_to_stromal_stronger"
    if forward_delta > margin and reverse_delta > margin:
        return "bidirectional_or_shared_donor_state"
    return "weak_or_inconclusive"


def build_bidirectional_records(
    forward_records: list[dict],
    reverse_records: list[dict],
    stromal_predictors: list[str],
    hep_predictor: str,
) -> list[dict]:
    forward_by_pair = {(row["predictor"], row["target"]): row for row in forward_records}
    reverse_by_pair = {(row["predictor"], row["target"]): row for row in reverse_records}
    out = []
    for stromal_predictor in stromal_predictors:
        forward = forward_by_pair[(stromal_predictor, hep_predictor)]
        reverse = reverse_by_pair[(hep_predictor, stromal_predictor)]
        out.append(
            {
                "stromal_signature": stromal_predictor,
                "hepatocyte_signature": hep_predictor,
                "forward_edge": f"{stromal_predictor}->{hep_predictor}",
                "reverse_edge": f"{hep_predictor}->{stromal_predictor}",
                "forward_loo_delta_r2": forward["loo_delta_r2"],
                "reverse_loo_delta_r2": reverse["loo_delta_r2"],
                "forward_minus_reverse": forward["loo_delta_r2"] - reverse["loo_delta_r2"],
                "forward_coeff": forward["mean_coeff"],
                "reverse_coeff": reverse["mean_coeff"],
                "forward_relation": effect_label(forward["mean_coeff"], forward["loo_delta_r2"]),
                "reverse_relation": effect_label(reverse["mean_coeff"], reverse["loo_delta_r2"]),
                "forward_empirical_p": forward["empirical_p"],
                "reverse_empirical_p": reverse["empirical_p"],
                "forward_pearson_r": forward["pearson_r"],
                "reverse_pearson_r": reverse["pearson_r"],
                "direction_call": bidirectional_call(forward["loo_delta_r2"], reverse["loo_delta_r2"]),
            }
        )
    out.sort(key=lambda row: row["forward_loo_delta_r2"], reverse=True)
    return out


def write_outputs(
    output_dir: Path,
    donor_table: list[dict],
    pair_records: list[dict],
    reverse_pair_records: list[dict],
    bidirectional_records: list[dict],
    fold_records: list[dict],
    reverse_fold_records: list[dict],
    null_records: list[dict],
    reverse_null_records: list[dict],
    signature_lines: list[str],
    summary_lines: list[str],
) -> None:
    donor_header = [
        "donor_id",
        "stromal_tumor_cells",
        "selected_tumor_hep_cells",
        *[f"stromal_{name}" for name in STROMAL_SIGNATURES],
        *[f"hep_{name}" for name in HEPATOCYTE_SIGNATURES],
    ]
    donor_lines = ["\t".join(donor_header)]
    for row in donor_table:
        donor_lines.append(
            "\t".join(
                [
                    str(row[key]) if isinstance(row[key], str) else f"{row[key]:.6f}"
                    for key in donor_header
                ]
            )
        )

    pair_lines = [
        "rank\tpredictor\ttarget\tn_donors\tloo_r2\tloo_baseline_r2\tloo_delta_r2\t"
        "mean_coeff\tpositive_coeff_fraction\tpearson_r\tempirical_p\tnull_mean\tnull_95pct"
    ]
    for rank, row in enumerate(pair_records, start=1):
        pair_lines.append(
            "\t".join(
                [
                    str(rank),
                    row["predictor"],
                    row["target"],
                    str(row["n_donors"]),
                    f"{row['loo_r2']:.6f}",
                    f"{row['loo_baseline_r2']:.6f}",
                    f"{row['loo_delta_r2']:.6f}",
                    f"{row['mean_coeff']:.6f}",
                    f"{row['positive_coeff_fraction']:.6f}",
                    f"{row['pearson_r']:.6f}",
                    f"{row['empirical_p']:.6f}",
                    f"{row['null_mean']:.6f}",
                    f"{row['null_95pct']:.6f}",
                ]
            )
        )

    reverse_pair_lines = [
        "rank\tpredictor\ttarget\tn_donors\tloo_r2\tloo_baseline_r2\tloo_delta_r2\t"
        "mean_coeff\tpositive_coeff_fraction\tpearson_r\tempirical_p\tnull_mean\tnull_95pct"
    ]
    for rank, row in enumerate(reverse_pair_records, start=1):
        reverse_pair_lines.append(
            "\t".join(
                [
                    str(rank),
                    row["predictor"],
                    row["target"],
                    str(row["n_donors"]),
                    f"{row['loo_r2']:.6f}",
                    f"{row['loo_baseline_r2']:.6f}",
                    f"{row['loo_delta_r2']:.6f}",
                    f"{row['mean_coeff']:.6f}",
                    f"{row['positive_coeff_fraction']:.6f}",
                    f"{row['pearson_r']:.6f}",
                    f"{row['empirical_p']:.6f}",
                    f"{row['null_mean']:.6f}",
                    f"{row['null_95pct']:.6f}",
                ]
            )
        )

    bidirectional_lines = [
        "rank\tstromal_signature\thepatocyte_signature\tforward_edge\treverse_edge\t"
        "forward_loo_delta_r2\treverse_loo_delta_r2\tforward_minus_reverse\tforward_coeff\treverse_coeff\t"
        "forward_relation\treverse_relation\tforward_empirical_p\treverse_empirical_p\t"
        "forward_pearson_r\treverse_pearson_r\tdirection_call"
    ]
    for rank, row in enumerate(bidirectional_records, start=1):
        bidirectional_lines.append(
            "\t".join(
                [
                    str(rank),
                    row["stromal_signature"],
                    row["hepatocyte_signature"],
                    row["forward_edge"],
                    row["reverse_edge"],
                    f"{row['forward_loo_delta_r2']:.6f}",
                    f"{row['reverse_loo_delta_r2']:.6f}",
                    f"{row['forward_minus_reverse']:.6f}",
                    f"{row['forward_coeff']:.6f}",
                    f"{row['reverse_coeff']:.6f}",
                    row["forward_relation"],
                    row["reverse_relation"],
                    f"{row['forward_empirical_p']:.6f}",
                    f"{row['reverse_empirical_p']:.6f}",
                    f"{row['forward_pearson_r']:.6f}",
                    f"{row['reverse_pearson_r']:.6f}",
                    row["direction_call"],
                ]
            )
        )

    all_fold_records = fold_records + reverse_fold_records
    fold_lines = ["predictor\ttarget\tdonor_id\tobserved\tpredicted\tbaseline_predicted\tprediction_error"]
    for row in all_fold_records:
        fold_lines.append(
            "\t".join(
                [
                    row["predictor"],
                    row["target"],
                    row["donor_id"],
                    f"{row['observed']:.6f}",
                    f"{row['predicted']:.6f}",
                    f"{row['baseline_predicted']:.6f}",
                    f"{row['prediction_error']:.6f}",
                ]
            )
        )

    all_null_records = null_records + reverse_null_records
    null_lines = ["predictor\ttarget\tpermutation\tloo_delta_r2"]
    for row in all_null_records:
        null_lines.append(
            "\t".join(
                [
                    row["predictor"],
                    row["target"],
                    str(row["permutation"]),
                    f"{row['loo_delta_r2']:.6f}",
                ]
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_text(output_dir / "summary.tsv", "\n".join(summary_lines) + "\n")
    save_text(output_dir / "donor_signature_table.tsv", "\n".join(donor_lines) + "\n")
    save_text(output_dir / "signature_pair_tests.tsv", "\n".join(pair_lines) + "\n")
    save_text(output_dir / "reverse_signature_pair_tests.tsv", "\n".join(reverse_pair_lines) + "\n")
    save_text(output_dir / "bidirectional_signature_pairs.tsv", "\n".join(bidirectional_lines) + "\n")
    save_text(output_dir / "leave_one_donor_predictions.tsv", "\n".join(fold_lines) + "\n")
    save_text(output_dir / "permutation_null.tsv", "\n".join(null_lines) + "\n")
    save_text(output_dir / "signature_reference.tsv", "\n".join(signature_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-signature Tumor validation for SCP2154 stromal-hepatocyte coupling")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--barcodes", type=Path, required=True)
    parser.add_argument("--max-cells-per-donor-phenotype", type=int, default=120)
    parser.add_argument("--max-cells-per-phenotype", type=int, default=1600)
    parser.add_argument("--malignant-percentile", type=float, default=60.0)
    parser.add_argument("--immune-percentile", type=float, default=80.0)
    parser.add_argument("--selection-scope", choices=["donor", "global"], default="donor")
    parser.add_argument("--min-stromal-cells-per-donor", type=int, default=8)
    parser.add_argument("--min-hep-cells-per-donor", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("results/scp2154_fixed_signature_tumor_validation"))
    args = parser.parse_args()

    metadata = read_metadata(args.metadata)
    stromal_rows = sample_liver_rows(
        metadata,
        "Stromal",
        args.max_cells_per_donor_phenotype,
        args.max_cells_per_phenotype,
        args.random_state,
    )
    hep_rows = sample_liver_rows(
        metadata,
        "Hepatocyte",
        args.max_cells_per_donor_phenotype,
        args.max_cells_per_phenotype,
        args.random_state + 101,
    )
    all_rows = stromal_rows + hep_rows
    print(f"Reading one shared matrix subset for {len(all_rows)} cells...", flush=True)
    counts = read_10x_mtx_subset(args.matrix, args.features, args.barcodes, [row["NAME"] for row in all_rows])

    stromal_counts, stromal_rows = celltype_matrix(counts, stromal_rows)
    hep_counts, hep_rows = celltype_matrix(counts, hep_rows)
    stromal_matrix = log1p_library_normalize(stromal_counts)
    hep_matrix = log1p_library_normalize(hep_counts)
    stromal_scores, stromal_used = signature_scores(stromal_matrix, counts.gene_names, STROMAL_SIGNATURES)
    hep_scores, hep_used = signature_scores(hep_matrix, counts.gene_names, HEPATOCYTE_SIGNATURES)
    hep_selected = tumor_hepatocyte_selection(
        hep_rows,
        hep_scores,
        args.malignant_percentile,
        args.immune_percentile,
        args.selection_scope,
    )

    donor_table = build_donor_table(
        stromal_rows,
        hep_rows,
        stromal_scores,
        hep_scores,
        hep_selected,
        args.min_stromal_cells_per_donor,
        args.min_hep_cells_per_donor,
    )
    if len(donor_table) < 4:
        raise SystemExit(f"Need at least 4 shared donors after filtering, got {len(donor_table)}.")

    predictors = [f"stromal_{name}" for name in STROMAL_SIGNATURES]
    targets = ["hep_malignant_like", "hep_secretory_stress"]
    pair_records, fold_records, null_records = evaluate_pairs(
        donor_table,
        predictors,
        targets,
        args.alpha,
        args.permutations,
        args.random_state + 999,
    )
    reverse_pair_records, reverse_fold_records, reverse_null_records = evaluate_pairs(
        donor_table,
        ["hep_secretory_stress"],
        ["stromal_caf_contractile", "stromal_ecm_matrix"],
        args.alpha,
        args.permutations,
        args.random_state + 1999,
    )
    bidirectional_records = build_bidirectional_records(
        pair_records,
        reverse_pair_records,
        ["stromal_caf_contractile", "stromal_ecm_matrix"],
        "hep_secretory_stress",
    )

    stromal_label_counts = Counter(row["indication"] for row in stromal_rows)
    hep_label_counts = Counter(row["indication"] for row in hep_rows)
    selected_counts = Counter(hep_rows[idx]["donor_id"] for idx, selected in enumerate(hep_selected) if selected)
    signature_lines = ["cell_type\tsignature\tmatched_genes"]
    for name, genes in stromal_used.items():
        signature_lines.append(f"Stromal\t{name}\t{','.join(genes)}")
    for name, genes in hep_used.items():
        signature_lines.append(f"Hepatocyte\t{name}\t{','.join(genes)}")

    best = pair_records[0]
    summary_lines = [
        "metric\tvalue",
        f"stromal_cells\t{len(stromal_rows)}",
        f"stromal_label_counts\t{';'.join(f'{key}:{value}' for key, value in sorted(stromal_label_counts.items()))}",
        f"hepatocyte_cells\t{len(hep_rows)}",
        f"hepatocyte_label_counts\t{';'.join(f'{key}:{value}' for key, value in sorted(hep_label_counts.items()))}",
        f"selected_tumor_hepatocytes\t{int(hep_selected.sum())}",
        f"selected_tumor_hepatocyte_donors\t{len(selected_counts)}",
        f"shared_donors_after_filter\t{len(donor_table)}",
        f"best_predictor\t{best['predictor']}",
        f"best_target\t{best['target']}",
        f"best_loo_delta_r2\t{best['loo_delta_r2']:.6f}",
        f"best_mean_coeff\t{best['mean_coeff']:.6f}",
        f"best_empirical_p\t{best['empirical_p']:.6f}",
        f"malignant_percentile\t{args.malignant_percentile:.2f}",
        f"immune_percentile\t{args.immune_percentile:.2f}",
        f"selection_scope\t{args.selection_scope}",
    ]

    write_outputs(
        args.output_dir,
        donor_table,
        pair_records,
        reverse_pair_records,
        bidirectional_records,
        fold_records,
        reverse_fold_records,
        null_records,
        reverse_null_records,
        signature_lines,
        summary_lines,
    )
    print("SCP2154 fixed-signature Tumor validation", flush=True)
    print(f"Shared donors after filtering: {len(donor_table)}", flush=True)
    print(f"Selected Tumor hepatocytes: {int(hep_selected.sum())}", flush=True)
    print(
        f"Best pair: {best['predictor']} -> {best['target']}, "
        f"loo_delta_r2={best['loo_delta_r2']:+.4f}, p={best['empirical_p']:.4f}",
        flush=True,
    )
    print("Saved:", flush=True)
    print(f"  {args.output_dir / 'summary.tsv'}", flush=True)
    print(f"  {args.output_dir / 'donor_signature_table.tsv'}", flush=True)
    print(f"  {args.output_dir / 'signature_pair_tests.tsv'}", flush=True)
    print(f"  {args.output_dir / 'bidirectional_signature_pairs.tsv'}", flush=True)
    print(f"  {args.output_dir / 'leave_one_donor_predictions.tsv'}", flush=True)
    print(f"  {args.output_dir / 'permutation_null.tsv'}", flush=True)
    print(f"  {args.output_dir / 'signature_reference.tsv'}", flush=True)


if __name__ == "__main__":
    main()
