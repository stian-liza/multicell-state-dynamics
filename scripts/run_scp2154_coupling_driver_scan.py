from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from run_scp2154_fixed_signature_tumor_validation import (
    HEPATOCYTE_SIGNATURES,
    STROMAL_SIGNATURES,
    build_donor_table,
    celltype_matrix,
    empirical_p_value,
    loo_prediction_test,
    safe_corr,
    sample_liver_rows,
    signature_scores,
    tumor_hepatocyte_selection,
)
from run_scp2154_phenotype_baseline import read_metadata
from run_scp2154_stromal_to_hepatocyte_coupling import save_text
from multicell_dynamics import log1p_library_normalize, read_10x_mtx_subset


DRIVER_SIGNATURES = {
    "Stromal": {
        **STROMAL_SIGNATURES,
        "inflammatory_caf": ["IL6", "CXCL12", "CXCL14", "CCL2", "CXCL1", "CXCL2", "ICAM1", "VCAM1", "JUNB", "FOS"],
        "complement_matrix": ["C1R", "C1S", "C3", "CFD", "SERPING1", "COL1A1", "COL3A1", "DCN", "LUM"],
    },
    "Myeloid": {
        "inflammatory_monocyte": ["S100A8", "S100A9", "S100A12", "LYZ", "FCN1", "VCAN", "IL1B", "THBS1"],
        "c1qc_macrophage": ["C1QA", "C1QB", "C1QC", "APOE", "TREM2", "LIPA", "CD163", "MARCO", "MS4A7"],
        "spp1_macrophage": ["SPP1", "TREM2", "GPNMB", "LGALS3", "CTSD", "APOE", "LPL", "CD9"],
        "antigen_presentation": ["HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "CD74", "HLA-DQA1", "HLA-DQB1"],
        "interferon_myeloid": ["ISG15", "IFI6", "IFITM3", "IFIT1", "IFIT3", "MX1", "IFI27", "OAS1"],
    },
    "Endothelial": {
        "sinusoidal_identity": ["CLEC4G", "FCN3", "CRHBP", "LYVE1", "STAB1", "OIT3", "CLEC1B", "CLEC4M"],
        "angiogenic_endothelial": ["VWF", "PECAM1", "KDR", "FLT1", "ESAM", "PLVAP", "ENG", "EMCN"],
        "inflammatory_endothelial": ["VCAM1", "ICAM1", "SELE", "ACKR1", "CCL2", "CXCL2", "CXCL8", "JUNB"],
        "capillarized_matrix": ["VIM", "SPARC", "COL4A1", "COL4A2", "MGP", "PLVAP", "CD34", "AQP1"],
    },
    "TNKcell": {
        "cytotoxic": ["NKG7", "GNLY", "GZMB", "GZMA", "PRF1", "KLRD1", "KLRF1", "CCL5"],
        "inflammatory_tnk": ["CCL4", "CCL5", "IFNG", "TNF", "NFKBIA", "JUNB", "CD69", "DUSP2"],
        "exhaustion_checkpoint": ["PDCD1", "LAG3", "HAVCR2", "CTLA4", "TIGIT", "TOX", "ENTPD1"],
        "treg_like": ["FOXP3", "IL2RA", "CTLA4", "IKZF2", "TNFRSF18", "TIGIT"],
    },
    "Bcell": {
        "plasma": ["JCHAIN", "MZB1", "XBP1", "SSR4", "IGKC", "IGHG1", "IGHA1", "IGLC2"],
        "antigen_presentation": ["HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "CD74", "HLA-DQA1", "HLA-DQB1"],
        "naive_memory": ["MS4A1", "CD79A", "CD79B", "BANK1", "LTB", "TCL1A", "IGHM"],
    },
}


def unique_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        out.setdefault(row["NAME"], row)
    return list(out.values())


def tumor_means_by_donor(
    rows: list[dict[str, str]],
    scores: dict[str, np.ndarray],
    min_cells: int,
) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for idx, row in enumerate(rows):
        if row["indication"] != "Tumor":
            continue
        for score_name, values in scores.items():
            grouped[row["donor_id"]][score_name].append(float(values[idx]))

    out = {}
    for donor, by_score in grouped.items():
        n_cells = len(next(iter(by_score.values()))) if by_score else 0
        if n_cells < min_cells:
            continue
        out[donor] = {score_name: float(np.mean(values)) for score_name, values in by_score.items()}
        out[donor]["n_cells"] = float(n_cells)
    return out


def zscore(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / max(float(values.std()), 1e-8)


def add_coupling_axes(donor_table: list[dict]) -> None:
    ecm = np.array([row["stromal_ecm_matrix"] for row in donor_table], dtype=float)
    caf = np.array([row["stromal_caf_contractile"] for row in donor_table], dtype=float)
    secretory = np.array([row["hep_secretory_stress"] for row in donor_table], dtype=float)
    malignant = np.array([row["hep_malignant_like"] for row in donor_table], dtype=float)
    ecm_axis = zscore(ecm) - zscore(secretory)
    caf_axis = zscore(caf) - zscore(secretory)
    malignant_axis = zscore(ecm) + zscore(malignant)
    for idx, row in enumerate(donor_table):
        row["axis_ecm_high_hep_secretory_low"] = float(ecm_axis[idx])
        row["axis_caf_high_hep_secretory_low"] = float(caf_axis[idx])
        row["axis_ecm_high_hep_malignant_high"] = float(malignant_axis[idx])


def cell_type_composition(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row.get("organ__ontology_label") == "liver" and row.get("indication") == "Tumor":
            counts[row["donor_id"]][row["Cell_Type"]] += 1
    out = {}
    for donor, counter in counts.items():
        total = sum(counter.values())
        out[donor] = {f"composition_{cell_type}": value / max(total, 1) for cell_type, value in counter.items()}
    return out


def evaluate_driver(
    donor_table: list[dict],
    predictor_values: dict[str, float],
    predictor_name: str,
    target_name: str,
    alpha: float,
    permutations: int,
    rng: np.random.Generator,
) -> dict | None:
    donors = [row["donor_id"] for row in donor_table if row["donor_id"] in predictor_values]
    if len(donors) < 5:
        return None
    row_by_donor = {row["donor_id"]: row for row in donor_table}
    x = np.array([[predictor_values[donor]] for donor in donors], dtype=float)
    y = np.array([row_by_donor[donor][target_name] for donor in donors], dtype=float)
    observed = loo_prediction_test(x, y, alpha)
    null_values = []
    for _ in range(permutations):
        permuted_x = x[rng.permutation(len(x))]
        null_values.append(float(loo_prediction_test(permuted_x, y, alpha)["loo_delta_r2"]))
    null_array = np.array(null_values, dtype=float)
    return {
        "predictor": predictor_name,
        "target": target_name,
        "n_donors": len(donors),
        "loo_delta_r2": float(observed["loo_delta_r2"]),
        "loo_r2": float(observed["loo_r2"]),
        "loo_baseline_r2": float(observed["loo_baseline_r2"]),
        "mean_coeff": float(observed["mean_coeff"]),
        "positive_coeff_fraction": float(observed["positive_coeff_fraction"]),
        "pearson_r": safe_corr(x.ravel(), y),
        "empirical_p": empirical_p_value(float(observed["loo_delta_r2"]), null_array),
        "null_mean": float(null_array.mean()),
        "null_95pct": float(np.quantile(null_array, 0.95)),
        "donors": ",".join(donors),
    }


def evaluate_value_pair(
    donor_values: dict[str, dict[str, float]],
    predictor_name: str,
    target_name: str,
    alpha: float,
    permutations: int,
    rng: np.random.Generator,
) -> dict | None:
    donors = [
        donor
        for donor, values in donor_values.items()
        if predictor_name in values and target_name in values
    ]
    if len(donors) < 5:
        return None
    x = np.array([[donor_values[donor][predictor_name]] for donor in donors], dtype=float)
    y = np.array([donor_values[donor][target_name] for donor in donors], dtype=float)
    observed = loo_prediction_test(x, y, alpha)
    null_values = []
    for _ in range(permutations):
        permuted_x = x[rng.permutation(len(x))]
        null_values.append(float(loo_prediction_test(permuted_x, y, alpha)["loo_delta_r2"]))
    null_array = np.array(null_values, dtype=float)
    return {
        "predictor": predictor_name,
        "target": target_name,
        "n_donors": len(donors),
        "loo_delta_r2": float(observed["loo_delta_r2"]),
        "loo_r2": float(observed["loo_r2"]),
        "loo_baseline_r2": float(observed["loo_baseline_r2"]),
        "mean_coeff": float(observed["mean_coeff"]),
        "positive_coeff_fraction": float(observed["positive_coeff_fraction"]),
        "pearson_r": safe_corr(x.ravel(), y),
        "empirical_p": empirical_p_value(float(observed["loo_delta_r2"]), null_array),
        "null_mean": float(null_array.mean()),
        "null_95pct": float(np.quantile(null_array, 0.95)),
        "donors": ",".join(donors),
    }


def effect_label(coeff: float, delta_r2: float) -> str:
    if delta_r2 <= 0:
        return "no_heldout_gain"
    if coeff > 0:
        return "positive_predictive_relation"
    if coeff < 0:
        return "negative_predictive_relation"
    return "zero_coefficient"


def direction_call(forward_delta: float, reverse_delta: float, margin: float = 0.05) -> str:
    if forward_delta > margin and forward_delta > reverse_delta + margin:
        return "driver_to_axis_stronger"
    if reverse_delta > margin and reverse_delta > forward_delta + margin:
        return "axis_to_driver_stronger"
    if forward_delta > margin and reverse_delta > margin:
        return "bidirectional_or_shared_donor_state"
    return "weak_or_inconclusive"


def bidirectional_driver_records(
    donor_table: list[dict],
    driver_donor_scores: dict[str, dict[str, float]],
    predictor_names: list[str],
    targets: list[str],
    alpha: float,
    permutations: int,
    random_state: int,
) -> list[dict]:
    donor_values = {
        row["donor_id"]: {key: value for key, value in row.items() if key != "donor_id"}
        for row in donor_table
    }
    for donor, scores in driver_donor_scores.items():
        donor_values.setdefault(donor, {}).update(scores)

    rng = np.random.default_rng(random_state)
    records = []
    for predictor_name in predictor_names:
        for target in targets:
            forward = evaluate_value_pair(donor_values, predictor_name, target, alpha, permutations, rng)
            reverse = evaluate_value_pair(donor_values, target, predictor_name, alpha, permutations, rng)
            if forward is None or reverse is None:
                continue
            records.append(
                {
                    "driver": predictor_name,
                    "axis_or_target": target,
                    "forward_edge": f"{predictor_name}->{target}",
                    "reverse_edge": f"{target}->{predictor_name}",
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
                    "direction_call": direction_call(forward["loo_delta_r2"], reverse["loo_delta_r2"]),
                    "n_donors": forward["n_donors"],
                    "donors": forward["donors"],
                }
            )
    records.sort(
        key=lambda row: (
            row["axis_or_target"].startswith("axis_"),
            max(row["forward_loo_delta_r2"], row["reverse_loo_delta_r2"]),
            abs(row["forward_minus_reverse"]),
        ),
        reverse=True,
    )
    return records


def write_outputs(
    output_dir: Path,
    driver_records: list[dict],
    bidirectional_records: list[dict],
    donor_table: list[dict],
    driver_donor_scores: dict[str, dict[str, float]],
    signature_lines: list[str],
    summary_lines: list[str],
) -> None:
    driver_records.sort(key=lambda row: (row["loo_delta_r2"], abs(row["mean_coeff"])), reverse=True)
    result_lines = [
        "rank\tpredictor\ttarget\tn_donors\tloo_delta_r2\tloo_r2\tloo_baseline_r2\t"
        "mean_coeff\tpositive_coeff_fraction\tpearson_r\tempirical_p\tnull_mean\tnull_95pct\tdonors"
    ]
    for rank, row in enumerate(driver_records, start=1):
        result_lines.append(
            "\t".join(
                [
                    str(rank),
                    row["predictor"],
                    row["target"],
                    str(row["n_donors"]),
                    f"{row['loo_delta_r2']:.6f}",
                    f"{row['loo_r2']:.6f}",
                    f"{row['loo_baseline_r2']:.6f}",
                    f"{row['mean_coeff']:.6f}",
                    f"{row['positive_coeff_fraction']:.6f}",
                    f"{row['pearson_r']:.6f}",
                    f"{row['empirical_p']:.6f}",
                    f"{row['null_mean']:.6f}",
                    f"{row['null_95pct']:.6f}",
                    row["donors"],
                ]
            )
        )

    bidirectional_lines = [
        "rank\tdriver\taxis_or_target\tforward_edge\treverse_edge\t"
        "forward_loo_delta_r2\treverse_loo_delta_r2\tforward_minus_reverse\tforward_coeff\treverse_coeff\t"
        "forward_relation\treverse_relation\tforward_empirical_p\treverse_empirical_p\t"
        "forward_pearson_r\treverse_pearson_r\tdirection_call\tn_donors\tdonors"
    ]
    for rank, row in enumerate(bidirectional_records, start=1):
        bidirectional_lines.append(
            "\t".join(
                [
                    str(rank),
                    row["driver"],
                    row["axis_or_target"],
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
                    str(row["n_donors"]),
                    row["donors"],
                ]
            )
        )

    donor_keys = sorted({key for scores in driver_donor_scores.values() for key in scores})
    target_keys = [
        "stromal_ecm_matrix",
        "stromal_caf_contractile",
        "hep_secretory_stress",
        "hep_malignant_like",
        "axis_ecm_high_hep_secretory_low",
        "axis_caf_high_hep_secretory_low",
        "axis_ecm_high_hep_malignant_high",
    ]
    row_by_donor = {row["donor_id"]: row for row in donor_table}
    donor_lines = ["donor_id\t" + "\t".join(target_keys + donor_keys)]
    for donor in sorted(row_by_donor):
        row = row_by_donor[donor]
        values = [f"{row[key]:.6f}" for key in target_keys]
        values.extend(
            f"{driver_donor_scores.get(donor, {}).get(key, float('nan')):.6f}"
            for key in donor_keys
        )
        donor_lines.append(donor + "\t" + "\t".join(values))

    output_dir.mkdir(parents=True, exist_ok=True)
    save_text(output_dir / "driver_scan.tsv", "\n".join(result_lines) + "\n")
    save_text(output_dir / "bidirectional_driver_scan.tsv", "\n".join(bidirectional_lines) + "\n")
    save_text(output_dir / "driver_donor_table.tsv", "\n".join(donor_lines) + "\n")
    save_text(output_dir / "driver_signature_reference.tsv", "\n".join(signature_lines) + "\n")
    save_text(output_dir / "summary.tsv", "\n".join(summary_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan candidate drivers of SCP2154 Tumor stromal-hepatocyte coupling axes")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--barcodes", type=Path, required=True)
    parser.add_argument("--max-cells-per-donor-phenotype", type=int, default=120)
    parser.add_argument("--max-cells-per-phenotype", type=int, default=1600)
    parser.add_argument("--malignant-percentile", type=float, default=60.0)
    parser.add_argument("--immune-percentile", type=float, default=80.0)
    parser.add_argument("--min-target-stromal-cells", type=int, default=8)
    parser.add_argument("--min-target-hep-cells", type=int, default=8)
    parser.add_argument("--min-driver-cells", type=int, default=8)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("results/scp2154_coupling_driver_scan"))
    args = parser.parse_args()

    metadata = read_metadata(args.metadata)
    cell_types = list(DRIVER_SIGNATURES)
    rows_by_cell_type = {
        cell_type: sample_liver_rows(
            metadata,
            cell_type,
            args.max_cells_per_donor_phenotype,
            args.max_cells_per_phenotype,
            args.random_state + idx * 101,
        )
        for idx, cell_type in enumerate(["Hepatocyte", *cell_types])
    }
    sampled_rows = unique_rows([row for rows in rows_by_cell_type.values() for row in rows])
    print(f"Reading one shared matrix subset for {len(sampled_rows)} cells...", flush=True)
    counts = read_10x_mtx_subset(args.matrix, args.features, args.barcodes, [row["NAME"] for row in sampled_rows])

    matrix_by_cell_type = {}
    kept_rows_by_cell_type = {}
    for cell_type, rows in rows_by_cell_type.items():
        matrix, kept_rows = celltype_matrix(counts, rows)
        matrix_by_cell_type[cell_type] = log1p_library_normalize(matrix)
        kept_rows_by_cell_type[cell_type] = kept_rows

    stromal_scores, _ = signature_scores(matrix_by_cell_type["Stromal"], counts.gene_names, STROMAL_SIGNATURES)
    hep_scores, _ = signature_scores(matrix_by_cell_type["Hepatocyte"], counts.gene_names, HEPATOCYTE_SIGNATURES)
    hep_selected = tumor_hepatocyte_selection(
        kept_rows_by_cell_type["Hepatocyte"],
        hep_scores,
        args.malignant_percentile,
        args.immune_percentile,
        "donor",
    )
    donor_table = build_donor_table(
        kept_rows_by_cell_type["Stromal"],
        kept_rows_by_cell_type["Hepatocyte"],
        stromal_scores,
        hep_scores,
        hep_selected,
        args.min_target_stromal_cells,
        args.min_target_hep_cells,
    )
    add_coupling_axes(donor_table)
    target_donors = {row["donor_id"] for row in donor_table}

    driver_donor_scores: dict[str, dict[str, float]] = defaultdict(dict)
    signature_lines = ["cell_type\tsignature\tmatched_genes"]
    for cell_type in cell_types:
        scores, used = signature_scores(matrix_by_cell_type[cell_type], counts.gene_names, DRIVER_SIGNATURES[cell_type])
        for signature_name, genes in used.items():
            signature_lines.append(f"{cell_type}\t{signature_name}\t{','.join(genes)}")
        donor_scores = tumor_means_by_donor(kept_rows_by_cell_type[cell_type], scores, args.min_driver_cells)
        for donor, values in donor_scores.items():
            if donor not in target_donors:
                continue
            for signature_name, value in values.items():
                if signature_name == "n_cells":
                    driver_donor_scores[donor][f"{cell_type}_n_cells"] = value
                else:
                    driver_donor_scores[donor][f"{cell_type}_{signature_name}"] = value

    composition = cell_type_composition(metadata)
    for donor in target_donors:
        for key, value in composition.get(donor, {}).items():
            driver_donor_scores[donor][key] = value

    targets = [
        "axis_ecm_high_hep_secretory_low",
        "axis_caf_high_hep_secretory_low",
        "stromal_ecm_matrix",
        "stromal_caf_contractile",
        "hep_secretory_stress",
        "hep_malignant_like",
    ]
    predictor_names = sorted({key for scores in driver_donor_scores.values() for key in scores})
    rng = np.random.default_rng(args.random_state + 999)
    driver_records = []
    for predictor_name in predictor_names:
        predictor_values = {
            donor: scores[predictor_name]
            for donor, scores in driver_donor_scores.items()
            if predictor_name in scores
        }
        for target in targets:
            record = evaluate_driver(
                donor_table,
                predictor_values,
                predictor_name,
                target,
                args.alpha,
                args.permutations,
                rng,
            )
            if record is not None:
                driver_records.append(record)
    bidirectional_records = bidirectional_driver_records(
        donor_table,
        driver_donor_scores,
        predictor_names,
        targets,
        args.alpha,
        args.permutations,
        args.random_state + 1999,
    )

    best_by_target = {}
    for target in targets:
        target_records = [row for row in driver_records if row["target"] == target]
        if target_records:
            best_by_target[target] = max(target_records, key=lambda row: row["loo_delta_r2"])

    summary_lines = [
        "metric\tvalue",
        f"target_donors\t{len(donor_table)}",
        f"selected_tumor_hepatocytes\t{int(hep_selected.sum())}",
        f"n_driver_tests\t{len(driver_records)}",
        f"n_bidirectional_driver_tests\t{len(bidirectional_records)}",
    ]
    for target, best in best_by_target.items():
        summary_lines.extend(
            [
                f"best_{target}_predictor\t{best['predictor']}",
                f"best_{target}_loo_delta_r2\t{best['loo_delta_r2']:.6f}",
                f"best_{target}_coeff\t{best['mean_coeff']:.6f}",
                f"best_{target}_p\t{best['empirical_p']:.6f}",
            ]
        )

    write_outputs(
        args.output_dir,
        driver_records,
        bidirectional_records,
        donor_table,
        driver_donor_scores,
        signature_lines,
        summary_lines,
    )
    print("SCP2154 coupling driver scan", flush=True)
    print(f"Target donors: {len(donor_table)}", flush=True)
    print(f"Driver tests: {len(driver_records)}", flush=True)
    for target, best in best_by_target.items():
        print(
            f"  {target}: {best['predictor']} "
            f"delta_r2={best['loo_delta_r2']:+.4f}, coeff={best['mean_coeff']:+.4f}, p={best['empirical_p']:.4f}",
            flush=True,
        )
    print("Saved:", flush=True)
    print(f"  {args.output_dir / 'driver_scan.tsv'}", flush=True)
    print(f"  {args.output_dir / 'bidirectional_driver_scan.tsv'}", flush=True)
    print(f"  {args.output_dir / 'driver_donor_table.tsv'}", flush=True)
    print(f"  {args.output_dir / 'driver_signature_reference.tsv'}", flush=True)
    print(f"  {args.output_dir / 'summary.tsv'}", flush=True)


if __name__ == "__main__":
    main()
