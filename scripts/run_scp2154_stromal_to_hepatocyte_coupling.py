from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from run_scp2154_phenotype_baseline import read_metadata, split_donors, stratified_sample
from multicell_dynamics import (
    fit_module_representation,
    fit_population_dynamics,
    local_direction_from_pseudotime,
    orient_pseudotime_by_labels,
    pca_embedding,
    pseudotime_from_embedding,
    read_10x_mtx_subset,
    select_highly_variable_genes,
    top_genes_per_module,
    velocity_r2_score,
    velocity_sign_agreement,
)


PHENOTYPES = ["healthy", "Tumor"]


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sample_celltype_rows(
    rows: list[dict[str, str]],
    cell_type: str,
    max_cells_per_donor_phenotype: int,
    max_cells_per_phenotype: int,
    random_state: int,
) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if row["organ__ontology_label"] == "liver"
        and row["Cell_Type"] == cell_type
        and row["indication"] in set(PHENOTYPES)
    ]
    return stratified_sample(
        selected,
        phenotype_col="indication",
        donor_col="donor_id",
        max_cells_per_donor_phenotype=max_cells_per_donor_phenotype,
        max_cells_per_phenotype=max_cells_per_phenotype,
        random_state=random_state,
    )


def fit_celltype_representation(
    counts,
    rows: list[dict[str, str]],
    n_modules: int,
    hvg: int,
    random_state: int,
) -> dict:
    cell_to_index = {cell: idx for idx, cell in enumerate(counts.cell_barcodes)}
    idx = np.array([cell_to_index[row["NAME"]] for row in rows], dtype=int)
    matrix = counts.matrix[idx]
    labels = np.array([row["indication"] for row in rows], dtype=object)
    hvg_matrix, hvg_names = select_highly_variable_genes(matrix, counts.gene_names, top_k=hvg)
    rep = fit_module_representation(hvg_matrix, n_modules=n_modules, random_state=random_state, max_iter=700)
    embedding = pca_embedding(hvg_matrix, n_components=2)
    pseudotime = pseudotime_from_embedding(embedding, axis=0)
    pseudotime = orient_pseudotime_by_labels(pseudotime, labels, low_label="healthy", high_label="Tumor")
    local_velocity = local_direction_from_pseudotime(rep.module_activity, pseudotime, k_neighbors=20)
    return {
        "rows": rows,
        "labels": labels,
        "rep": rep,
        "embedding": embedding,
        "pseudotime": pseudotime,
        "local_velocity": local_velocity,
        "hvg_names": hvg_names,
        "top_genes": top_genes_per_module(rep.module_weights, hvg_names, top_k=12),
    }


def module_shift(module_activity: np.ndarray, labels: np.ndarray) -> list[tuple[int, float, float, float]]:
    out = []
    for module_idx in range(module_activity.shape[1]):
        tumor_mean = float(module_activity[labels == "Tumor", module_idx].mean())
        healthy_mean = float(module_activity[labels == "healthy", module_idx].mean())
        out.append((module_idx, tumor_mean, healthy_mean, tumor_mean - healthy_mean))
    return out


def patient_phenotype_module_scores(
    rows: list[dict[str, str]],
    module_activity: np.ndarray,
) -> dict[tuple[str, str], np.ndarray]:
    grouped: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[(row["donor_id"], row["indication"])].append(module_activity[idx])
    return {key: np.mean(np.vstack(items), axis=0) for key, items in grouped.items()}


def build_stromal_input(
    hep_rows: list[dict[str, str]],
    stromal_scores: dict[tuple[str, str], np.ndarray],
) -> np.ndarray:
    global_mean = np.mean(np.vstack(list(stromal_scores.values())), axis=0)
    out = np.zeros((len(hep_rows), global_mean.shape[0]), dtype=float)
    for idx, row in enumerate(hep_rows):
        out[idx] = stromal_scores.get((row["donor_id"], row["indication"]), global_mean)
    return out


def split_indices_by_donor(rows: list[dict[str, str]], test_fraction: float = 0.25, random_state: int = 7) -> tuple[np.ndarray, np.ndarray]:
    split = split_donors(rows, "donor_id", random_state=random_state)
    train = [idx for idx, row in enumerate(rows) if split[row["donor_id"]] != "test"]
    test = [idx for idx, row in enumerate(rows) if split[row["donor_id"]] == "test"]
    return np.array(train, dtype=int), np.array(test, dtype=int)


def per_module_r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    vals = []
    for idx in range(y_true.shape[1]):
        vals.append(velocity_r2_score(y_true[:, [idx]], y_pred[:, [idx]]))
    return np.array(vals, dtype=float)


def zscore_by_train(values: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    mean = values[train_idx].mean(axis=0, keepdims=True)
    std = values[train_idx].std(axis=0, keepdims=True)
    return (values - mean) / np.maximum(std, 1e-8)


def top_gene_names(top_genes: list[tuple[str, float]]) -> list[str]:
    return [gene for gene, _ in top_genes]


def tumor_relation(delta: float, threshold: float = 0.1) -> str:
    if delta > threshold:
        return "Tumor-up"
    if delta < -threshold:
        return "Tumor-down"
    return "near-neutral"


def annotate_program(cell_type: str, genes: list[str]) -> str:
    gene_set = {gene.upper() for gene in genes}
    if {"ACTA2", "TAGLN", "MYL9"} & gene_set and {"CALD1", "TPM2", "IGFBP7", "VIM"} & gene_set:
        return "myofibroblast_contractile_ecm"
    if {"COL1A1", "COL1A2", "COL3A1", "DCN", "LUM"} & gene_set:
        return "matrix_fibroblast_ecm"
    if {"SAA1", "SAA2", "HP", "ORM1", "SERPINA1", "FGA", "FGB"} & gene_set:
        return "acute_phase_inflammatory_secretory"
    if {"CYP3A5", "CYP2A7", "SLC22A7", "MLXIPL", "APOB", "FTCD"} & gene_set:
        return "mature_hepatocyte_metabolic_transport"
    if {"HLA-A", "HLA-B", "HLA-C", "SRGN", "IL32", "TMSB10", "TMSB4X"} & gene_set:
        return "immune_mhc_inflammatory_or_doublet"
    if {"IGKC", "JCHAIN", "IGHG1", "IGHA1"} & gene_set:
        return "b_cell_plasma_or_ambient_signal"
    if {"ALB", "APOA1", "APOA2", "APOC3", "TTR", "RBP4", "AMBP"} & gene_set:
        return "hepatocyte_secretory_metabolic"
    return f"{cell_type.lower()}_mixed_program"


def coupling_direction(coeff: float, hep_delta: float, delta_r2: float) -> str:
    if delta_r2 <= 0:
        return "no_heldout_gain"
    if coeff > 0 and hep_delta > 0:
        return "candidate_supports_tumor_up_module"
    if coeff < 0 and hep_delta < 0:
        return "candidate_supports_tumor_down_module"
    if coeff < 0 and hep_delta > 0:
        return "opposes_tumor_up_module"
    if coeff > 0 and hep_delta < 0:
        return "opposes_tumor_down_module"
    return "weak_direction"


def write_module_reference(
    output_dir: Path,
    cell_type: str,
    shifts: list[tuple[int, float, float, float]],
    top_genes: dict[int, list[tuple[str, float]]],
    selected_module: int | None = None,
) -> list[str]:
    lines = []
    for module_idx, tumor_mean, healthy_mean, delta in shifts:
        genes = top_gene_names(top_genes[module_idx])
        lines.append(
            "\t".join(
                [
                    cell_type,
                    f"m_{module_idx}",
                    f"{tumor_mean:.6f}",
                    f"{healthy_mean:.6f}",
                    f"{delta:.6f}",
                    tumor_relation(delta),
                    annotate_program(cell_type, genes),
                    "yes" if selected_module == module_idx else "no",
                    ",".join(genes),
                ]
            )
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="SCP2154 stromal-to-hepatocyte coupling prototype")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--barcodes", type=Path, required=True)
    parser.add_argument("--max-cells-per-donor-phenotype", type=int, default=120)
    parser.add_argument("--max-cells-per-phenotype", type=int, default=1500)
    parser.add_argument("--hvg", type=int, default=1000)
    parser.add_argument("--modules", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=7)
    args = parser.parse_args()

    metadata = read_metadata(args.metadata)
    stromal_rows = sample_celltype_rows(
        metadata,
        "Stromal",
        args.max_cells_per_donor_phenotype,
        args.max_cells_per_phenotype,
        args.random_state,
    )
    hep_rows = sample_celltype_rows(
        metadata,
        "Hepatocyte",
        args.max_cells_per_donor_phenotype,
        args.max_cells_per_phenotype,
        args.random_state + 11,
    )
    all_cells = [row["NAME"] for row in stromal_rows + hep_rows]
    counts = read_10x_mtx_subset(args.matrix, args.features, args.barcodes, all_cells)

    stromal = fit_celltype_representation(counts, stromal_rows, args.modules, args.hvg, args.random_state)
    hep = fit_celltype_representation(counts, hep_rows, args.modules, args.hvg, args.random_state + 21)
    stromal_shift = module_shift(stromal["rep"].module_activity, stromal["labels"])
    hep_shift = module_shift(hep["rep"].module_activity, hep["labels"])
    stromal_module = max(stromal_shift, key=lambda item: item[3])[0]
    stromal_scores = patient_phenotype_module_scores(stromal_rows, stromal["rep"].module_activity)
    raw_stromal_input = build_stromal_input(hep_rows, stromal_scores)
    train_idx, test_idx = split_indices_by_donor(hep_rows, random_state=args.random_state)
    stromal_input_all = zscore_by_train(raw_stromal_input, train_idx)
    selected_stromal_input = stromal_input_all[:, [stromal_module]]

    baseline = fit_population_dynamics(
        module_activity=hep["rep"].module_activity[train_idx],
        state_embedding=hep["embedding"][train_idx],
        module_velocity=hep["local_velocity"][train_idx],
        alpha=0.03,
    )
    selected_coupled = fit_population_dynamics(
        module_activity=hep["rep"].module_activity[train_idx],
        state_embedding=hep["embedding"][train_idx],
        module_velocity=hep["local_velocity"][train_idx],
        external_input=selected_stromal_input[train_idx],
        alpha=0.03,
    )
    all_coupled = fit_population_dynamics(
        module_activity=hep["rep"].module_activity[train_idx],
        state_embedding=hep["embedding"][train_idx],
        module_velocity=hep["local_velocity"][train_idx],
        external_input=stromal_input_all[train_idx],
        alpha=0.03,
    )
    pred_base = baseline.predict_velocity(hep["rep"].module_activity[test_idx], hep["embedding"][test_idx])
    pred_selected = selected_coupled.predict_velocity(
        hep["rep"].module_activity[test_idx],
        hep["embedding"][test_idx],
        external_input=selected_stromal_input[test_idx],
    )
    pred_all = all_coupled.predict_velocity(
        hep["rep"].module_activity[test_idx],
        hep["embedding"][test_idx],
        external_input=stromal_input_all[test_idx],
    )
    true = hep["local_velocity"][test_idx]
    base_r2 = per_module_r2(true, pred_base)
    selected_r2 = per_module_r2(true, pred_selected)
    all_r2 = per_module_r2(true, pred_all)
    selected_delta = selected_r2 - base_r2
    all_delta = all_r2 - base_r2
    stromal_feature_name = "e_0"
    selected_stromal_idx = selected_coupled.feature_names.index(stromal_feature_name)
    pair_records = []
    for source_module_idx in range(stromal_input_all.shape[1]):
        single_input = stromal_input_all[:, [source_module_idx]]
        single_coupled = fit_population_dynamics(
            module_activity=hep["rep"].module_activity[train_idx],
            state_embedding=hep["embedding"][train_idx],
            module_velocity=hep["local_velocity"][train_idx],
            external_input=single_input[train_idx],
            alpha=0.03,
        )
        pred_single = single_coupled.predict_velocity(
            hep["rep"].module_activity[test_idx],
            hep["embedding"][test_idx],
            external_input=single_input[test_idx],
        )
        single_r2 = per_module_r2(true, pred_single)
        single_delta = single_r2 - base_r2
        single_e_idx = single_coupled.feature_names.index("e_0")
        all_e_idx = all_coupled.feature_names.index(f"e_{source_module_idx}")
        stromal_delta = [item[3] for item in stromal_shift if item[0] == source_module_idx][0]
        stromal_genes = top_gene_names(stromal["top_genes"][source_module_idx])
        stromal_program = annotate_program("Stromal", stromal_genes)
        for target_module_idx in range(hep["rep"].module_activity.shape[1]):
            hep_delta = [item[3] for item in hep_shift if item[0] == target_module_idx][0]
            hep_genes = top_gene_names(hep["top_genes"][target_module_idx])
            coeff = float(single_coupled.coefficient_matrix[target_module_idx, single_e_idx])
            pair_records.append(
                {
                    "stromal_module": source_module_idx,
                    "stromal_delta": stromal_delta,
                    "stromal_program": stromal_program,
                    "stromal_top_genes": ",".join(stromal_genes),
                    "hep_module": target_module_idx,
                    "hep_delta": hep_delta,
                    "hep_program": annotate_program("Hepatocyte", hep_genes),
                    "hep_top_genes": ",".join(hep_genes),
                    "single_input_coeff": coeff,
                    "single_input_delta_r2": float(single_delta[target_module_idx]),
                    "all_input_coeff": float(all_coupled.coefficient_matrix[target_module_idx, all_e_idx]),
                    "all_input_delta_r2": float(all_delta[target_module_idx]),
                    "direction": coupling_direction(coeff, hep_delta, float(single_delta[target_module_idx])),
                }
            )
    pair_records.sort(
        key=lambda item: (
            item["direction"] == "candidate_supports_tumor_up_module",
            item["single_input_delta_r2"],
            abs(item["single_input_coeff"]),
        ),
        reverse=True,
    )

    output_dir = Path("results/scp2154_stromal_to_hepatocyte")
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "hep_module\tbaseline_r2\tselected_stromal_r2\tselected_delta_r2\tall_stromal_r2\t"
        "all_delta_r2\tselected_stromal_coeff\thep_tumor_delta\thep_relation\thep_program\thep_top_genes"
    ]
    for module_idx in range(hep["rep"].module_activity.shape[1]):
        coeff = float(selected_coupled.coefficient_matrix[module_idx, selected_stromal_idx])
        hep_delta = [item[3] for item in hep_shift if item[0] == module_idx][0]
        genes = top_gene_names(hep["top_genes"][module_idx])
        lines.append(
            f"m_{module_idx}\t{base_r2[module_idx]:.6f}\t{selected_r2[module_idx]:.6f}\t"
            f"{selected_delta[module_idx]:.6f}\t{all_r2[module_idx]:.6f}\t"
            f"{all_delta[module_idx]:.6f}\t{coeff:.6f}\t{hep_delta:.6f}\t"
            f"{tumor_relation(hep_delta)}\t{annotate_program('Hepatocyte', genes)}\t{','.join(genes)}"
        )
    save_text(output_dir / "coupling_summary.tsv", "\n".join(lines) + "\n")
    module_lines = [
        "cell_type\tmodule\ttumor_mean\thealthy_mean\tdelta_tumor_minus_healthy\t"
        "tumor_relation\tputative_program\tselected_sender_module\ttop_genes"
    ]
    module_lines.extend(
        write_module_reference(output_dir, "Stromal", stromal_shift, stromal["top_genes"], selected_module=stromal_module)
    )
    module_lines.extend(write_module_reference(output_dir, "Hepatocyte", hep_shift, hep["top_genes"]))
    save_text(output_dir / "module_reference.tsv", "\n".join(module_lines) + "\n")
    pair_lines = [
        "rank\tstromal_module\tstromal_tumor_delta\tstromal_relation\tstromal_program\thep_module\t"
        "hep_tumor_delta\thep_relation\thep_program\tsingle_input_coeff\tsingle_input_delta_r2\t"
        "all_input_coeff\tall_input_delta_r2\tdirection\tstromal_top_genes\thep_top_genes"
    ]
    for rank, item in enumerate(pair_records, start=1):
        pair_lines.append(
            "\t".join(
                [
                    str(rank),
                    f"m_{item['stromal_module']}",
                    f"{item['stromal_delta']:.6f}",
                    tumor_relation(item["stromal_delta"]),
                    item["stromal_program"],
                    f"m_{item['hep_module']}",
                    f"{item['hep_delta']:.6f}",
                    tumor_relation(item["hep_delta"]),
                    item["hep_program"],
                    f"{item['single_input_coeff']:.6f}",
                    f"{item['single_input_delta_r2']:.6f}",
                    f"{item['all_input_coeff']:.6f}",
                    f"{item['all_input_delta_r2']:.6f}",
                    item["direction"],
                    item["stromal_top_genes"],
                    item["hep_top_genes"],
                ]
            )
        )
    save_text(output_dir / "coupling_pairs.tsv", "\n".join(pair_lines) + "\n")

    print("SCP2154 stromal-to-hepatocyte coupling prototype")
    print("Stromal cells:", len(stromal_rows), dict(Counter(stromal["labels"])))
    print("Hepatocyte cells:", len(hep_rows), dict(Counter(hep["labels"])))
    print(f"Selected stromal tumor-associated module: m_{stromal_module}")
    print("  program:", annotate_program("Stromal", top_gene_names(stromal["top_genes"][stromal_module])))
    print("  top genes:", ", ".join(top_gene_names(stromal["top_genes"][stromal_module])))
    print("Hepatocyte modules:")
    for module_idx, tumor_mean, healthy_mean, hep_delta in hep_shift:
        genes = top_gene_names(hep["top_genes"][module_idx])
        print(f"  m_{module_idx}: tumor_delta={hep_delta:+.4f}, {annotate_program('Hepatocyte', genes)}")
        print(f"    top genes: {', '.join(genes)}")
    print("Selected stromal module coupling summary:")
    for module_idx in range(hep["rep"].module_activity.shape[1]):
        coeff = float(selected_coupled.coefficient_matrix[module_idx, selected_stromal_idx])
        print(
            f"  stromal_m{stromal_module} -> hep_m{module_idx}: "
            f"coeff={coeff:+.4f}, delta_r2={selected_delta[module_idx]:+.4f}, "
            f"baseline_r2={base_r2[module_idx]:.4f}, coupled_r2={selected_r2[module_idx]:.4f}"
        )
    print("Top candidate stromal-to-hepatocyte pairs:")
    for item in pair_records[:6]:
        print(
            f"  stromal_m{item['stromal_module']} -> hep_m{item['hep_module']}: "
            f"delta_r2={item['single_input_delta_r2']:+.4f}, coeff={item['single_input_coeff']:+.4f}, "
            f"{item['direction']}"
        )
    print("Overall baseline R2:", round(velocity_r2_score(true, pred_base), 4))
    print("Overall selected-stromal coupled R2:", round(velocity_r2_score(true, pred_selected), 4))
    print("Overall all-stromal coupled R2:", round(velocity_r2_score(true, pred_all), 4))
    print("Overall baseline sign:", round(velocity_sign_agreement(true, pred_base), 4))
    print("Overall selected-stromal coupled sign:", round(velocity_sign_agreement(true, pred_selected), 4))
    print("Overall all-stromal coupled sign:", round(velocity_sign_agreement(true, pred_all), 4))
    print("Saved:")
    print(f"  {output_dir / 'coupling_summary.tsv'}")
    print(f"  {output_dir / 'module_reference.tsv'}")
    print(f"  {output_dir / 'coupling_pairs.tsv'}")


if __name__ == "__main__":
    main()
