from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from run_scp2154_phenotype_baseline import (
    accuracy,
    confusion_matrix,
    infer_column,
    read_metadata,
    select_highly_variable_genes,
    split_donors,
    stratified_sample,
    train_softmax_regression,
)

from multicell_dynamics import read_10x_mtx_subset


DEFAULT_CELL_TYPES = ["Myeloid", "Hepatocyte", "Endothelial", "Stromal", "TNKcell", "Bcell"]


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sample_rows_for_celltype(
    all_rows: list[dict[str, str]],
    cell_type: str,
    columns: list[str],
    args,
) -> list[dict[str, str]]:
    donor_col = args.donor_col or infer_column(columns, ["donor", "donor_id", "patient", "sample"], required=True)
    phenotype_col = args.phenotype_col or infer_column(
        columns,
        ["phenotype", "disease", "diagnosis", "condition", "disease_status", "sample_type", "indication"],
        required=True,
    )
    tissue_col = args.tissue_col or infer_column(columns, ["tissue", "organ", "source_tissue", "anatomical_region", "organ__ontology_label"], required=False)
    celltype_col = args.celltype_col or infer_column(columns, ["Cell_Type", "cell_type", "celltype", "cell_type__ontology_label"], required=True)

    rows = all_rows
    if tissue_col:
        rows = [row for row in rows if row.get(tissue_col, "").lower() == args.tissue_value.lower()]
    phenotypes = [item.strip() for item in args.phenotypes.split(",") if item.strip()]
    rows = [row for row in rows if row.get(celltype_col) == cell_type and row.get(phenotype_col) in set(phenotypes)]
    return stratified_sample(
        rows,
        phenotype_col=phenotype_col,
        donor_col=donor_col,
        max_cells_per_donor_phenotype=args.max_cells_per_donor_phenotype,
        max_cells_per_phenotype=args.max_cells_per_phenotype,
        random_state=args.random_state,
    )


def run_one_celltype(
    cell_type: str,
    rows: list[dict[str, str]],
    matrix: np.ndarray,
    gene_names: np.ndarray,
    columns: list[str],
    args,
) -> dict | None:
    donor_col = args.donor_col or infer_column(columns, ["donor", "donor_id", "patient", "sample"], required=True)
    phenotype_col = args.phenotype_col or infer_column(
        columns,
        ["phenotype", "disease", "diagnosis", "condition", "disease_status", "sample_type", "indication"],
        required=True,
    )
    phenotypes = [item.strip() for item in args.phenotypes.split(",") if item.strip()]
    if len(rows) < args.min_cells:
        return None

    donor_split = split_donors(rows, donor_col, random_state=args.random_state)
    labels = np.array([phenotypes.index(row[phenotype_col]) for row in rows], dtype=int)
    splits = np.array([donor_split[row[donor_col]] for row in rows], dtype=object)

    hvg_matrix, _ = select_highly_variable_genes(matrix, gene_names, top_k=args.hvg)
    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        return None

    mean = hvg_matrix[train_mask].mean(axis=0, keepdims=True)
    std = np.maximum(hvg_matrix[train_mask].std(axis=0, keepdims=True), 1e-6)
    x = (hvg_matrix - mean) / std
    weights, bias = train_softmax_regression(
        x[train_mask],
        labels[train_mask],
        n_classes=len(phenotypes),
        epochs=args.epochs,
    )
    cm = confusion_matrix(x[test_mask], labels[test_mask], weights, bias, len(phenotypes))
    return {
        "cell_type": cell_type,
        "n_cells": len(rows),
        "n_donors": len(set(row[donor_col] for row in rows)),
        "phenotype_counts": Counter(row[phenotype_col] for row in rows),
        "train_accuracy": accuracy(x[train_mask], labels[train_mask], weights, bias),
        "val_accuracy": accuracy(x[val_mask], labels[val_mask], weights, bias),
        "test_accuracy": accuracy(x[test_mask], labels[test_mask], weights, bias),
        "confusion_matrix": cm,
        "phenotypes": phenotypes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SCP2154 phenotype baseline per cell type")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--barcodes", type=Path, required=True)
    parser.add_argument("--cell-col", default="NAME")
    parser.add_argument("--donor-col", default="donor_id")
    parser.add_argument("--phenotype-col", default="indication")
    parser.add_argument("--celltype-col", default="Cell_Type")
    parser.add_argument("--tissue-col", default="organ__ontology_label")
    parser.add_argument("--tissue-value", default="liver")
    parser.add_argument("--phenotypes", default="healthy,low_steatosis,Tumor")
    parser.add_argument("--cell-types", default=",".join(DEFAULT_CELL_TYPES))
    parser.add_argument("--max-cells-per-donor-phenotype", type=int, default=120)
    parser.add_argument("--max-cells-per-phenotype", type=int, default=2000)
    parser.add_argument("--min-cells", type=int, default=300)
    parser.add_argument("--hvg", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--random-state", type=int, default=7)
    args = parser.parse_args()

    all_rows = read_metadata(args.metadata)
    columns = list(all_rows[0])
    cell_col = args.cell_col or infer_column(columns, ["cell", "cell_id", "cell_id__", "barcode", "NAME"])
    cell_types = [item.strip() for item in args.cell_types.split(",") if item.strip()]
    output_dir = Path("results/scp2154_celltype_baselines")
    output_dir.mkdir(parents=True, exist_ok=True)

    sampled_by_type = {
        cell_type: sample_rows_for_celltype(all_rows, cell_type, columns, args)
        for cell_type in cell_types
    }
    all_sampled_rows = [row for rows in sampled_by_type.values() for row in rows]
    all_cells = [row[cell_col] for row in all_sampled_rows]
    print(f"Reading one shared matrix subset for {len(all_cells)} sampled cells...")
    counts = read_10x_mtx_subset(args.matrix, args.features, args.barcodes, all_cells)
    cell_to_index = {cell: idx for idx, cell in enumerate(counts.cell_barcodes)}

    summary_lines = [
        "cell_type\tn_cells\tn_donors\ttrain_accuracy\tval_accuracy\ttest_accuracy\tphenotype_counts"
    ]
    print("SCP2154 per-cell-type phenotype baselines")
    for cell_type in cell_types:
        rows = [row for row in sampled_by_type[cell_type] if row[cell_col] in cell_to_index]
        idx = np.array([cell_to_index[row[cell_col]] for row in rows], dtype=int)
        matrix = counts.matrix[idx]
        result = run_one_celltype(cell_type, rows, matrix, counts.gene_names, columns, args)
        if result is None:
            print(f"  {cell_type}: skipped")
            continue
        phenotype_counts = ";".join(f"{k}:{v}" for k, v in sorted(result["phenotype_counts"].items()))
        summary_lines.append(
            f"{result['cell_type']}\t{result['n_cells']}\t{result['n_donors']}\t"
            f"{result['train_accuracy']:.6f}\t{result['val_accuracy']:.6f}\t{result['test_accuracy']:.6f}\t"
            f"{phenotype_counts}"
        )
        lines = ["true\\pred\t" + "\t".join(result["phenotypes"])]
        for idx, phenotype in enumerate(result["phenotypes"]):
            lines.append(phenotype + "\t" + "\t".join(str(v) for v in result["confusion_matrix"][idx]))
        save_text(output_dir / f"{cell_type}_confusion_matrix.tsv", "\n".join(lines) + "\n")
        print(
            f"  {cell_type}: cells={result['n_cells']}, donors={result['n_donors']}, "
            f"test_acc={result['test_accuracy']:.4f}"
        )

    summary_path = output_dir / "celltype_summary.tsv"
    save_text(summary_path, "\n".join(summary_lines) + "\n")
    print("Saved:")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
