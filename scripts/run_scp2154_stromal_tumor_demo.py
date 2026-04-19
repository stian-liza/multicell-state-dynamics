from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from run_scp2154_phenotype_baseline import (
    accuracy,
    confusion_matrix,
    read_metadata,
    split_donors,
    stratified_sample,
    train_softmax_regression,
)
from multicell_dynamics import (
    fit_module_representation,
    read_10x_mtx_subset,
    select_highly_variable_genes,
    top_genes_per_module,
)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def module_shift_summary(module_activity: np.ndarray, labels: np.ndarray) -> list[tuple[int, float, float, float]]:
    out = []
    for module_idx in range(module_activity.shape[1]):
        tumor_mean = float(module_activity[labels == "Tumor", module_idx].mean())
        healthy_mean = float(module_activity[labels == "healthy", module_idx].mean())
        out.append((module_idx, tumor_mean, healthy_mean, tumor_mean - healthy_mean))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="SCP2154 stromal healthy-vs-tumor baseline and module demo")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--barcodes", type=Path, required=True)
    parser.add_argument("--max-cells-per-donor-phenotype", type=int, default=120)
    parser.add_argument("--max-cells-per-phenotype", type=int, default=1500)
    parser.add_argument("--hvg", type=int, default=1000)
    parser.add_argument("--modules", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--random-state", type=int, default=7)
    args = parser.parse_args()

    rows = read_metadata(args.metadata)
    rows = [
        row
        for row in rows
        if row["organ__ontology_label"] == "liver"
        and row["Cell_Type"] == "Stromal"
        and row["indication"] in {"healthy", "Tumor"}
    ]
    rows = stratified_sample(
        rows,
        phenotype_col="indication",
        donor_col="donor_id",
        max_cells_per_donor_phenotype=args.max_cells_per_donor_phenotype,
        max_cells_per_phenotype=args.max_cells_per_phenotype,
        random_state=args.random_state,
    )
    cells = [row["NAME"] for row in rows]
    counts = read_10x_mtx_subset(args.matrix, args.features, args.barcodes, cells)
    index_by_cell = {cell: idx for idx, cell in enumerate(counts.cell_barcodes)}
    rows = [row for row in rows if row["NAME"] in index_by_cell]
    order = np.array([index_by_cell[row["NAME"]] for row in rows], dtype=int)
    matrix = counts.matrix[order]
    labels_text = np.array([row["indication"] for row in rows], dtype=object)
    labels = np.array([0 if value == "healthy" else 1 for value in labels_text], dtype=int)

    hvg_matrix, hvg_names = select_highly_variable_genes(matrix, counts.gene_names, top_k=args.hvg)
    split = split_donors(rows, "donor_id", random_state=args.random_state)
    splits = np.array([split[row["donor_id"]] for row in rows], dtype=object)
    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"
    mean = hvg_matrix[train_mask].mean(axis=0, keepdims=True)
    std = np.maximum(hvg_matrix[train_mask].std(axis=0, keepdims=True), 1e-6)
    x = (hvg_matrix - mean) / std
    weights, bias = train_softmax_regression(x[train_mask], labels[train_mask], n_classes=2, epochs=args.epochs)

    rep = fit_module_representation(hvg_matrix, n_modules=args.modules, random_state=args.random_state, max_iter=700)
    module_genes = top_genes_per_module(rep.module_weights, hvg_names, top_k=12)
    module_shift = module_shift_summary(rep.module_activity, labels_text)

    output_dir = Path("results/scp2154_stromal_tumor")
    output_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(x[test_mask], labels[test_mask], weights, bias, 2)
    save_text(
        output_dir / "confusion_matrix.tsv",
        "true\\pred\thealthy\tTumor\n"
        + "\n".join(["healthy\t" + "\t".join(str(v) for v in cm[0]), "Tumor\t" + "\t".join(str(v) for v in cm[1])])
        + "\n",
    )
    lines = ["module\ttumor_mean\thealthy_mean\tdelta_tumor_minus_healthy\ttop_genes"]
    for module_idx, tumor_mean, healthy_mean, delta in module_shift:
        genes = ",".join(gene for gene, _ in module_genes[module_idx])
        lines.append(f"m_{module_idx}\t{tumor_mean:.6f}\t{healthy_mean:.6f}\t{delta:.6f}\t{genes}")
    save_text(output_dir / "module_summary.tsv", "\n".join(lines) + "\n")

    print("SCP2154 Stromal healthy-vs-Tumor demo")
    print("Cells:", len(rows), dict(Counter(labels_text)))
    print("Donors:", len(set(row["donor_id"] for row in rows)), dict(Counter(split[row["donor_id"]] for row in rows)))
    print("HVGs:", len(hvg_names))
    print("Train accuracy:", round(accuracy(x[train_mask], labels[train_mask], weights, bias), 4))
    print("Validation accuracy:", round(accuracy(x[val_mask], labels[val_mask], weights, bias), 4))
    print("Test accuracy:", round(accuracy(x[test_mask], labels[test_mask], weights, bias), 4))
    print("Module shifts:")
    for module_idx, tumor_mean, healthy_mean, delta in module_shift:
        print(f"  m_{module_idx}: tumor_mean={tumor_mean:.4f}, healthy_mean={healthy_mean:.4f}, delta={delta:+.4f}")
        print(f"    top genes: {', '.join(gene for gene, _ in module_genes[module_idx])}")
    print("Saved:")
    print(f"  {output_dir / 'confusion_matrix.tsv'}")
    print(f"  {output_dir / 'module_summary.tsv'}")


if __name__ == "__main__":
    main()
