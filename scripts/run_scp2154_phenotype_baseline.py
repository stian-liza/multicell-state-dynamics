from __future__ import annotations

import argparse
import csv
import gzip
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from multicell_dynamics import (
    log1p_library_normalize,
    read_10x_mtx_subset,
    read_dense_gene_cell_table_subset,
    select_highly_variable_genes,
)


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open("r", encoding="utf-8")


def read_metadata(path: Path) -> list[dict[str, str]]:
    with open_text(path) as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        first = next(reader)
        rows = []
        if first and first[0] == "TYPE":
            pass
        else:
            rows.append({header[idx]: first[idx] for idx in range(min(len(header), len(first)))})
        for raw in reader:
            rows.append({header[idx]: raw[idx] for idx in range(min(len(header), len(raw)))})
    return rows


def infer_column(columns: list[str], candidates: list[str], required: bool = True) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    for column in columns:
        normalized = column.lower().replace("_", "").replace(".", "")
        for candidate in candidates:
            if candidate.lower().replace("_", "").replace(".", "") == normalized:
                return column
    if required:
        raise ValueError(f"Could not infer column from candidates: {candidates}")
    return None


def choose_phenotypes(rows: list[dict[str, str]], phenotype_col: str, requested: str | None, n_phenotypes: int) -> list[str]:
    if requested:
        return [item.strip() for item in requested.split(",") if item.strip()]
    counts = Counter(row[phenotype_col] for row in rows if row.get(phenotype_col))
    return [phenotype for phenotype, _ in counts.most_common(n_phenotypes)]


def stratified_sample(
    rows: list[dict[str, str]],
    phenotype_col: str,
    donor_col: str,
    max_cells_per_donor_phenotype: int,
    max_cells_per_phenotype: int,
    random_state: int,
) -> list[dict[str, str]]:
    rng = np.random.default_rng(random_state)
    by_pair: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_pair[(row[phenotype_col], row[donor_col])].append(row)

    sampled: list[dict[str, str]] = []
    per_phenotype_count: Counter[str] = Counter()
    for phenotype, donor in sorted(by_pair):
        group = by_pair[(phenotype, donor)]
        if per_phenotype_count[phenotype] >= max_cells_per_phenotype:
            continue
        limit = min(
            max_cells_per_donor_phenotype,
            max_cells_per_phenotype - per_phenotype_count[phenotype],
            len(group),
        )
        if limit <= 0:
            continue
        if len(group) <= limit:
            selected = group
        else:
            idx = rng.choice(len(group), size=limit, replace=False)
            selected = [group[i] for i in sorted(idx)]
        sampled.extend(selected)
        per_phenotype_count[phenotype] += len(selected)
    return sampled


def split_donors(rows: list[dict[str, str]], donor_col: str, random_state: int) -> dict[str, str]:
    rng = np.random.default_rng(random_state)
    donors = sorted({row[donor_col] for row in rows})
    rng.shuffle(donors)
    n = len(donors)
    n_train = max(1, int(round(0.6 * n)))
    n_val = max(1, int(round(0.2 * n))) if n >= 5 else max(0, int(round(0.2 * n)))
    split = {}
    for idx, donor in enumerate(donors):
        if idx < n_train:
            split[donor] = "train"
        elif idx < n_train + n_val:
            split[donor] = "val"
        else:
            split[donor] = "test"
    return split


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def train_softmax_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    learning_rate: float = 0.2,
    l2: float = 1e-3,
    epochs: int = 800,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.zeros((x_train.shape[1], n_classes), dtype=float)
    bias = np.zeros(n_classes, dtype=float)
    y_onehot = np.eye(n_classes)[y_train]
    for _ in range(epochs):
        probs = softmax(x_train @ weights + bias)
        error = probs - y_onehot
        grad_w = x_train.T @ error / len(x_train) + l2 * weights
        grad_b = error.mean(axis=0)
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b
    return weights, bias


def accuracy(x: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> float:
    pred = np.argmax(x @ weights + bias, axis=1)
    return float(np.mean(pred == y)) if len(y) else float("nan")


def confusion_matrix(x: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: np.ndarray, n_classes: int) -> np.ndarray:
    pred = np.argmax(x @ weights + bias, axis=1)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true, guess in zip(y, pred):
        matrix[true, guess] += 1
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="SCP2154 liver phenotype classification baseline")
    parser.add_argument("--metadata", type=Path, default=Path("data/raw/scp2154/metadata.tsv.gz"))
    parser.add_argument("--matrix", type=Path, default=Path("data/raw/scp2154/counts.tsv.gz"))
    parser.add_argument("--features", type=Path, default=None)
    parser.add_argument("--barcodes", type=Path, default=None)
    parser.add_argument("--matrix-format", choices=["dense", "10x"], default="dense")
    parser.add_argument("--cell-col", default=None)
    parser.add_argument("--donor-col", default=None)
    parser.add_argument("--phenotype-col", default=None)
    parser.add_argument("--tissue-col", default=None)
    parser.add_argument("--tissue-value", default="liver")
    parser.add_argument("--phenotypes", default=None, help="Comma-separated phenotypes. Default: top phenotypes.")
    parser.add_argument("--n-phenotypes", type=int, default=3)
    parser.add_argument("--max-cells-per-donor-phenotype", type=int, default=250)
    parser.add_argument("--max-cells-per-phenotype", type=int, default=4000)
    parser.add_argument("--hvg", type=int, default=1500)
    parser.add_argument("--random-state", type=int, default=7)
    args = parser.parse_args()

    rows = read_metadata(args.metadata)
    if not rows:
        raise SystemExit("Metadata is empty.")
    columns = list(rows[0])
    cell_col = args.cell_col or infer_column(columns, ["cell", "cell_id", "cell_id__", "barcode", "NAME"])
    donor_col = args.donor_col or infer_column(columns, ["donor", "donor_id", "patient", "sample"], required=True)
    phenotype_col = args.phenotype_col or infer_column(
        columns,
        ["phenotype", "disease", "diagnosis", "condition", "disease_status", "sample_type"],
        required=True,
    )
    tissue_col = args.tissue_col or infer_column(columns, ["tissue", "organ", "source_tissue", "anatomical_region"], required=False)

    if tissue_col:
        rows = [row for row in rows if row.get(tissue_col, "").lower() == args.tissue_value.lower()]
    phenotypes = choose_phenotypes(rows, phenotype_col, args.phenotypes, args.n_phenotypes)
    rows = [row for row in rows if row.get(phenotype_col) in set(phenotypes)]
    rows = stratified_sample(
        rows,
        phenotype_col=phenotype_col,
        donor_col=donor_col,
        max_cells_per_donor_phenotype=args.max_cells_per_donor_phenotype,
        max_cells_per_phenotype=args.max_cells_per_phenotype,
        random_state=args.random_state,
    )
    donor_split = split_donors(rows, donor_col, random_state=args.random_state)
    cells = [row[cell_col] for row in rows]

    if args.matrix_format == "10x":
        if args.features is None or args.barcodes is None:
            raise SystemExit("--features and --barcodes are required when --matrix-format 10x is used.")
        counts = read_10x_mtx_subset(args.matrix, args.features, args.barcodes, cells)
    else:
        counts = read_dense_gene_cell_table_subset(args.matrix, cells)
    index_by_cell = {cell: idx for idx, cell in enumerate(counts.cell_barcodes)}
    keep_rows = [row for row in rows if row[cell_col] in index_by_cell]
    order = np.array([index_by_cell[row[cell_col]] for row in keep_rows], dtype=int)
    matrix = counts.matrix[order]
    labels = np.array([phenotypes.index(row[phenotype_col]) for row in keep_rows], dtype=int)
    splits = np.array([donor_split[row[donor_col]] for row in keep_rows], dtype=object)

    normalized = matrix if args.matrix_format == "10x" else log1p_library_normalize(matrix)
    hvg_matrix, hvg_names = select_highly_variable_genes(normalized, counts.gene_names, top_k=args.hvg)
    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"
    mean = hvg_matrix[train_mask].mean(axis=0, keepdims=True)
    std = np.maximum(hvg_matrix[train_mask].std(axis=0, keepdims=True), 1e-6)
    x = (hvg_matrix - mean) / std
    weights, bias = train_softmax_regression(x[train_mask], labels[train_mask], n_classes=len(phenotypes))

    output_dir = Path("results/scp2154_phenotype_baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(x[test_mask], labels[test_mask], weights, bias, len(phenotypes))
    with (output_dir / "confusion_matrix.tsv").open("w", encoding="utf-8") as handle:
        handle.write("true\\pred\t" + "\t".join(phenotypes) + "\n")
        for idx, phenotype in enumerate(phenotypes):
            handle.write(phenotype + "\t" + "\t".join(str(v) for v in cm[idx]) + "\n")

    print("SCP2154 liver phenotype classification baseline")
    print("Inferred columns:", {"cell": cell_col, "donor": donor_col, "phenotype": phenotype_col, "tissue": tissue_col})
    print("Phenotypes:", phenotypes)
    print("Cells after sampling:", len(keep_rows), dict(Counter(row[phenotype_col] for row in keep_rows)))
    print("Donors:", len(set(row[donor_col] for row in keep_rows)), dict(Counter(donor_split[row[donor_col]] for row in keep_rows)))
    print("HVGs:", len(hvg_names))
    print("Train accuracy:", round(accuracy(x[train_mask], labels[train_mask], weights, bias), 4))
    print("Validation accuracy:", round(accuracy(x[val_mask], labels[val_mask], weights, bias), 4))
    print("Test accuracy:", round(accuracy(x[test_mask], labels[test_mask], weights, bias), 4))
    print("Saved:")
    print(f"  {output_dir / 'confusion_matrix.tsv'}")


if __name__ == "__main__":
    main()
