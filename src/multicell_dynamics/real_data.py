from __future__ import annotations

import csv
import gzip
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CountMatrix:
    matrix: np.ndarray
    gene_names: np.ndarray
    cell_barcodes: np.ndarray


def _open_maybe_gzip(path: str | Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open("r")


def read_metadata_table(path: str | Path, delimiter: str = "\t", max_rows: int | None = None) -> list[dict[str, str]]:
    """Read tabular metadata into a list of row dictionaries."""
    rows: list[dict[str, str]] = []
    with _open_maybe_gzip(path) as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        header = next(reader)
        for idx, raw_row in enumerate(reader):
            row = _align_metadata_row(header, raw_row)
            rows.append(row)
            if max_rows is not None and idx + 1 >= max_rows:
                break
    return rows


def metadata_columns(path: str | Path, delimiter: str = "\t") -> list[str]:
    with _open_maybe_gzip(path) as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        header = next(reader)
    return [str(column) for column in header]


def _align_metadata_row(header: list[str], raw_row: list[str]) -> dict[str, str]:
    if len(raw_row) == len(header):
        aligned = raw_row
        extra = ""
    elif len(raw_row) == len(header) + 1:
        extra = raw_row[0]
        aligned = raw_row[1:]
    else:
        extra = ""
        if len(raw_row) < len(header):
            aligned = raw_row + [""] * (len(header) - len(raw_row))
        else:
            aligned = raw_row[: len(header)]
    row = {header[idx]: aligned[idx] for idx in range(len(header))}
    if extra:
        row["_row_id"] = extra
    return row


def read_10x_triplet_from_zip(path: str | Path) -> CountMatrix:
    """Read a zipped 10x-style matrix/features/barcodes bundle into memory."""
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        matrix_name = _pick_name(names, suffixes=("matrix.mtx", "matrix.mtx.gz"))
        barcode_name = _pick_name(names, suffixes=("barcodes.tsv", "barcodes.tsv.gz"))
        feature_name = _pick_name(names, suffixes=("features.tsv", "features.tsv.gz", "genes.tsv", "genes.tsv.gz"))
        matrix = _read_mtx_from_zip(archive, matrix_name)
        barcodes = _read_text_lines_from_zip(archive, barcode_name)
        features = _read_feature_names_from_zip(archive, feature_name)
    return CountMatrix(matrix=matrix, gene_names=features, cell_barcodes=barcodes)


def _pick_name(names: list[str], suffixes: tuple[str, ...]) -> str:
    for suffix in suffixes:
        for name in names:
            if name.endswith(suffix):
                return name
    raise FileNotFoundError(f"Could not find file with suffixes {suffixes!r}")


def _read_text_lines_from_zip(archive: zipfile.ZipFile, member_name: str) -> np.ndarray:
    raw = archive.read(member_name)
    if member_name.endswith(".gz"):
        raw = gzip.decompress(raw)
    lines = raw.decode("utf-8").strip().splitlines()
    return np.array([line.split("\t")[0] for line in lines], dtype=object)


def _read_feature_names_from_zip(archive: zipfile.ZipFile, member_name: str) -> np.ndarray:
    raw = archive.read(member_name)
    if member_name.endswith(".gz"):
        raw = gzip.decompress(raw)
    rows = raw.decode("utf-8").strip().splitlines()
    names: list[str] = []
    for row in rows:
        parts = row.split("\t")
        if len(parts) >= 2:
            names.append(parts[1])
        else:
            names.append(parts[0])
    return np.array(names, dtype=object)


def _read_mtx_from_zip(archive: zipfile.ZipFile, member_name: str) -> np.ndarray:
    raw = archive.read(member_name)
    if member_name.endswith(".gz"):
        raw = gzip.decompress(raw)
    handle = io.StringIO(raw.decode("utf-8"))
    shape_read = False
    rows = cols = nnz = 0
    triplets: list[tuple[int, int, float]] = []
    for line in handle:
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        if not shape_read:
            rows, cols, nnz = [int(value) for value in line.split()]
            shape_read = True
            continue
        i_str, j_str, value_str = line.split()
        triplets.append((int(i_str) - 1, int(j_str) - 1, float(value_str)))
    matrix = np.zeros((rows, cols), dtype=np.float32)
    for i_idx, j_idx, value in triplets:
        matrix[i_idx, j_idx] = value
    return matrix.T


def select_highly_variable_genes(matrix: np.ndarray, gene_names: np.ndarray, top_k: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Keep genes with largest dispersion var / mean within the matrix."""
    mean = matrix.mean(axis=0)
    var = matrix.var(axis=0)
    dispersion = var / np.maximum(mean, 1e-8)
    order = np.argsort(dispersion)[::-1][: min(top_k, matrix.shape[1])]
    return matrix[:, order], gene_names[order]


def log1p_library_normalize(matrix: np.ndarray, target_sum: float = 1e4) -> np.ndarray:
    """Per-cell library normalization followed by log1p transform.

    For each cell c:
        x'_cg = log(1 + x_cg / sum_g x_cg * target_sum)
    """
    library_size = matrix.sum(axis=1, keepdims=True)
    normalized = matrix / np.maximum(library_size, 1.0) * target_sum
    return np.log1p(normalized)


def subset_cells(matrix: np.ndarray, barcodes: np.ndarray, wanted_barcodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wanted = set(str(barcode) for barcode in wanted_barcodes)
    mask = np.array([str(barcode) in wanted for barcode in barcodes], dtype=bool)
    return matrix[mask], barcodes[mask]


def read_dense_gene_cell_table_subset(
    path: str | Path,
    selected_cells: list[str] | np.ndarray,
    delimiter: str = "\t",
) -> CountMatrix:
    selected_list = [str(cell) for cell in selected_cells]
    selected_set = set(selected_list)
    with gzip.open(path, "rt") as handle:
        header = handle.readline().rstrip("\n").split(delimiter)
        cell_names = header
        selected_indices = [idx for idx, name in enumerate(cell_names) if name in selected_set]
        selected_barcodes = np.array([cell_names[idx] for idx in selected_indices], dtype=object)

        gene_names: list[str] = []
        rows: list[np.ndarray] = []
        for line in handle:
            parts = line.rstrip("\n").split(delimiter)
            gene_names.append(parts[0])
            values = np.array([float(parts[idx + 1]) for idx in selected_indices], dtype=np.float32)
            rows.append(values)

    matrix = np.vstack(rows).T if rows else np.zeros((0, 0), dtype=np.float32)
    return CountMatrix(matrix=matrix, gene_names=np.array(gene_names, dtype=object), cell_barcodes=selected_barcodes)


def read_10x_mtx_subset(
    matrix_path: str | Path,
    features_path: str | Path,
    barcodes_path: str | Path,
    selected_cells: list[str] | np.ndarray,
) -> CountMatrix:
    """Load only a selected subset of cells from 10x mtx/features/barcodes files."""
    selected_list = [str(cell) for cell in selected_cells]
    selected_set = set(selected_list)
    selected_pos = {barcode: idx for idx, barcode in enumerate(selected_list)}

    with gzip.open(barcodes_path, "rt") as handle:
        all_barcodes = [line.rstrip("\n").split("\t")[0] for line in handle]
    selected_columns = {
        col_idx + 1: selected_pos[barcode]
        for col_idx, barcode in enumerate(all_barcodes)
        if barcode in selected_set
    }
    selected_barcodes = np.array(selected_list, dtype=object)

    gene_names: list[str] = []
    with gzip.open(features_path, "rt") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            gene_names.append(parts[1] if len(parts) > 1 else parts[0])

    matrix = np.zeros((len(selected_list), len(gene_names)), dtype=np.float32)
    with gzip.open(matrix_path, "rt") as handle:
        shape_seen = False
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("%"):
                continue
            if not shape_seen:
                shape_seen = True
                continue
            row_str, col_str, value_str = line.split()
            col_idx = int(col_str)
            out_idx = selected_columns.get(col_idx)
            if out_idx is None:
                continue
            gene_idx = int(row_str) - 1
            matrix[out_idx, gene_idx] = float(value_str)

    return CountMatrix(matrix=matrix, gene_names=np.array(gene_names, dtype=object), cell_barcodes=selected_barcodes)


def read_10x_mtx_gene_cell_subset(
    matrix_path: str | Path,
    features_path: str | Path,
    barcodes_path: str | Path,
    selected_cells: list[str] | np.ndarray,
    selected_genes: list[str] | np.ndarray,
) -> CountMatrix:
    """Load only selected cells and selected genes from a 10x matrix bundle.

    This is a lighter-weight alternative to `read_10x_mtx_subset` when the
    downstream analysis only needs a fixed signature library instead of the
    full transcriptome.
    """
    selected_cell_list = [str(cell) for cell in selected_cells]
    selected_cell_set = set(selected_cell_list)
    selected_cell_pos = {barcode: idx for idx, barcode in enumerate(selected_cell_list)}

    wanted_gene_upper = {str(gene).upper() for gene in selected_genes}

    with gzip.open(barcodes_path, "rt") as handle:
        all_barcodes = [line.rstrip("\n").split("\t")[0] for line in handle]
    selected_columns = {
        col_idx + 1: selected_cell_pos[barcode]
        for col_idx, barcode in enumerate(all_barcodes)
        if barcode in selected_cell_set
    }

    selected_row_map: dict[int, int] = {}
    kept_gene_names: list[str] = []
    seen_gene_upper: set[str] = set()
    with gzip.open(features_path, "rt") as handle:
        for gene_idx, line in enumerate(handle, start=1):
            parts = line.rstrip("\n").split("\t")
            gene_name = parts[1] if len(parts) > 1 else parts[0]
            gene_upper = gene_name.upper()
            if gene_upper not in wanted_gene_upper or gene_upper in seen_gene_upper:
                continue
            selected_row_map[gene_idx] = len(kept_gene_names)
            kept_gene_names.append(gene_name)
            seen_gene_upper.add(gene_upper)

    matrix = np.zeros((len(selected_cell_list), len(kept_gene_names)), dtype=np.float32)
    with gzip.open(matrix_path, "rt") as handle:
        shape_seen = False
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("%"):
                continue
            if not shape_seen:
                shape_seen = True
                continue
            row_str, col_str, value_str = line.split()
            out_col = selected_columns.get(int(col_str))
            out_row = selected_row_map.get(int(row_str))
            if out_col is None or out_row is None:
                continue
            matrix[out_col, out_row] = float(value_str)

    return CountMatrix(
        matrix=matrix,
        gene_names=np.array(kept_gene_names, dtype=object),
        cell_barcodes=np.array(selected_cell_list, dtype=object),
    )
