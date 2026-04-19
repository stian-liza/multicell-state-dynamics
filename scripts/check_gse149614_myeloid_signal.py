from __future__ import annotations

import csv
import gzip
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from multicell_dynamics import log1p_library_normalize, read_dense_gene_cell_table_subset


MAX_CELLS_PER_GROUP = 500
RANDOM_STATE = 23

MYELOID_MARKERS = [
    "LYZ",
    "LST1",
    "CST3",
    "TYROBP",
    "AIF1",
    "FCGR3A",
    "FCER1G",
    "S100A8",
    "S100A9",
    "CTSS",
]

HEPATOCYTE_MARKERS = [
    "ALB",
    "APOA1",
    "APOA2",
    "APOC3",
    "APOE",
    "AHSG",
    "RBP4",
    "TTR",
    "HP",
    "CYP2E1",
]

INFLAMMATORY_MARKERS = [
    "CCL3",
    "CCL3L3",
    "CCL4",
    "CCL4L2",
    "CXCL8",
    "CXCL3",
    "CXCL2",
    "IL1B",
    "CCL20",
    "NFKBIA",
]


def load_metadata(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def choose_myeloid_cells(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rng = np.random.default_rng(RANDOM_STATE)
    by_group: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["celltype"] == "Myeloid" and row["site"] in {"Tumor", "Normal", "PVTT", "Lymph"}:
            by_group[(row["patient"], row["site"])].append(row)

    chosen: list[dict[str, str]] = []
    for key in sorted(by_group):
        group = by_group[key]
        if len(group) <= MAX_CELLS_PER_GROUP:
            chosen.extend(group)
        else:
            idx = rng.choice(len(group), size=MAX_CELLS_PER_GROUP, replace=False)
            chosen.extend(group[i] for i in sorted(idx))
    return chosen


def marker_score(matrix: np.ndarray, gene_names: np.ndarray, markers: list[str]) -> np.ndarray:
    index = {str(gene): idx for idx, gene in enumerate(gene_names)}
    present = [index[gene] for gene in markers if gene in index]
    if not present:
        return np.zeros(matrix.shape[0], dtype=float)
    return matrix[:, present].mean(axis=1)


def zscore(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / max(float(values.std()), 1e-8)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    metadata_path = Path("data/raw/gse149614/GSE149614_HCC.metadata.updated.txt.gz")
    counts_path = Path("data/raw/gse149614/GSE149614_HCC.scRNAseq.S71915.count.txt.gz")
    output_dir = Path("results/gse149614_myeloid_qc")
    rows = load_metadata(metadata_path)
    myeloid_rows = choose_myeloid_cells(rows)
    cells = [row["Cell"] for row in myeloid_rows]
    counts = read_dense_gene_cell_table_subset(counts_path, cells)
    index_by_cell = {cell: idx for idx, cell in enumerate(counts.cell_barcodes)}
    order = np.array([index_by_cell[cell] for cell in cells], dtype=int)
    matrix = counts.matrix[order]
    normalized = log1p_library_normalize(matrix)

    myeloid_score = marker_score(normalized, counts.gene_names, MYELOID_MARKERS)
    hepatocyte_score = marker_score(normalized, counts.gene_names, HEPATOCYTE_MARKERS)
    inflammatory_score = marker_score(normalized, counts.gene_names, INFLAMMATORY_MARKERS)
    hepatocyte_like_z = zscore(hepatocyte_score)
    myeloid_z = zscore(myeloid_score)
    inflammatory_z = zscore(inflammatory_score)
    hepatocyte_like_flag = hepatocyte_like_z >= 1.5
    possible_doublet_flag = hepatocyte_like_flag & (myeloid_z >= 0.0)

    cell_lines = [
        "cell\tpatient\tsite\tmyeloid_score\thepatocyte_marker_score\tinflammatory_score\t"
        "myeloid_z\thepatocyte_marker_z\tinflammatory_z\thepatocyte_like_flag\tpossible_doublet_or_uptake"
    ]
    for idx, row in enumerate(myeloid_rows):
        cell_lines.append(
            f"{row['Cell']}\t{row['patient']}\t{row['site']}\t"
            f"{myeloid_score[idx]:.6f}\t{hepatocyte_score[idx]:.6f}\t{inflammatory_score[idx]:.6f}\t"
            f"{myeloid_z[idx]:.6f}\t{hepatocyte_like_z[idx]:.6f}\t{inflammatory_z[idx]:.6f}\t"
            f"{int(hepatocyte_like_flag[idx])}\t{int(possible_doublet_flag[idx])}"
        )
    save_text(output_dir / "myeloid_marker_scores.tsv", "\n".join(cell_lines) + "\n")

    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(myeloid_rows):
        grouped[(row["patient"], row["site"])].append(idx)

    summary_lines = [
        "patient\tsite\tn_cells\tmean_myeloid_score\tmean_hepatocyte_marker_score\t"
        "mean_inflammatory_score\thepatocyte_like_fraction\tpossible_doublet_or_uptake_fraction"
    ]
    for key in sorted(grouped):
        idx = np.array(grouped[key], dtype=int)
        summary_lines.append(
            f"{key[0]}\t{key[1]}\t{len(idx)}\t"
            f"{float(myeloid_score[idx].mean()):.6f}\t"
            f"{float(hepatocyte_score[idx].mean()):.6f}\t"
            f"{float(inflammatory_score[idx].mean()):.6f}\t"
            f"{float(hepatocyte_like_flag[idx].mean()):.6f}\t"
            f"{float(possible_doublet_flag[idx].mean()):.6f}"
        )
    save_text(output_dir / "patient_site_summary.tsv", "\n".join(summary_lines) + "\n")

    high_idx = np.where(hepatocyte_like_flag)[0]
    low_idx = np.where(~hepatocyte_like_flag)[0]
    contrast_lines = [
        "group\tn_cells\tmean_myeloid_score\tmean_hepatocyte_marker_score\tmean_inflammatory_score\t"
        "mean_myeloid_z\tmean_hepatocyte_marker_z\tmean_inflammatory_z\tpossible_doublet_or_uptake_fraction"
    ]
    for group_name, idx in [("hepatocyte_like_high", high_idx), ("hepatocyte_like_low", low_idx)]:
        contrast_lines.append(
            f"{group_name}\t{len(idx)}\t"
            f"{float(myeloid_score[idx].mean()):.6f}\t"
            f"{float(hepatocyte_score[idx].mean()):.6f}\t"
            f"{float(inflammatory_score[idx].mean()):.6f}\t"
            f"{float(myeloid_z[idx].mean()):.6f}\t"
            f"{float(hepatocyte_like_z[idx].mean()):.6f}\t"
            f"{float(inflammatory_z[idx].mean()):.6f}\t"
            f"{float(possible_doublet_flag[idx].mean()):.6f}"
        )
    save_text(output_dir / "hepatocyte_like_high_low_contrast.tsv", "\n".join(contrast_lines) + "\n")

    distribution_lines = ["group\tpatient\tsite\tn_cells\tfraction_within_group"]
    for group_name, idx in [("hepatocyte_like_high", high_idx), ("hepatocyte_like_low", low_idx)]:
        counts = Counter((myeloid_rows[i]["patient"], myeloid_rows[i]["site"]) for i in idx)
        denom = max(len(idx), 1)
        for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            distribution_lines.append(f"{group_name}\t{key[0]}\t{key[1]}\t{value}\t{value / denom:.6f}")
    save_text(output_dir / "hepatocyte_like_group_distribution.tsv", "\n".join(distribution_lines) + "\n")

    site_summary = Counter(row["site"] for row in myeloid_rows)
    print("GSE149614 myeloid hepatocyte-like signal QC")
    print("Cells used:", len(myeloid_rows), dict(site_summary))
    print("Output directory:", output_dir)
    print("Overall hepatocyte-like fraction:", round(float(hepatocyte_like_flag.mean()), 4))
    print("Overall possible doublet/uptake fraction:", round(float(possible_doublet_flag.mean()), 4))
    print("High-vs-low hepatocyte-like contrast:")
    for group_name, idx in [("high", high_idx), ("low", low_idx)]:
        print(
            f"  {group_name}: n={len(idx)}, myeloid={myeloid_score[idx].mean():.4f}, "
            f"hepatocyte_like={hepatocyte_score[idx].mean():.4f}, inflammatory={inflammatory_score[idx].mean():.4f}, "
            f"possible_doublet_or_uptake={possible_doublet_flag[idx].mean():.4f}"
        )
    print("Mean marker scores by site:")
    for site in sorted({row["site"] for row in myeloid_rows}):
        idx = np.array([i for i, row in enumerate(myeloid_rows) if row["site"] == site], dtype=int)
        print(
            f"  {site}: n={len(idx)}, myeloid={myeloid_score[idx].mean():.4f}, "
            f"hepatocyte_like={hepatocyte_score[idx].mean():.4f}, inflammatory={inflammatory_score[idx].mean():.4f}, "
            f"hep_like_fraction={hepatocyte_like_flag[idx].mean():.4f}"
        )
    print("Saved:")
    print(f"  {output_dir / 'myeloid_marker_scores.tsv'}")
    print(f"  {output_dir / 'patient_site_summary.tsv'}")
    print(f"  {output_dir / 'hepatocyte_like_high_low_contrast.tsv'}")
    print(f"  {output_dir / 'hepatocyte_like_group_distribution.tsv'}")


if __name__ == "__main__":
    main()
