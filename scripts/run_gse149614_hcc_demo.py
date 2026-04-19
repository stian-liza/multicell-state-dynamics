from __future__ import annotations

import csv
import gzip
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from multicell_dynamics import (
    fit_module_representation,
    log1p_library_normalize,
    pca_embedding,
    read_dense_gene_cell_table_subset,
    select_highly_variable_genes,
    top_genes_per_module,
)


MATCHED_PATIENTS = ["HCC03", "HCC04", "HCC06", "HCC09", "HCC10"]
MAX_CELLS_PER_GROUP = 300
HEPATOCYTE_MARKERS = {
    "ALB",
    "APOA1",
    "APOA2",
    "APOA4",
    "APOC1",
    "APOC2",
    "APOC3",
    "APOE",
    "AHSG",
    "FGA",
    "FGB",
    "FGG",
    "HP",
    "RBP4",
    "TTR",
    "CYP2E1",
}
QC_PATH = Path("results/gse149614_myeloid_qc/myeloid_marker_scores.tsv")


def load_metadata(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def choose_cells(rows: list[dict[str, str]], celltype: str, random_state: int = 7) -> list[dict[str, str]]:
    rng = np.random.default_rng(random_state)
    by_group: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["patient"] in MATCHED_PATIENTS and row["celltype"] == celltype and row["site"] in {"Tumor", "Normal"}:
            by_group[(row["patient"], row["site"], row["celltype"])].append(row)

    chosen: list[dict[str, str]] = []
    for key in sorted(by_group):
        group = by_group[key]
        if len(group) <= MAX_CELLS_PER_GROUP:
            chosen.extend(group)
        else:
            idx = rng.choice(len(group), size=MAX_CELLS_PER_GROUP, replace=False)
            chosen.extend(group[i] for i in sorted(idx))
    return chosen


def load_hepatocyte_like_myeloid_flags(path: Path = QC_PATH) -> dict[str, int]:
    if not path.exists():
        return {}
    flags: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            flags[row["cell"]] = int(row["hepatocyte_like_flag"])
    return flags


def filter_myeloid_rows(rows: list[dict[str, str]], remove_hepatocyte_like: bool) -> list[dict[str, str]]:
    if not remove_hepatocyte_like:
        return rows
    flags = load_hepatocyte_like_myeloid_flags()
    if not flags:
        print("Warning: QC flags not found; run scripts/check_gse149614_myeloid_signal.py first.")
        return rows
    return [row for row in rows if flags.get(row["Cell"], 0) == 0]


def module_shift_summary(module_activity: np.ndarray, sites: np.ndarray) -> list[tuple[int, float, float, float]]:
    summary = []
    for module_idx in range(module_activity.shape[1]):
        tumor_mean = float(module_activity[sites == "Tumor", module_idx].mean())
        normal_mean = float(module_activity[sites == "Normal", module_idx].mean())
        summary.append((module_idx, tumor_mean, normal_mean, tumor_mean - normal_mean))
    return summary


def patient_site_score_table(
    labels: list[dict[str, str]],
    module_activity: np.ndarray,
    tumor_module_idx: int,
) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for idx, row in enumerate(labels):
        grouped[(row["patient"], row["site"])].append(float(module_activity[idx, tumor_module_idx]))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def contamination_fraction(genes: list[tuple[str, float]]) -> float:
    if not genes:
        return 0.0
    hits = sum(1 for gene, _ in genes if gene in HEPATOCYTE_MARKERS)
    return hits / len(genes)


def choose_clean_tumor_shift_module(
    shift: list[tuple[int, float, float, float]],
    module_genes: list[list[tuple[str, float]]],
    max_contamination: float = 0.35,
) -> int:
    candidates = [
        item
        for item in shift
        if item[3] > 0 and contamination_fraction(module_genes[item[0]]) <= max_contamination
    ]
    if candidates:
        return max(candidates, key=lambda item: item[3])[0]
    return max(shift, key=lambda item: item[3])[0]


def main() -> None:
    metadata_path = Path("data/raw/gse149614/GSE149614_HCC.metadata.updated.txt.gz")
    counts_path = Path("data/raw/gse149614/GSE149614_HCC.scRNAseq.S71915.count.txt.gz")
    if not metadata_path.exists():
        raise SystemExit(f"Missing metadata file: {metadata_path}")
    if not counts_path.exists():
        raise SystemExit(f"Missing count file: {counts_path}")

    rows = load_metadata(metadata_path)
    hep_rows = choose_cells(rows, "Hepatocyte", random_state=7)
    myeloid_rows = choose_cells(rows, "Myeloid", random_state=17)
    clean_myeloid_rows = filter_myeloid_rows(myeloid_rows, remove_hepatocyte_like=True)

    hep_cells = [row["Cell"] for row in hep_rows]
    myeloid_cells = [row["Cell"] for row in myeloid_rows]
    clean_myeloid_cells = [row["Cell"] for row in clean_myeloid_rows]
    selected_cells = hep_cells + myeloid_cells + clean_myeloid_cells

    counts = read_dense_gene_cell_table_subset(counts_path, selected_cells)
    index_by_cell = {cell: idx for idx, cell in enumerate(counts.cell_barcodes)}
    hep_idx = np.array([index_by_cell[cell] for cell in hep_cells], dtype=int)
    myeloid_idx = np.array([index_by_cell[cell] for cell in myeloid_cells], dtype=int)
    clean_myeloid_idx = np.array([index_by_cell[cell] for cell in clean_myeloid_cells], dtype=int)

    hep_matrix = counts.matrix[hep_idx]
    myeloid_matrix = counts.matrix[myeloid_idx]
    clean_myeloid_matrix = counts.matrix[clean_myeloid_idx]
    hep_sites = np.array([row["site"] for row in hep_rows], dtype=object)
    myeloid_sites = np.array([row["site"] for row in myeloid_rows], dtype=object)
    clean_myeloid_sites = np.array([row["site"] for row in clean_myeloid_rows], dtype=object)

    hep_norm = log1p_library_normalize(hep_matrix)
    myeloid_norm = log1p_library_normalize(myeloid_matrix)
    clean_myeloid_norm = log1p_library_normalize(clean_myeloid_matrix)
    hep_hvg, hep_genes = select_highly_variable_genes(hep_norm, counts.gene_names, top_k=1200)
    myeloid_hvg, myeloid_genes = select_highly_variable_genes(myeloid_norm, counts.gene_names, top_k=1200)
    clean_myeloid_hvg, clean_myeloid_genes = select_highly_variable_genes(
        clean_myeloid_norm, counts.gene_names, top_k=1200
    )

    hep_rep = fit_module_representation(hep_hvg, n_modules=4, random_state=7)
    myeloid_rep = fit_module_representation(myeloid_hvg, n_modules=4, random_state=17)
    clean_myeloid_rep = fit_module_representation(clean_myeloid_hvg, n_modules=4, random_state=19)

    hep_shift = module_shift_summary(hep_rep.module_activity, hep_sites)
    myeloid_shift = module_shift_summary(myeloid_rep.module_activity, myeloid_sites)
    clean_myeloid_shift = module_shift_summary(clean_myeloid_rep.module_activity, clean_myeloid_sites)
    hep_module_genes = top_genes_per_module(hep_rep.module_weights, hep_genes, top_k=10)
    myeloid_module_genes = top_genes_per_module(myeloid_rep.module_weights, myeloid_genes, top_k=10)
    clean_myeloid_module_genes = top_genes_per_module(clean_myeloid_rep.module_weights, clean_myeloid_genes, top_k=10)
    hep_tumor_module = max(hep_shift, key=lambda item: item[3])[0]
    myeloid_tumor_module = choose_clean_tumor_shift_module(myeloid_shift, myeloid_module_genes)
    clean_myeloid_tumor_module = choose_clean_tumor_shift_module(clean_myeloid_shift, clean_myeloid_module_genes)

    hep_patient_scores = patient_site_score_table(hep_rows, hep_rep.module_activity, hep_tumor_module)
    myeloid_patient_scores = patient_site_score_table(myeloid_rows, myeloid_rep.module_activity, myeloid_tumor_module)
    clean_myeloid_patient_scores = patient_site_score_table(
        clean_myeloid_rows,
        clean_myeloid_rep.module_activity,
        clean_myeloid_tumor_module,
    )
    shared_keys = sorted(set(hep_patient_scores) & set(myeloid_patient_scores))
    hep_vec = np.array([hep_patient_scores[key] for key in shared_keys], dtype=float)
    myeloid_vec = np.array([myeloid_patient_scores[key] for key in shared_keys], dtype=float)
    coupling_corr = float(np.corrcoef(myeloid_vec, hep_vec)[0, 1]) if len(shared_keys) >= 2 else float("nan")
    clean_shared_keys = sorted(set(hep_patient_scores) & set(clean_myeloid_patient_scores))
    clean_hep_vec = np.array([hep_patient_scores[key] for key in clean_shared_keys], dtype=float)
    clean_myeloid_vec = np.array([clean_myeloid_patient_scores[key] for key in clean_shared_keys], dtype=float)
    clean_coupling_corr = (
        float(np.corrcoef(clean_myeloid_vec, clean_hep_vec)[0, 1]) if len(clean_shared_keys) >= 2 else float("nan")
    )

    print("GSE149614 HCC sender-receiver prototype")
    print("Matched patients:", MATCHED_PATIENTS)
    print("Selected hepatocytes:", len(hep_rows), dict(Counter(hep_sites)))
    print("Selected myeloid cells:", len(myeloid_rows), dict(Counter(myeloid_sites)))
    print("Clean myeloid cells after removing hepatocyte-like flagged cells:", len(clean_myeloid_rows), dict(Counter(clean_myeloid_sites)))
    print("Hepatocyte top tumor-shift module:")
    for module_idx, tumor_mean, normal_mean, delta in hep_shift:
        print(f"  m_{module_idx}: tumor_mean={tumor_mean:.4f}, normal_mean={normal_mean:.4f}, delta={delta:+.4f}")
    print("Myeloid top tumor-shift module:")
    for module_idx, tumor_mean, normal_mean, delta in myeloid_shift:
        print(f"  m_{module_idx}: tumor_mean={tumor_mean:.4f}, normal_mean={normal_mean:.4f}, delta={delta:+.4f}")

    print(f"Hepatocyte tumor-associated module: m_{hep_tumor_module}")
    print("  top genes:", ", ".join(g for g, _ in hep_module_genes[hep_tumor_module]))
    print("All myeloid module top genes and hepatocyte-marker fractions:")
    for module_idx, genes in enumerate(myeloid_module_genes):
        frac = contamination_fraction(genes)
        print(f"  m_{module_idx} contamination_fraction={frac:.2f}: {', '.join(g for g, _ in genes)}")
    print(f"Myeloid selected tumor-associated module: m_{myeloid_tumor_module}")
    print("  top genes:", ", ".join(g for g, _ in myeloid_module_genes[myeloid_tumor_module]))
    print("Clean myeloid tumor-shift modules after removing hepatocyte-like flagged cells:")
    for module_idx, tumor_mean, normal_mean, delta in clean_myeloid_shift:
        print(f"  m_{module_idx}: tumor_mean={tumor_mean:.4f}, normal_mean={normal_mean:.4f}, delta={delta:+.4f}")
    print(f"Clean myeloid selected tumor-associated module: m_{clean_myeloid_tumor_module}")
    print("  top genes:", ", ".join(g for g, _ in clean_myeloid_module_genes[clean_myeloid_tumor_module]))
    print("Patient-site matched coupling scores:")
    for key in shared_keys:
        print(
            f"  {key[0]} {key[1]}: myeloid_score={myeloid_patient_scores[key]:.4f}, "
            f"hepatocyte_score={hep_patient_scores[key]:.4f}"
        )
    print("Sender-receiver correlation (myeloid tumor module vs hepatocyte tumor module):", round(coupling_corr, 4))
    print(
        "Sender-receiver correlation after removing hepatocyte-like myeloid:",
        round(clean_coupling_corr, 4),
    )


if __name__ == "__main__":
    main()
