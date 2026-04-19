from __future__ import annotations

import csv
import gzip
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from multicell_dynamics import (
    fit_module_representation,
    log1p_library_normalize,
    read_dense_gene_cell_table_subset,
    select_highly_variable_genes,
    top_genes_per_module,
)


MATCHED_PATIENTS = ["HCC03", "HCC04", "HCC06", "HCC09", "HCC10"]
SENDERS = ["Myeloid", "Fibroblast", "Endothelial", "T/NK", "B"]
MAX_CELLS_PER_GROUP = 250
MIN_CELLS_PER_SITE = 30
N_MODULES = 4
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
    "AKR1C2",
}


def load_metadata(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def choose_cells(rows: list[dict[str, str]], celltype: str, random_state: int) -> list[dict[str, str]]:
    rng = np.random.default_rng(random_state)
    by_group: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["patient"] in MATCHED_PATIENTS and row["celltype"] == celltype and row["site"] in {"Tumor", "Normal"}:
            by_group[(row["patient"], row["site"])].append(row)

    chosen: list[dict[str, str]] = []
    for key in sorted(by_group):
        group = by_group[key]
        if len(group) < MIN_CELLS_PER_SITE:
            continue
        if len(group) <= MAX_CELLS_PER_GROUP:
            chosen.extend(group)
        else:
            idx = rng.choice(len(group), size=MAX_CELLS_PER_GROUP, replace=False)
            chosen.extend(group[i] for i in sorted(idx))
    return chosen


def module_shift_summary(module_activity: np.ndarray, sites: np.ndarray) -> list[tuple[int, float, float, float]]:
    summary = []
    for module_idx in range(module_activity.shape[1]):
        tumor_mean = float(module_activity[sites == "Tumor", module_idx].mean())
        normal_mean = float(module_activity[sites == "Normal", module_idx].mean())
        summary.append((module_idx, tumor_mean, normal_mean, tumor_mean - normal_mean))
    return summary


def patient_site_score_table(
    rows: list[dict[str, str]],
    module_activity: np.ndarray,
    module_idx: int,
) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[(row["patient"], row["site"])].append(float(module_activity[idx, module_idx]))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def contamination_fraction(genes: list[tuple[str, float]]) -> float:
    if not genes:
        return 0.0
    hits = sum(1 for gene, _ in genes if gene in HEPATOCYTE_MARKERS)
    return hits / len(genes)


def choose_clean_tumor_module(
    shift: list[tuple[int, float, float, float]],
    top_genes: list[list[tuple[str, float]]],
    max_contamination: float = 0.35,
) -> int:
    candidates = [
        item
        for item in shift
        if item[3] > 0 and contamination_fraction(top_genes[item[0]]) <= max_contamination
    ]
    if candidates:
        return max(candidates, key=lambda item: item[3])[0]
    return max(shift, key=lambda item: item[3])[0]


def fit_celltype_modules(
    counts,
    rows: list[dict[str, str]],
    random_state: int,
) -> dict:
    cell_names = [row["Cell"] for row in rows]
    index_by_cell = {cell: idx for idx, cell in enumerate(counts.cell_barcodes)}
    idx = np.array([index_by_cell[cell] for cell in cell_names], dtype=int)
    matrix = counts.matrix[idx]
    sites = np.array([row["site"] for row in rows], dtype=object)
    norm = log1p_library_normalize(matrix)
    hvg, genes = select_highly_variable_genes(norm, counts.gene_names, top_k=1000)
    rep = fit_module_representation(hvg, n_modules=N_MODULES, random_state=random_state, max_iter=500)
    shift = module_shift_summary(rep.module_activity, sites)
    top_genes = top_genes_per_module(rep.module_weights, genes, top_k=10)
    tumor_module = choose_clean_tumor_module(shift, top_genes)
    scores = patient_site_score_table(rows, rep.module_activity, tumor_module)
    return {
        "rows": rows,
        "sites": sites,
        "rep": rep,
        "shift": shift,
        "tumor_module": tumor_module,
        "top_genes": top_genes,
        "scores": scores,
    }


def pearson_corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return float("nan")
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    metadata_path = Path("data/raw/gse149614/GSE149614_HCC.metadata.updated.txt.gz")
    counts_path = Path("data/raw/gse149614/GSE149614_HCC.scRNAseq.S71915.count.txt.gz")
    output_dir = Path("results/gse149614_multi_sender")
    rows = load_metadata(metadata_path)
    hepatocyte_rows = choose_cells(rows, "Hepatocyte", random_state=7)
    sender_rows = {
        sender: choose_cells(rows, sender, random_state=31 + idx)
        for idx, sender in enumerate(SENDERS)
    }

    all_cells = [row["Cell"] for row in hepatocyte_rows]
    for sender in SENDERS:
        all_cells.extend(row["Cell"] for row in sender_rows[sender])
    counts = read_dense_gene_cell_table_subset(counts_path, all_cells)

    hepatocyte_result = fit_celltype_modules(counts, hepatocyte_rows, random_state=7)
    hepatocyte_module = hepatocyte_result["tumor_module"]
    hepatocyte_scores = hepatocyte_result["scores"]

    result_lines = [
        "sender\tn_cells\tn_patient_sites\tsender_tumor_module\tsender_delta\t"
        "hepatocyte_marker_fraction\tcorrelation_with_hepatocyte_tumor_module\ttop_genes"
    ]
    pair_lines = ["sender\tpatient\tsite\tsender_score\thepatocyte_score"]

    print("GSE149614 multi-sender prototype")
    print("Matched patients:", MATCHED_PATIENTS)
    print("Hepatocytes:", len(hepatocyte_rows), dict(Counter(row["site"] for row in hepatocyte_rows)))
    print(f"Hepatocyte tumor-associated module: m_{hepatocyte_module}")
    print("  top genes:", ", ".join(g for g, _ in hepatocyte_result["top_genes"][hepatocyte_module]))
    print("Sender comparison:")

    for idx, sender in enumerate(SENDERS):
        rows_for_sender = sender_rows[sender]
        if not rows_for_sender:
            print(f"  {sender}: skipped, no cells after filtering")
            continue
        sender_result = fit_celltype_modules(counts, rows_for_sender, random_state=41 + idx)
        sender_module = sender_result["tumor_module"]
        sender_scores = sender_result["scores"]
        shared_keys = sorted(set(hepatocyte_scores) & set(sender_scores))
        sender_vec = np.array([sender_scores[key] for key in shared_keys], dtype=float)
        hep_vec = np.array([hepatocyte_scores[key] for key in shared_keys], dtype=float)
        corr = pearson_corr(sender_vec, hep_vec)
        sender_delta = [item[3] for item in sender_result["shift"] if item[0] == sender_module][0]
        genes = ", ".join(g for g, _ in sender_result["top_genes"][sender_module])
        contamination = contamination_fraction(sender_result["top_genes"][sender_module])

        print(
            f"  {sender}: cells={len(rows_for_sender)}, sites={len(shared_keys)}, "
            f"module=m_{sender_module}, delta={sender_delta:+.4f}, "
            f"hep_marker_fraction={contamination:.2f}, corr={corr:.4f}"
        )
        print(f"    top genes: {genes}")

        result_lines.append(
            f"{sender}\t{len(rows_for_sender)}\t{len(shared_keys)}\tm_{sender_module}\t"
            f"{sender_delta:.6f}\t{contamination:.6f}\t{corr:.6f}\t{genes}"
        )
        for key in shared_keys:
            pair_lines.append(
                f"{sender}\t{key[0]}\t{key[1]}\t{sender_scores[key]:.6f}\t{hepatocyte_scores[key]:.6f}"
            )

    save_text(output_dir / "sender_comparison.tsv", "\n".join(result_lines) + "\n")
    save_text(output_dir / "sender_patient_site_scores.tsv", "\n".join(pair_lines) + "\n")
    print("Saved:")
    print(f"  {output_dir / 'sender_comparison.tsv'}")
    print(f"  {output_dir / 'sender_patient_site_scores.tsv'}")


if __name__ == "__main__":
    main()
