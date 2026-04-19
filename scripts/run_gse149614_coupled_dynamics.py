from __future__ import annotations

import csv
import gzip
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from multicell_dynamics import (
    fit_module_representation,
    fit_population_dynamics,
    local_direction_from_pseudotime,
    log1p_library_normalize,
    orient_pseudotime_by_labels,
    pca_embedding,
    pseudotime_from_embedding,
    read_dense_gene_cell_table_subset,
    select_highly_variable_genes,
    velocity_r2_score,
    velocity_sign_agreement,
)


MATCHED_PATIENTS = ["HCC03", "HCC04", "HCC06", "HCC09", "HCC10"]
SENDERS = ["Myeloid", "Fibroblast", "Endothelial", "T/NK", "B"]
MAX_CELLS_PER_GROUP = 250
MIN_CELLS_PER_SITE = 30
N_MODULES = 4


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


def fit_modules_for_rows(counts, rows: list[dict[str, str]], random_state: int) -> dict:
    cell_names = [row["Cell"] for row in rows]
    index_by_cell = {cell: idx for idx, cell in enumerate(counts.cell_barcodes)}
    idx = np.array([index_by_cell[cell] for cell in cell_names], dtype=int)
    matrix = counts.matrix[idx]
    sites = np.array([row["site"] for row in rows], dtype=object)
    norm = log1p_library_normalize(matrix)
    hvg, _ = select_highly_variable_genes(norm, counts.gene_names, top_k=1000)
    rep = fit_module_representation(hvg, n_modules=N_MODULES, random_state=random_state, max_iter=500)
    embedding = pca_embedding(hvg, n_components=2)
    pseudotime = pseudotime_from_embedding(embedding, axis=0)
    pseudotime = orient_pseudotime_by_labels(pseudotime, sites, low_label="Normal", high_label="Tumor")
    local_velocity = local_direction_from_pseudotime(rep.module_activity, pseudotime, k_neighbors=20)
    return {
        "rows": rows,
        "sites": sites,
        "module_activity": rep.module_activity,
        "embedding": embedding,
        "pseudotime": pseudotime,
        "local_velocity": local_velocity,
    }


def patient_site_mean(rows: list[dict[str, str]], values: np.ndarray) -> dict[tuple[str, str], np.ndarray]:
    grouped: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[(row["patient"], row["site"])].append(values[idx])
    return {key: np.mean(np.vstack(items), axis=0) for key, items in grouped.items()}


def build_external_input(
    hep_rows: list[dict[str, str]],
    sender_means: dict[tuple[str, str], np.ndarray],
) -> np.ndarray:
    n_sender_modules = len(next(iter(sender_means.values())))
    external = np.zeros((len(hep_rows), n_sender_modules), dtype=float)
    global_mean = np.mean(np.vstack(list(sender_means.values())), axis=0)
    for idx, row in enumerate(hep_rows):
        external[idx] = sender_means.get((row["patient"], row["site"]), global_mean)
    return external


def split_by_patient(
    rows: list[dict[str, str]],
    test_patients: set[str],
) -> tuple[np.ndarray, np.ndarray]:
    train_idx = []
    test_idx = []
    for idx, row in enumerate(rows):
        if row["patient"] in test_patients:
            test_idx.append(idx)
        else:
            train_idx.append(idx)
    return np.array(train_idx, dtype=int), np.array(test_idx, dtype=int)


def evaluate_model(name: str, hep_result: dict, external_input: np.ndarray | None, test_patients: set[str]) -> dict:
    rows = hep_result["rows"]
    train_idx, test_idx = split_by_patient(rows, test_patients)
    model = fit_population_dynamics(
        module_activity=hep_result["module_activity"][train_idx],
        state_embedding=hep_result["embedding"][train_idx],
        module_velocity=hep_result["local_velocity"][train_idx],
        external_input=None if external_input is None else external_input[train_idx],
        alpha=0.03,
    )
    pred = model.predict_velocity(
        module_activity=hep_result["module_activity"][test_idx],
        state_embedding=hep_result["embedding"][test_idx],
        external_input=None if external_input is None else external_input[test_idx],
    )
    true = hep_result["local_velocity"][test_idx]
    return {
        "model": name,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "test_r2": velocity_r2_score(true, pred),
        "test_sign_agreement": velocity_sign_agreement(true, pred),
    }


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    metadata_path = Path("data/raw/gse149614/GSE149614_HCC.metadata.updated.txt.gz")
    counts_path = Path("data/raw/gse149614/GSE149614_HCC.scRNAseq.S71915.count.txt.gz")
    output_dir = Path("results/gse149614_coupled_dynamics")
    rows = load_metadata(metadata_path)

    hep_rows = choose_cells(rows, "Hepatocyte", random_state=7)
    sender_rows = {
        sender: choose_cells(rows, sender, random_state=37 + idx)
        for idx, sender in enumerate(SENDERS)
    }
    all_cells = [row["Cell"] for row in hep_rows]
    for sender in SENDERS:
        all_cells.extend(row["Cell"] for row in sender_rows[sender])
    counts = read_dense_gene_cell_table_subset(counts_path, all_cells)

    hep_result = fit_modules_for_rows(counts, hep_rows, random_state=7)
    sender_results = {
        sender: fit_modules_for_rows(counts, sender_rows[sender], random_state=47 + idx)
        for idx, sender in enumerate(SENDERS)
        if sender_rows[sender]
    }
    sender_external = {
        sender: build_external_input(
            hep_rows,
            patient_site_mean(result["rows"], result["module_activity"]),
        )
        for sender, result in sender_results.items()
    }
    all_external = np.concatenate([sender_external[sender] for sender in SENDERS if sender in sender_external], axis=1)

    test_patients = {"HCC09", "HCC10"}
    results = [evaluate_model("hepatocyte_only", hep_result, None, test_patients)]
    for sender in SENDERS:
        if sender in sender_external:
            results.append(evaluate_model(f"hepatocyte_plus_{sender}", hep_result, sender_external[sender], test_patients))
    results.append(evaluate_model("hepatocyte_plus_all_senders", hep_result, all_external, test_patients))

    baseline = results[0]
    lines = ["model\tn_train\tn_test\ttest_r2\tdelta_r2_vs_hepatocyte_only\ttest_sign_agreement\tdelta_sign_vs_hepatocyte_only"]
    print("GSE149614 coupled dynamics prototype")
    print("Test patients:", sorted(test_patients))
    print("Hepatocytes:", len(hep_rows), dict(Counter(row["site"] for row in hep_rows)))
    print("Sender cells:")
    for sender in SENDERS:
        print(f"  {sender}: {len(sender_rows[sender])} {dict(Counter(row['site'] for row in sender_rows[sender]))}")
    print("Model comparison:")
    for item in results:
        delta_r2 = item["test_r2"] - baseline["test_r2"]
        delta_sign = item["test_sign_agreement"] - baseline["test_sign_agreement"]
        print(
            f"  {item['model']}: test_r2={item['test_r2']:.4f} "
            f"(delta={delta_r2:+.4f}), sign={item['test_sign_agreement']:.4f} "
            f"(delta={delta_sign:+.4f})"
        )
        lines.append(
            f"{item['model']}\t{item['n_train']}\t{item['n_test']}\t"
            f"{item['test_r2']:.6f}\t{delta_r2:.6f}\t"
            f"{item['test_sign_agreement']:.6f}\t{delta_sign:.6f}"
        )
    save_text(output_dir / "model_comparison.tsv", "\n".join(lines) + "\n")
    print("Saved:")
    print(f"  {output_dir / 'model_comparison.tsv'}")


if __name__ == "__main__":
    main()
