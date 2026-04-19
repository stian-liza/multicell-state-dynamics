from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from run_scp2154_phenotype_baseline import read_metadata, split_donors, stratified_sample
from run_scp2154_sender_comparison import fit_baseline, module_delta
from run_scp2154_stromal_to_hepatocyte_coupling import annotate_program, per_module_r2, save_text, top_gene_names, zscore_by_train
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


RECEIVER = "Hepatocyte"
SENDER = "Stromal"
LABEL_COL = "_condition_label"


def parse_comparisons(raw: str) -> list[tuple[str, str]]:
    out = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Comparison should look like column:value, got {item!r}")
        column, value = item.split(":", 1)
        out.append((column.strip(), value.strip()))
    return out


def condition_key(column: str, value: str) -> str:
    return f"{column}_{value}".replace(" ", "_").replace("/", "_")


def condition_relation(delta: float, threshold: float = 0.1) -> str:
    if delta > threshold:
        return "Condition-up"
    if delta < -threshold:
        return "Condition-down"
    return "near-neutral"


def coupling_direction(coeff: float, hep_delta: float, delta_r2: float) -> str:
    if delta_r2 <= 0:
        return "no_heldout_gain"
    if coeff > 0 and hep_delta > 0:
        return "candidate_supports_condition_up_module"
    if coeff < 0 and hep_delta < 0:
        return "candidate_supports_condition_down_module"
    if coeff < 0 and hep_delta > 0:
        return "opposes_condition_up_module"
    if coeff > 0 and hep_delta < 0:
        return "opposes_condition_down_module"
    return "weak_direction"


def make_condition_rows(
    rows: list[dict[str, str]],
    cell_type: str,
    condition_col: str,
    condition_value: str,
    max_cells_per_donor_condition: int,
    max_cells_per_condition: int,
    random_state: int,
) -> list[dict[str, str]]:
    selected = []
    allowed = {"healthy", condition_value}
    for row in rows:
        if row.get("organ__ontology_label") != "liver":
            continue
        if row.get("Cell_Type") != cell_type:
            continue
        if row.get(condition_col) not in allowed:
            continue
        copied = dict(row)
        copied[LABEL_COL] = row[condition_col]
        selected.append(copied)
    return stratified_sample(
        selected,
        phenotype_col=LABEL_COL,
        donor_col="donor_id",
        max_cells_per_donor_phenotype=max_cells_per_donor_condition,
        max_cells_per_phenotype=max_cells_per_condition,
        random_state=random_state,
    )


def fit_condition_representation(
    counts,
    rows: list[dict[str, str]],
    condition_value: str,
    n_modules: int,
    hvg: int,
    random_state: int,
) -> dict:
    cell_to_index = {cell: idx for idx, cell in enumerate(counts.cell_barcodes)}
    idx = np.array([cell_to_index[row["NAME"]] for row in rows], dtype=int)
    matrix = counts.matrix[idx]
    labels = np.array([row[LABEL_COL] for row in rows], dtype=object)
    hvg_matrix, hvg_names = select_highly_variable_genes(matrix, counts.gene_names, top_k=hvg)
    rep = fit_module_representation(hvg_matrix, n_modules=n_modules, random_state=random_state, max_iter=700)
    embedding = pca_embedding(hvg_matrix, n_components=2)
    pseudotime = pseudotime_from_embedding(embedding, axis=0)
    pseudotime = orient_pseudotime_by_labels(pseudotime, labels, low_label="healthy", high_label=condition_value)
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


def module_shift(module_activity: np.ndarray, labels: np.ndarray, condition_value: str) -> list[tuple[int, float, float, float]]:
    out = []
    for module_idx in range(module_activity.shape[1]):
        condition_mean = float(module_activity[labels == condition_value, module_idx].mean())
        healthy_mean = float(module_activity[labels == "healthy", module_idx].mean())
        out.append((module_idx, condition_mean, healthy_mean, condition_mean - healthy_mean))
    return out


def sender_scores_by_donor_condition(rows: list[dict[str, str]], module_activity: np.ndarray) -> dict[tuple[str, str], np.ndarray]:
    grouped: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[(row["donor_id"], row[LABEL_COL])].append(module_activity[idx])
    return {key: np.mean(np.vstack(items), axis=0) for key, items in grouped.items()}


def build_sender_input(receiver_rows: list[dict[str, str]], sender_scores: dict[tuple[str, str], np.ndarray]) -> np.ndarray:
    global_mean = np.mean(np.vstack(list(sender_scores.values())), axis=0)
    out = np.zeros((len(receiver_rows), global_mean.shape[0]), dtype=float)
    for idx, row in enumerate(receiver_rows):
        out[idx] = sender_scores.get((row["donor_id"], row[LABEL_COL]), global_mean)
    return out


def split_indices_by_donor(rows: list[dict[str, str]], random_state: int) -> tuple[np.ndarray, np.ndarray]:
    split = split_donors(rows, "donor_id", random_state=random_state)
    train = [idx for idx, row in enumerate(rows) if split[row["donor_id"]] != "test"]
    test = [idx for idx, row in enumerate(rows) if split[row["donor_id"]] == "test"]
    return np.array(train, dtype=int), np.array(test, dtype=int)


def evaluate_condition(
    condition_col: str,
    condition_value: str,
    sender: dict,
    receiver: dict,
    receiver_rows: list[dict[str, str]],
    sender_shift: list[tuple[int, float, float, float]],
    receiver_shift: list[tuple[int, float, float, float]],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    alpha: float,
) -> tuple[list[dict], dict]:
    baseline = fit_baseline(receiver, train_idx, test_idx, alpha)
    sender_scores = sender_scores_by_donor_condition(sender["rows"], sender["rep"].module_activity)
    raw_sender_input = build_sender_input(receiver_rows, sender_scores)
    sender_input = zscore_by_train(raw_sender_input, train_idx)
    records = []
    for sender_module_idx in range(sender_input.shape[1]):
        single_input = sender_input[:, [sender_module_idx]]
        model = fit_population_dynamics(
            module_activity=receiver["rep"].module_activity[train_idx],
            state_embedding=receiver["embedding"][train_idx],
            module_velocity=receiver["local_velocity"][train_idx],
            external_input=single_input[train_idx],
            alpha=alpha,
        )
        prediction = model.predict_velocity(
            receiver["rep"].module_activity[test_idx],
            receiver["embedding"][test_idx],
            external_input=single_input[test_idx],
        )
        coupled_r2 = per_module_r2(receiver["local_velocity"][test_idx], prediction)
        coeff_idx = model.feature_names.index("e_0")
        sender_delta = module_delta(sender_shift, sender_module_idx)
        sender_genes = top_gene_names(sender["top_genes"][sender_module_idx])
        for receiver_module_idx in range(receiver["rep"].module_activity.shape[1]):
            receiver_delta = module_delta(receiver_shift, receiver_module_idx)
            receiver_genes = top_gene_names(receiver["top_genes"][receiver_module_idx])
            delta_r2 = float(coupled_r2[receiver_module_idx] - baseline["per_module_r2"][receiver_module_idx])
            coeff = float(model.coefficient_matrix[receiver_module_idx, coeff_idx])
            records.append(
                {
                    "condition": condition_value,
                    "condition_col": condition_col,
                    "sender_module": sender_module_idx,
                    "sender_delta": sender_delta,
                    "sender_relation": condition_relation(sender_delta),
                    "sender_program": annotate_program(SENDER, sender_genes),
                    "sender_top_genes": ",".join(sender_genes),
                    "receiver_module": receiver_module_idx,
                    "receiver_delta": receiver_delta,
                    "receiver_relation": condition_relation(receiver_delta),
                    "receiver_program": annotate_program(RECEIVER, receiver_genes),
                    "receiver_top_genes": ",".join(receiver_genes),
                    "baseline_r2": float(baseline["per_module_r2"][receiver_module_idx]),
                    "coupled_r2": float(coupled_r2[receiver_module_idx]),
                    "delta_r2": delta_r2,
                    "coeff": coeff,
                    "direction": coupling_direction(coeff, receiver_delta, delta_r2),
                }
            )
    candidates = [
        item
        for item in records
        if item["direction"] == "candidate_supports_condition_up_module"
        and item["sender_relation"] == "Condition-up"
        and item["receiver_relation"] == "Condition-up"
    ]
    if not candidates:
        candidates = records
    best = max(candidates, key=lambda item: item["delta_r2"])
    summary = {
        "condition": condition_value,
        "condition_col": condition_col,
        "receiver_cells": len(receiver_rows),
        "receiver_label_counts": Counter(receiver["labels"]),
        "sender_cells": len(sender["rows"]),
        "sender_label_counts": Counter(sender["labels"]),
        "baseline_overall_r2": baseline["overall_r2"],
        "baseline_overall_sign": baseline["overall_sign"],
        "best": best,
    }
    return records, summary


def write_outputs(output_dir: Path, summaries: list[dict], records: list[dict], module_lines: list[str]) -> None:
    records.sort(
        key=lambda item: (
            item["direction"] == "candidate_supports_condition_up_module",
            item["delta_r2"],
            abs(item["coeff"]),
        ),
        reverse=True,
    )
    summary_lines = [
        "condition\tcondition_col\treceiver_cells\treceiver_label_counts\tsender_cells\tsender_label_counts\t"
        "baseline_overall_r2\tbaseline_overall_sign\tbest_sender_module\tbest_receiver_module\t"
        "best_delta_r2\tbest_coeff\tbest_direction\tbest_sender_program\tbest_receiver_program\t"
        "best_sender_top_genes\tbest_receiver_top_genes"
    ]
    for item in summaries:
        best = item["best"]
        receiver_counts = ";".join(f"{key}:{value}" for key, value in sorted(item["receiver_label_counts"].items()))
        sender_counts = ";".join(f"{key}:{value}" for key, value in sorted(item["sender_label_counts"].items()))
        summary_lines.append(
            "\t".join(
                [
                    item["condition"],
                    item["condition_col"],
                    str(item["receiver_cells"]),
                    receiver_counts,
                    str(item["sender_cells"]),
                    sender_counts,
                    f"{item['baseline_overall_r2']:.6f}",
                    f"{item['baseline_overall_sign']:.6f}",
                    f"m_{best['sender_module']}",
                    f"m_{best['receiver_module']}",
                    f"{best['delta_r2']:.6f}",
                    f"{best['coeff']:.6f}",
                    best["direction"],
                    best["sender_program"],
                    best["receiver_program"],
                    best["sender_top_genes"],
                    best["receiver_top_genes"],
                ]
            )
        )

    pair_lines = [
        "condition\tcondition_col\tsender_module\tsender_delta\tsender_relation\tsender_program\t"
        "receiver_module\treceiver_delta\treceiver_relation\treceiver_program\tbaseline_r2\t"
        "coupled_r2\tdelta_r2\tcoeff\tdirection\tsender_top_genes\treceiver_top_genes"
    ]
    for item in records:
        pair_lines.append(
            "\t".join(
                [
                    item["condition"],
                    item["condition_col"],
                    f"m_{item['sender_module']}",
                    f"{item['sender_delta']:.6f}",
                    item["sender_relation"],
                    item["sender_program"],
                    f"m_{item['receiver_module']}",
                    f"{item['receiver_delta']:.6f}",
                    item["receiver_relation"],
                    item["receiver_program"],
                    f"{item['baseline_r2']:.6f}",
                    f"{item['coupled_r2']:.6f}",
                    f"{item['delta_r2']:.6f}",
                    f"{item['coeff']:.6f}",
                    item["direction"],
                    item["sender_top_genes"],
                    item["receiver_top_genes"],
                ]
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_text(output_dir / "condition_summary.tsv", "\n".join(summary_lines) + "\n")
    save_text(output_dir / "coupling_pairs.tsv", "\n".join(pair_lines) + "\n")
    save_text(output_dir / "module_reference.tsv", "\n".join(module_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="SCP2154 pre-tumor stromal-to-hepatocyte coupling")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--barcodes", type=Path, required=True)
    parser.add_argument("--comparisons", default="health:low_steatosis,health:cirrhotic")
    parser.add_argument("--max-cells-per-donor-condition", type=int, default=90)
    parser.add_argument("--max-cells-per-condition", type=int, default=1000)
    parser.add_argument("--hvg", type=int, default=800)
    parser.add_argument("--modules", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.03)
    parser.add_argument("--min-condition-cells", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=Path("results/scp2154_pretumor_coupling"))
    parser.add_argument("--random-state", type=int, default=7)
    args = parser.parse_args()

    metadata = read_metadata(args.metadata)
    comparisons = parse_comparisons(args.comparisons)
    sampled = {}
    all_rows = []
    for comp_idx, (condition_col, condition_value) in enumerate(comparisons):
        receiver_rows = make_condition_rows(
            metadata,
            RECEIVER,
            condition_col,
            condition_value,
            args.max_cells_per_donor_condition,
            args.max_cells_per_condition,
            args.random_state + 101 + comp_idx,
        )
        sender_rows = make_condition_rows(
            metadata,
            SENDER,
            condition_col,
            condition_value,
            args.max_cells_per_donor_condition,
            args.max_cells_per_condition,
            args.random_state + 201 + comp_idx,
        )
        receiver_condition_cells = sum(row[LABEL_COL] == condition_value for row in receiver_rows)
        sender_condition_cells = sum(row[LABEL_COL] == condition_value for row in sender_rows)
        if receiver_condition_cells < args.min_condition_cells or sender_condition_cells < args.min_condition_cells:
            print(
                f"Skipping {condition_col}:{condition_value}, condition cells receiver={receiver_condition_cells}, sender={sender_condition_cells}",
                flush=True,
            )
            continue
        sampled[(condition_col, condition_value)] = (receiver_rows, sender_rows)
        all_rows.extend(receiver_rows)
        all_rows.extend(sender_rows)
    unique_cells = list(dict.fromkeys(row["NAME"] for row in all_rows))
    print(f"Reading one shared matrix subset for {len(unique_cells)} unique cells...", flush=True)
    counts = read_10x_mtx_subset(args.matrix, args.features, args.barcodes, unique_cells)

    all_records = []
    summaries = []
    module_lines = [
        "condition\tcondition_col\tcell_type\tmodule\tcondition_mean\thealthy_mean\t"
        "delta_condition_minus_healthy\tcondition_relation\tputative_program\ttop_genes"
    ]
    print("SCP2154 pre-tumor coupling validation", flush=True)
    for comp_idx, ((condition_col, condition_value), (receiver_rows, sender_rows)) in enumerate(sampled.items()):
        receiver = fit_condition_representation(
            counts,
            receiver_rows,
            condition_value,
            args.modules,
            args.hvg,
            args.random_state + 301 + comp_idx,
        )
        sender = fit_condition_representation(
            counts,
            sender_rows,
            condition_value,
            args.modules,
            args.hvg,
            args.random_state + 401 + comp_idx,
        )
        receiver_shift = module_shift(receiver["rep"].module_activity, receiver["labels"], condition_value)
        sender_shift = module_shift(sender["rep"].module_activity, sender["labels"], condition_value)
        for cell_type, shifts, rep in [(RECEIVER, receiver_shift, receiver), (SENDER, sender_shift, sender)]:
            for module_idx, condition_mean, healthy_mean, delta in shifts:
                genes = top_gene_names(rep["top_genes"][module_idx])
                module_lines.append(
                    "\t".join(
                        [
                            condition_value,
                            condition_col,
                            cell_type,
                            f"m_{module_idx}",
                            f"{condition_mean:.6f}",
                            f"{healthy_mean:.6f}",
                            f"{delta:.6f}",
                            condition_relation(delta),
                            annotate_program(cell_type, genes),
                            ",".join(genes),
                        ]
                    )
                )
        train_idx, test_idx = split_indices_by_donor(receiver_rows, random_state=args.random_state)
        records, summary = evaluate_condition(
            condition_col,
            condition_value,
            sender,
            receiver,
            receiver_rows,
            sender_shift,
            receiver_shift,
            train_idx,
            test_idx,
            args.alpha,
        )
        all_records.extend(records)
        summaries.append(summary)
        best = summary["best"]
        print(
            f"  {condition_col}:{condition_value}: best {SENDER}_m{best['sender_module']} -> "
            f"{RECEIVER}_m{best['receiver_module']}, delta_r2={best['delta_r2']:+.4f}, "
            f"direction={best['direction']}",
            flush=True,
        )

    write_outputs(args.output_dir, summaries, all_records, module_lines)
    print("Saved:", flush=True)
    print(f"  {args.output_dir / 'condition_summary.tsv'}", flush=True)
    print(f"  {args.output_dir / 'coupling_pairs.tsv'}", flush=True)
    print(f"  {args.output_dir / 'module_reference.tsv'}", flush=True)


if __name__ == "__main__":
    main()
