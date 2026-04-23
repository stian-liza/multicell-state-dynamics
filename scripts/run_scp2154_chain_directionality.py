from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_scp2154_fixed_signature_tumor_validation import empirical_p_value, safe_corr
from run_scp2154_stagewise_bidirectional_network import (
    all_signatures,
    celltype_matrix,
    evaluate_pair,
    parse_stages,
    sample_celltype_stage_rows,
    stage_node_tables,
)
from run_scp2154_coupling_driver_scan import unique_rows
from run_scp2154_phenotype_baseline import read_metadata
from run_scp2154_stromal_to_hepatocyte_coupling import save_text
from multicell_dynamics import log1p_library_normalize, read_10x_mtx_subset


DEFAULT_CHAIN = [
    ("hop_1", "cirrhotic", "Myeloid.c1qc_macrophage", "Myeloid.inflammatory_monocyte"),
    ("hop_2", "Tumor", "Myeloid.inflammatory_monocyte", "Hepatocyte.secretory_stress"),
    ("hop_3", "Tumor", "Hepatocyte.secretory_stress", "Hepatocyte.malignant_like"),
]

DEFAULT_TRIPLETS = [
    ("tumor_inflammatory_to_malignant", "Tumor", "Myeloid.inflammatory_monocyte", "Hepatocyte.secretory_stress", "Hepatocyte.malignant_like"),
]


def ridge_fit_predict_multi(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = np.maximum(x_train.std(axis=0, keepdims=True), 1e-8)
    x_train_z = (x_train - x_mean) / x_std
    x_test_z = (x_test - x_mean) / x_std
    y_mean = float(y_train.mean())
    y_centered = y_train - y_mean
    gram = x_train_z.T @ x_train_z + alpha * np.eye(x_train_z.shape[1])
    weights = np.linalg.solve(gram, x_train_z.T @ y_centered)
    pred = x_test_z @ weights + y_mean
    return pred, weights, y_mean


def loo_prediction_test_multi(x: np.ndarray, y: np.ndarray, alpha: float) -> dict:
    pred = np.zeros_like(y, dtype=float)
    baseline_pred = np.zeros_like(y, dtype=float)
    coeffs = np.zeros((len(y), x.shape[1]), dtype=float)
    for heldout_idx in range(len(y)):
        train_idx = np.array([idx for idx in range(len(y)) if idx != heldout_idx], dtype=int)
        test_idx = np.array([heldout_idx], dtype=int)
        fold_pred, fold_weights, baseline = ridge_fit_predict_multi(x[train_idx], y[train_idx], x[test_idx], alpha)
        pred[test_idx] = fold_pred
        baseline_pred[test_idx] = baseline
        coeffs[heldout_idx] = fold_weights
    denom = float(np.sum((y - y.mean()) ** 2))
    if denom <= 1e-8:
        model_r2 = float("nan")
        baseline_r2 = float("nan")
    else:
        model_r2 = 1.0 - float(np.sum((y - pred) ** 2)) / denom
        baseline_r2 = 1.0 - float(np.sum((y - baseline_pred) ** 2)) / denom
    return {
        "loo_r2": model_r2,
        "loo_baseline_r2": baseline_r2,
        "loo_delta_r2": model_r2 - baseline_r2,
        "mean_coeffs": coeffs.mean(axis=0),
    }


def parse_chain(raw: str) -> list[tuple[str, str, str, str]]:
    if not raw.strip():
        return list(DEFAULT_CHAIN)
    out = []
    for idx, item in enumerate(raw.split(";"), start=1):
        parts = [part.strip() for part in item.split("|")]
        if len(parts) != 3:
            raise ValueError("Each chain hop must be formatted as stage|source|target")
        out.append((f"hop_{idx}", parts[0], parts[1], parts[2]))
    return out


def parse_triplets(raw: str) -> list[tuple[str, str, str, str, str]]:
    if not raw.strip():
        return list(DEFAULT_TRIPLETS)
    out = []
    for idx, item in enumerate(raw.split(";"), start=1):
        parts = [part.strip() for part in item.split("|")]
        if len(parts) != 4:
            raise ValueError("Each triplet must be formatted as stage|source|mediator|target")
        out.append((f"triplet_{idx}", parts[0], parts[1], parts[2], parts[3]))
    return out


def build_stage_node_values(
    metadata_path: Path,
    matrix_path: Path,
    features_path: Path,
    barcodes_path: Path,
    stages: list[dict],
    max_cells_per_donor_stage: int,
    max_cells_per_stage: int,
    min_cells_per_donor_node: int,
    random_state: int,
) -> dict[str, dict[str, dict[str, float]]]:
    metadata = read_metadata(metadata_path)
    signatures = all_signatures()
    rows_by_cell_type = {}
    sampled_rows = []
    for idx, cell_type in enumerate(signatures):
        rows = sample_celltype_stage_rows(
            metadata,
            cell_type,
            stages,
            max_cells_per_donor_stage,
            max_cells_per_stage,
            random_state + idx * 1000,
        )
        rows_by_cell_type[cell_type] = rows
        sampled_rows.extend(rows)
    sampled_rows = unique_rows(sampled_rows)
    print(f"Reading one shared matrix subset for {len(sampled_rows)} sampled cells...", flush=True)
    counts = read_10x_mtx_subset(matrix_path, features_path, barcodes_path, [row["NAME"] for row in sampled_rows])

    scores_by_node = {}
    kept_rows_by_cell_type = {}
    for cell_type, rows in rows_by_cell_type.items():
        matrix, kept_rows = celltype_matrix(counts, rows)
        kept_rows_by_cell_type[cell_type] = kept_rows
        normalized = log1p_library_normalize(matrix)
        from run_scp2154_fixed_signature_tumor_validation import signature_scores

        scores, _ = signature_scores(normalized, counts.gene_names, signatures[cell_type])
        for signature, values in scores.items():
            scores_by_node[f"{cell_type}.{signature}"] = values

    _, stage_node_values = stage_node_tables(
        kept_rows_by_cell_type,
        scores_by_node,
        stages,
        min_cells_per_donor_node,
    )
    return stage_node_values


def evaluate_hops(
    stage_node_values: dict[str, dict[str, dict[str, float]]],
    chain: list[tuple[str, str, str, str]],
    alpha: float,
    permutations: int,
    random_state: int,
) -> list[dict]:
    """Evaluate each pre-specified hop in both directions.

    Each hop reports:
    - forward_delta_r2 for source -> target
    - reverse_delta_r2 for target -> source
    - winner and winner_p based on the stronger held-out direction
    """
    rng = np.random.default_rng(random_state)
    records = []
    for hop_name, stage, source, target in chain:
        node_values = stage_node_values.get(stage, {})
        if source not in node_values or target not in node_values:
            records.append(
                {
                    "hop": hop_name,
                    "stage": stage,
                    "source": source,
                    "target": target,
                    "status": "missing_node",
                }
            )
            continue
        forward = evaluate_pair(node_values[source], node_values[target], source, target, alpha, permutations, rng)
        reverse = evaluate_pair(node_values[target], node_values[source], target, source, alpha, permutations, rng)
        if forward is None or reverse is None:
            records.append(
                {
                    "hop": hop_name,
                    "stage": stage,
                    "source": source,
                    "target": target,
                    "status": "insufficient_donors",
                }
            )
            continue
        if forward["loo_delta_r2"] > reverse["loo_delta_r2"]:
            winner = forward["source"] + "->" + forward["target"]
            winner_p = forward["empirical_p"]
        else:
            winner = reverse["source"] + "->" + reverse["target"]
            winner_p = reverse["empirical_p"]
        records.append(
            {
                "hop": hop_name,
                "stage": stage,
                "source": source,
                "target": target,
                "status": "ok",
                "n_donors": forward["n_donors"],
                "forward_delta_r2": forward["loo_delta_r2"],
                "reverse_delta_r2": reverse["loo_delta_r2"],
                "forward_p": forward["empirical_p"],
                "reverse_p": reverse["empirical_p"],
                "forward_coeff": forward["mean_coeff"],
                "reverse_coeff": reverse["mean_coeff"],
                "forward_corr": forward["pearson_r"],
                "reverse_corr": reverse["pearson_r"],
                "winner": winner,
                "winner_p": winner_p,
                "direction_margin": forward["loo_delta_r2"] - reverse["loo_delta_r2"],
            }
        )
    return records


def evaluate_triplets(
    stage_node_values: dict[str, dict[str, dict[str, float]]],
    triplets: list[tuple[str, str, str, str, str]],
    alpha: float,
    permutations: int,
    random_state: int,
) -> list[dict]:
    """Check whether a middle node mediates a source -> target relation.

    For source A, mediator B, target C:
    - source-only model:      C ~ A
    - mediator-only model:    C ~ B
    - joint model:            C ~ A + B

    The key reported quantity is joint_gain_over_source:
        Delta = R2_joint - R2_source_only

    A permutation p-value asks whether adding the mediator helps more than
    expected by chance.
    """
    rng = np.random.default_rng(random_state)
    records = []
    for triplet_name, stage, source, mediator, target in triplets:
        node_values = stage_node_values.get(stage, {})
        required = [source, mediator, target]
        if any(node not in node_values for node in required):
            records.append(
                {
                    "triplet": triplet_name,
                    "stage": stage,
                    "source": source,
                    "mediator": mediator,
                    "target": target,
                    "status": "missing_node",
                }
            )
            continue
        donors = sorted(set(node_values[source]) & set(node_values[mediator]) & set(node_values[target]))
        if len(donors) < 5:
            records.append(
                {
                    "triplet": triplet_name,
                    "stage": stage,
                    "source": source,
                    "mediator": mediator,
                    "target": target,
                    "status": "insufficient_donors",
                }
            )
            continue
        source_values = np.array([node_values[source][donor] for donor in donors], dtype=float)
        mediator_values = np.array([node_values[mediator][donor] for donor in donors], dtype=float)
        target_values = np.array([node_values[target][donor] for donor in donors], dtype=float)

        source_only = loo_prediction_test_multi(source_values[:, None], target_values, alpha)
        mediator_only = loo_prediction_test_multi(mediator_values[:, None], target_values, alpha)
        joint = loo_prediction_test_multi(np.column_stack([source_values, mediator_values]), target_values, alpha)

        null_joint_gain = []
        for _ in range(permutations):
            permuted_mediator = mediator_values[rng.permutation(len(mediator_values))]
            permuted_joint = loo_prediction_test_multi(np.column_stack([source_values, permuted_mediator]), target_values, alpha)
            null_joint_gain.append(float(permuted_joint["loo_delta_r2"] - source_only["loo_delta_r2"]))
        null_joint_gain = np.array(null_joint_gain, dtype=float)
        joint_gain = float(joint["loo_delta_r2"] - source_only["loo_delta_r2"])
        source_coeff_only = float(source_only["mean_coeffs"][0])
        source_coeff_joint = float(joint["mean_coeffs"][0])
        mediator_coeff_joint = float(joint["mean_coeffs"][1])
        if joint_gain > 0 and abs(source_coeff_joint) < abs(source_coeff_only):
            mediation_call = "partial_mediation_supported"
        elif joint_gain > 0:
            mediation_call = "joint_model_better_but_source_not_reduced"
        else:
            mediation_call = "no_mediation_gain"
        records.append(
            {
                "triplet": triplet_name,
                "stage": stage,
                "source": source,
                "mediator": mediator,
                "target": target,
                "status": "ok",
                "n_donors": len(donors),
                "source_only_delta_r2": float(source_only["loo_delta_r2"]),
                "mediator_only_delta_r2": float(mediator_only["loo_delta_r2"]),
                "joint_delta_r2": float(joint["loo_delta_r2"]),
                "joint_gain_over_source": joint_gain,
                "joint_gain_p": empirical_p_value(joint_gain, null_joint_gain),
                "source_coeff_only": source_coeff_only,
                "source_coeff_joint": source_coeff_joint,
                "mediator_coeff_joint": mediator_coeff_joint,
                "source_to_mediator_corr": safe_corr(source_values, mediator_values),
                "mediator_to_target_corr": safe_corr(mediator_values, target_values),
                "source_to_target_corr": safe_corr(source_values, target_values),
                "mediation_call": mediation_call,
            }
        )
    return records


def write_outputs(output_dir: Path, hop_records: list[dict], triplet_records: list[dict]) -> None:
    hop_lines = [
        "hop\tstage\tsource\ttarget\tstatus\tn_donors\tforward_delta_r2\treverse_delta_r2\tforward_p\treverse_p\t"
        "forward_coeff\treverse_coeff\tforward_corr\treverse_corr\twinner\twinner_p\tdirection_margin"
    ]
    for record in hop_records:
        if record["status"] != "ok":
            hop_lines.append(
                "\t".join([record["hop"], record["stage"], record["source"], record["target"], record["status"]])
            )
            continue
        hop_lines.append(
            "\t".join(
                [
                    record["hop"],
                    record["stage"],
                    record["source"],
                    record["target"],
                    record["status"],
                    str(record["n_donors"]),
                    f"{record['forward_delta_r2']:.6f}",
                    f"{record['reverse_delta_r2']:.6f}",
                    f"{record['forward_p']:.6f}",
                    f"{record['reverse_p']:.6f}",
                    f"{record['forward_coeff']:.6f}",
                    f"{record['reverse_coeff']:.6f}",
                    f"{record['forward_corr']:.6f}",
                    f"{record['reverse_corr']:.6f}",
                    record["winner"],
                    f"{record['winner_p']:.6f}",
                    f"{record['direction_margin']:.6f}",
                ]
            )
        )

    triplet_lines = [
        "triplet\tstage\tsource\tmediator\ttarget\tstatus\tn_donors\tsource_only_delta_r2\tmediator_only_delta_r2\t"
        "joint_delta_r2\tjoint_gain_over_source\tjoint_gain_p\tsource_coeff_only\tsource_coeff_joint\t"
        "mediator_coeff_joint\tsource_to_mediator_corr\tmediator_to_target_corr\tsource_to_target_corr\tmediation_call"
    ]
    for record in triplet_records:
        if record["status"] != "ok":
            triplet_lines.append(
                "\t".join(
                    [record["triplet"], record["stage"], record["source"], record["mediator"], record["target"], record["status"]]
                )
            )
            continue
        triplet_lines.append(
            "\t".join(
                [
                    record["triplet"],
                    record["stage"],
                    record["source"],
                    record["mediator"],
                    record["target"],
                    record["status"],
                    str(record["n_donors"]),
                    f"{record['source_only_delta_r2']:.6f}",
                    f"{record['mediator_only_delta_r2']:.6f}",
                    f"{record['joint_delta_r2']:.6f}",
                    f"{record['joint_gain_over_source']:.6f}",
                    f"{record['joint_gain_p']:.6f}",
                    f"{record['source_coeff_only']:.6f}",
                    f"{record['source_coeff_joint']:.6f}",
                    f"{record['mediator_coeff_joint']:.6f}",
                    f"{record['source_to_mediator_corr']:.6f}",
                    f"{record['mediator_to_target_corr']:.6f}",
                    f"{record['source_to_target_corr']:.6f}",
                    record["mediation_call"],
                ]
            )
        )

    summary_lines = ["metric\tvalue"]
    ok_hops = [record for record in hop_records if record["status"] == "ok"]
    ok_triplets = [record for record in triplet_records if record["status"] == "ok"]
    summary_lines.append(f"hops_tested\t{len(ok_hops)}")
    for record in ok_hops:
        summary_lines.append(
            f"{record['hop']}\t{record['stage']}:{record['winner']}:winner_p={record['winner_p']:.4f}:margin={record['direction_margin']:.4f}"
        )
    summary_lines.append(f"triplets_tested\t{len(ok_triplets)}")
    for record in ok_triplets:
        summary_lines.append(
            f"{record['triplet']}\t{record['stage']}:{record['mediation_call']}:joint_gain={record['joint_gain_over_source']:.4f}:p={record['joint_gain_p']:.4f}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_text(output_dir / "chain_hops.tsv", "\n".join(hop_lines) + "\n")
    save_text(output_dir / "chain_triplets.tsv", "\n".join(triplet_lines) + "\n")
    save_text(output_dir / "chain_summary.tsv", "\n".join(summary_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Directionality and same-stage mediation checks for SCP2154 chain candidates")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--barcodes", type=Path, required=True)
    parser.add_argument("--stages", default="low_steatosis,NAFLD,alcohol,cirrhotic,Tumor")
    parser.add_argument("--chain", default="")
    parser.add_argument("--triplets", default="")
    parser.add_argument("--max-cells-per-donor-stage", type=int, default=80)
    parser.add_argument("--max-cells-per-stage", type=int, default=700)
    parser.add_argument("--min-cells-per-donor-node", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--permutations", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("results/scp2154_chain_directionality"))
    args = parser.parse_args()

    stages = parse_stages(args.stages)
    chain = parse_chain(args.chain)
    triplets = parse_triplets(args.triplets)
    stage_node_values = build_stage_node_values(
        args.metadata,
        args.matrix,
        args.features,
        args.barcodes,
        stages,
        args.max_cells_per_donor_stage,
        args.max_cells_per_stage,
        args.min_cells_per_donor_node,
        args.random_state,
    )
    hop_records = evaluate_hops(stage_node_values, chain, args.alpha, args.permutations, args.random_state + 11)
    triplet_records = evaluate_triplets(stage_node_values, triplets, args.alpha, args.permutations, args.random_state + 111)
    write_outputs(args.output_dir, hop_records, triplet_records)

    print("SCP2154 chain directionality", flush=True)
    for record in hop_records:
        if record["status"] != "ok":
            print(f"  {record['hop']}: {record['status']}", flush=True)
            continue
        print(
            f"  {record['hop']} {record['stage']}: {record['winner']} "
            f"forward={record['forward_delta_r2']:+.4f} reverse={record['reverse_delta_r2']:+.4f} "
            f"winner_p={record['winner_p']:.4f}",
            flush=True,
        )
    for record in triplet_records:
        if record["status"] != "ok":
            print(f"  {record['triplet']}: {record['status']}", flush=True)
            continue
        print(
            f"  {record['triplet']} {record['stage']}: {record['mediation_call']} "
            f"joint_gain={record['joint_gain_over_source']:+.4f} p={record['joint_gain_p']:.4f}",
            flush=True,
        )
    print("Saved:", flush=True)
    print(f"  {args.output_dir / 'chain_hops.tsv'}", flush=True)
    print(f"  {args.output_dir / 'chain_triplets.tsv'}", flush=True)
    print(f"  {args.output_dir / 'chain_summary.tsv'}", flush=True)


if __name__ == "__main__":
    main()
