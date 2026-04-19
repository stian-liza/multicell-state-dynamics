from __future__ import annotations

from collections import Counter
from html import escape
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
    read_10x_triplet_from_zip,
    read_metadata_table,
    select_highly_variable_genes,
    top_genes_per_module,
    velocity_r2_score,
    velocity_sign_agreement,
)


def make_train_test_split(n_items: int, test_fraction: float = 0.25, random_state: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n_items)
    test_size = max(1, int(round(n_items * test_fraction)))
    test_idx = np.sort(indices[:test_size])
    train_idx = np.sort(indices[test_size:])
    return train_idx, test_idx


def module_label_summary(module_activity: np.ndarray, labels: np.ndarray) -> list[tuple[int, float, float, float]]:
    summary: list[tuple[int, float, float, float]] = []
    for module_idx in range(module_activity.shape[1]):
        inf_mean = float(module_activity[labels == "InfMac", module_idx].mean())
        non_mean = float(module_activity[labels == "NonInfMac", module_idx].mean())
        summary.append((module_idx, inf_mean, non_mean, inf_mean - non_mean))
    return summary


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_summary_table(
    path: Path,
    module_summary: list[tuple[int, float, float, float]],
    module_genes: list[list[tuple[str, float]]],
) -> None:
    lines = ["module\tinf_mean\tnon_mean\tdelta_inf_minus_non\ttop_genes"]
    for module_idx, inf_mean, non_mean, delta in module_summary:
        genes = ",".join(gene for gene, _ in module_genes[module_idx])
        lines.append(f"m_{module_idx}\t{inf_mean:.6f}\t{non_mean:.6f}\t{delta:.6f}\t{genes}")
    save_text(path, "\n".join(lines) + "\n")


def save_pca_scatter_svg(path: Path, embedding: np.ndarray, labels: np.ndarray, pseudotime: np.ndarray) -> None:
    width = 900
    height = 680
    margin = 70
    colors = {"NonInfMac": "#2E7D32", "InfMac": "#C62828"}
    x = embedding[:, 0]
    y = embedding[:, 1]
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    def sx(val: float) -> float:
        return margin + (val - x_min) / max(x_max - x_min, 1e-8) * (width - 2 * margin)

    def sy(val: float) -> float:
        return height - margin - (val - y_min) / max(y_max - y_min, 1e-8) * (height - 2 * margin)

    circles = []
    order = np.argsort(pseudotime)
    for idx in order:
        label = str(labels[idx])
        fill = colors.get(label, "#666666")
        circles.append(
            f'<circle cx="{sx(float(x[idx])):.2f}" cy="{sy(float(y[idx])):.2f}" r="4.2" '
            f'fill="{fill}" fill-opacity="0.70" stroke="white" stroke-width="0.5" />'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#F8F6F1"/>
<text x="{margin}" y="38" font-size="26" font-family="Arial, Helvetica, sans-serif" fill="#1F2933">GSE185477 C41 macrophage PCA</text>
<text x="{margin}" y="62" font-size="15" font-family="Arial, Helvetica, sans-serif" fill="#52606D">Points colored by manual label; pseudotime oriented from NonInfMac to InfMac.</text>
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#7B8794" stroke-width="1.5"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#7B8794" stroke-width="1.5"/>
{''.join(circles)}
<rect x="{width - 220}" y="78" width="16" height="16" fill="#2E7D32"/><text x="{width - 196}" y="91" font-size="14" font-family="Arial, Helvetica, sans-serif" fill="#1F2933">NonInfMac</text>
<rect x="{width - 220}" y="104" width="16" height="16" fill="#C62828"/><text x="{width - 196}" y="117" font-size="14" font-family="Arial, Helvetica, sans-serif" fill="#1F2933">InfMac</text>
<text x="{width / 2:.1f}" y="{height - 18}" text-anchor="middle" font-size="14" font-family="Arial, Helvetica, sans-serif" fill="#52606D">PC1</text>
<text x="20" y="{height / 2:.1f}" transform="rotate(-90 20,{height / 2:.1f})" text-anchor="middle" font-size="14" font-family="Arial, Helvetica, sans-serif" fill="#52606D">PC2</text>
</svg>
"""
    save_text(path, svg)


def save_module_shift_svg(path: Path, module_summary: list[tuple[int, float, float, float]]) -> None:
    width = 900
    height = 560
    margin = 90
    bar_w = 110
    gap = 70
    values = [item[3] for item in module_summary]
    vmax = max(max(abs(v) for v in values), 1e-6)
    baseline = height / 2 + 40

    bars = []
    labels_svg = []
    for idx, (module_idx, inf_mean, non_mean, delta) in enumerate(module_summary):
        x = margin + idx * (bar_w + gap)
        scaled = (delta / vmax) * 170
        y = baseline - max(scaled, 0)
        h = abs(scaled)
        fill = "#C62828" if delta >= 0 else "#2E7D32"
        bars.append(f'<rect x="{x}" y="{y:.2f}" width="{bar_w}" height="{h:.2f}" fill="{fill}" rx="8" />')
        labels_svg.append(
            f'<text x="{x + bar_w/2:.1f}" y="{baseline + 26}" text-anchor="middle" font-size="16" '
            f'font-family="Arial, Helvetica, sans-serif" fill="#1F2933">m_{module_idx}</text>'
            f'<text x="{x + bar_w/2:.1f}" y="{y - 10 if delta >= 0 else y + h + 18:.1f}" text-anchor="middle" font-size="13" '
            f'font-family="Arial, Helvetica, sans-serif" fill="#52606D">{delta:+.3f}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#F8F6F1"/>
<text x="{margin}" y="38" font-size="26" font-family="Arial, Helvetica, sans-serif" fill="#1F2933">Module Activity Shifts</text>
<text x="{margin}" y="62" font-size="15" font-family="Arial, Helvetica, sans-serif" fill="#52606D">Bar height shows InfMac mean minus NonInfMac mean for each module.</text>
<line x1="{margin - 20}" y1="{baseline}" x2="{width - margin + 20}" y2="{baseline}" stroke="#7B8794" stroke-width="1.5"/>
{''.join(bars)}
{''.join(labels_svg)}
<rect x="{width - 260}" y="86" width="16" height="16" fill="#C62828"/><text x="{width - 236}" y="99" font-size="14" font-family="Arial, Helvetica, sans-serif" fill="#1F2933">Higher in InfMac</text>
<rect x="{width - 260}" y="112" width="16" height="16" fill="#2E7D32"/><text x="{width - 236}" y="125" font-size="14" font-family="Arial, Helvetica, sans-serif" fill="#1F2933">Higher in NonInfMac</text>
</svg>
"""
    save_text(path, svg)


def save_top_gene_svg(path: Path, module_genes: list[list[tuple[str, float]]]) -> None:
    width = 1100
    row_h = 32
    header_h = 90
    module_gap = 36
    rows_per_module = max(len(genes) for genes in module_genes)
    height = header_h + len(module_genes) * (rows_per_module * row_h + module_gap) + 40
    max_weight = max(weight for genes in module_genes for _, weight in genes)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#F8F6F1"/>',
        '<text x="60" y="38" font-size="26" font-family="Arial, Helvetica, sans-serif" fill="#1F2933">Top Genes Per Module</text>',
        '<text x="60" y="62" font-size="15" font-family="Arial, Helvetica, sans-serif" fill="#52606D">Longer bars indicate higher module weights within the HVG subset.</text>',
    ]
    y0 = header_h
    palette = ["#355C7D", "#C06C84", "#6C5B7B", "#F67280"]
    for module_idx, genes in enumerate(module_genes):
        y_start = y0 + module_idx * (rows_per_module * row_h + module_gap)
        parts.append(
            f'<text x="60" y="{y_start - 10}" font-size="18" font-family="Arial, Helvetica, sans-serif" fill="#1F2933">m_{module_idx}</text>'
        )
        for row_idx, (gene, weight) in enumerate(genes):
            y = y_start + row_idx * row_h
            bar_len = 420 * (weight / max(max_weight, 1e-8))
            color = palette[module_idx % len(palette)]
            parts.append(f'<text x="60" y="{y + 18}" font-size="14" font-family="Arial, Helvetica, sans-serif" fill="#1F2933">{escape(gene)}</text>')
            parts.append(f'<rect x="210" y="{y + 4}" width="{bar_len:.1f}" height="18" rx="4" fill="{color}" fill-opacity="0.9"/>')
            parts.append(f'<text x="{210 + bar_len + 10:.1f}" y="{y + 18}" font-size="12" font-family="Arial, Helvetica, sans-serif" fill="#52606D">{weight:.3f}</text>')
    parts.append("</svg>")
    save_text(path, "".join(parts))


def main() -> None:
    metadata_path = Path("data/raw/gse185477/GSE185477_Final_Metadata.txt.gz")
    matrix_path = Path("data/raw/gse185477/GSE185477_GSM3178784_C41_SC_raw_counts.zip")
    if not metadata_path.exists():
        raise SystemExit(f"Missing metadata file: {metadata_path}")
    if not matrix_path.exists():
        raise SystemExit(f"Missing count zip: {matrix_path}")

    rows = read_metadata_table(metadata_path)
    c41_mac = [row for row in rows if row["sample"] == "C41" and row["Subcluster_Group"] == "Macrophage"]
    if not c41_mac:
        raise SystemExit("No C41 macrophage cells found in metadata.")

    label_by_barcode = {row["cell_barcode"]: row["Manual_Annotation"] for row in c41_mac}
    counts = read_10x_triplet_from_zip(matrix_path)

    normalized_barcodes = np.array([barcode.split("-")[0] for barcode in counts.cell_barcodes], dtype=object)
    keep_mask = np.array([barcode in label_by_barcode for barcode in normalized_barcodes], dtype=bool)
    if int(np.sum(keep_mask)) == 0:
        raise SystemExit("No overlapping macrophage barcodes between metadata and count matrix.")

    matrix = counts.matrix[keep_mask]
    barcodes = normalized_barcodes[keep_mask]
    labels = np.array([label_by_barcode[barcode] for barcode in barcodes], dtype=object)

    normalized = log1p_library_normalize(matrix)
    hvg_matrix, hvg_names = select_highly_variable_genes(normalized, counts.gene_names, top_k=1500)
    rep = fit_module_representation(hvg_matrix, n_modules=4, random_state=7)
    embedding = pca_embedding(hvg_matrix, n_components=2)
    pseudotime = pseudotime_from_embedding(embedding, axis=0)
    pseudotime = orient_pseudotime_by_labels(pseudotime, labels, low_label="NonInfMac", high_label="InfMac")
    local_velocity = local_direction_from_pseudotime(rep.module_activity, pseudotime, k_neighbors=15)
    module_genes = top_genes_per_module(rep.module_weights, hvg_names, top_k=10)
    module_summary = module_label_summary(rep.module_activity, labels)

    train_idx, test_idx = make_train_test_split(len(barcodes), test_fraction=0.25, random_state=7)
    model = fit_population_dynamics(
        module_activity=rep.module_activity[train_idx],
        state_embedding=embedding[train_idx],
        module_velocity=local_velocity[train_idx],
        alpha=0.03,
    )

    pred_train = model.predict_velocity(rep.module_activity[train_idx], embedding[train_idx])
    pred_test = model.predict_velocity(rep.module_activity[test_idx], embedding[test_idx])
    output_dir = Path("results/gse185477_demo")
    save_pca_scatter_svg(output_dir / "c41_macrophage_pca.svg", embedding, labels, pseudotime)
    save_module_shift_svg(output_dir / "module_activity_shifts.svg", module_summary)
    save_top_gene_svg(output_dir / "module_top_genes.svg", module_genes)
    write_summary_table(output_dir / "module_summary.tsv", module_summary, module_genes)

    print("GSE185477 real-data prototype")
    print("Cells used:", len(barcodes))
    print("Label counts:", dict(Counter(labels)))
    print("Genes after HVG filter:", len(hvg_names))
    print("Reconstruction error:", round(rep.reconstruction_error, 4))
    print("Training R^2:", round(model.train_r2, 4))
    print("Train sign agreement:", round(velocity_sign_agreement(local_velocity[train_idx], pred_train), 4))
    print("Test R^2:", round(velocity_r2_score(local_velocity[test_idx], pred_test), 4))
    print("Test sign agreement:", round(velocity_sign_agreement(local_velocity[test_idx], pred_test), 4))
    print("Module activity shifts (InfMac minus NonInfMac):")
    for module_idx, inf_mean, non_mean, delta in module_summary:
        print(
            f"  m_{module_idx}: inf_mean={inf_mean:.4f}, non_mean={non_mean:.4f}, "
            f"delta={delta:+.4f}"
        )
    print("Top genes per module:")
    for module_idx, gene_list in enumerate(module_genes):
        genes = ", ".join(gene for gene, _ in gene_list)
        print(f"  m_{module_idx}: {genes}")
    print("Top inferred edges:")
    for source, target, weight in model.top_edges(top_k=10):
        print(f"  {source:>6} -> {target:<8} {weight:+.3f}")
    print("Saved figures:")
    print(f"  {output_dir / 'c41_macrophage_pca.svg'}")
    print(f"  {output_dir / 'module_activity_shifts.svg'}")
    print(f"  {output_dir / 'module_top_genes.svg'}")
    print(f"  {output_dir / 'module_summary.tsv'}")


if __name__ == "__main__":
    main()
