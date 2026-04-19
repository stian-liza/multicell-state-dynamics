from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from run_scp2154_stromal_to_hepatocyte_coupling import save_text


CELL_TYPE_COLORS = {
    "Stromal": "#d95f02",
    "Myeloid": "#1b9e77",
    "Endothelial": "#7570b3",
    "TNKcell": "#e7298a",
    "Bcell": "#66a61e",
    "Hepatocyte": "#e6ab02",
}


def load_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def safe_float(raw: str, default: float = 0.0) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def sanitize_id(raw: str) -> str:
    return (
        raw.replace(".", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("+", "plus")
        .replace(" ", "_")
    )


def short_label(node_id: str) -> str:
    cell_type, signature = node_id.split(".", 1)
    return f"{cell_type}\\n{signature}"


def choose_nodes(
    node_rows: list[dict[str, str]],
    stages: list[str],
    node_abs_delta_min: float,
) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    nodes_by_id: dict[str, dict] = {}
    by_stage: dict[str, list[dict]] = defaultdict(list)
    for row in node_rows:
        stage = row["stage"]
        if stage not in stages:
            continue
        abs_delta = safe_float(row["abs_delta"])
        node_id = row["node_id"]
        record = {
            "node_id": node_id,
            "stage": stage,
            "cell_type": row["cell_type"],
            "signature": row["signature"],
            "delta": safe_float(row["delta"]),
            "abs_delta": abs_delta,
            "first_altered_stage": row["first_altered_stage"],
            "passes_node_filter": abs_delta >= node_abs_delta_min,
        }
        by_stage[stage].append(record)
        current = nodes_by_id.get(node_id)
        if current is None or abs_delta > current["abs_delta"]:
            nodes_by_id[node_id] = record
    return nodes_by_id, by_stage


def choose_edges(
    edge_rows: list[dict[str, str]],
    stages: list[str],
    edge_min_delta: float,
    edge_max_p: float,
    max_abs_delta: float,
    top_edges_per_stage: int,
) -> dict[str, list[dict]]:
    by_stage: dict[str, list[dict]] = defaultdict(list)
    for row in edge_rows:
        stage = row["stage"]
        if stage not in stages:
            continue
        delta = safe_float(row["selected_delta_r2"])
        p_value = safe_float(row["selected_empirical_p"], default=1.0)
        if delta < edge_min_delta or p_value > edge_max_p or abs(delta) > max_abs_delta:
            continue
        by_stage[stage].append(
            {
                "stage": stage,
                "source": row["chain_source"],
                "target": row["chain_target"],
                "selected_delta_r2": delta,
                "selected_empirical_p": p_value,
                "selected_coeff": safe_float(row["selected_coeff"]),
                "stage_consistency": row["stage_consistency"],
                "selected_edge_stage_count": int(float(row["selected_edge_stage_count"] or 0)),
            }
        )
    for stage, edges in by_stage.items():
        edges.sort(
            key=lambda item: (
                item["stage_consistency"] == "recurrent",
                item["selected_delta_r2"],
                abs(item["selected_coeff"]),
            ),
            reverse=True,
        )
        by_stage[stage] = edges[:top_edges_per_stage]
    return by_stage


def build_mermaid(
    stage_order: list[str],
    nodes_by_id: dict[str, dict],
    stage_nodes: dict[str, list[dict]],
    stage_edges: dict[str, list[dict]],
) -> str:
    lines = ["flowchart LR"]
    used_nodes = set()

    for stage in stage_order:
        lines.append(f'  subgraph {sanitize_id(stage)}["{stage}"]')
        stage_node_ids = set()
        for edge in stage_edges.get(stage, []):
            stage_node_ids.add(edge["source"])
            stage_node_ids.add(edge["target"])
        ranked_nodes = sorted(
            [node for node in stage_nodes.get(stage, []) if node["node_id"] in stage_node_ids],
            key=lambda item: item["abs_delta"],
            reverse=True,
        )
        for node in ranked_nodes:
            node_key = sanitize_id(f"{stage}_{node['node_id']}")
            stage_node_ids.add(node["node_id"])
            used_nodes.add((stage, node["node_id"]))
            label = short_label(node["node_id"])
            lines.append(f'    {node_key}["{label}"]')
        lines.append("  end")

    for stage in stage_order:
        for edge in stage_edges.get(stage, []):
            source_key = sanitize_id(f"{stage}_{edge['source']}")
            target_key = sanitize_id(f"{stage}_{edge['target']}")
            label = f"{edge['selected_delta_r2']:.2f}|p={edge['selected_empirical_p']:.2f}"
            lines.append(f"  {source_key} -->|{label}| {target_key}")

    for stage, node_id in used_nodes:
        node = nodes_by_id[node_id]
        color = CELL_TYPE_COLORS.get(node["cell_type"], "#999999")
        node_key = sanitize_id(f"{stage}_{node_id}")
        lines.append(f"  style {node_key} fill:{color},stroke:#333,stroke-width:1px,color:#111")
    return "\n".join(lines) + "\n"


def build_stage_summary(stage_order: list[str], stage_nodes: dict[str, list[dict]], stage_edges: dict[str, list[dict]]) -> str:
    lines = ["stage\tnodes_shown\tedges_shown\ttop_nodes\ttop_edges"]
    for stage in stage_order:
        nodes = sorted(stage_nodes.get(stage, []), key=lambda item: item["abs_delta"], reverse=True)
        edges = stage_edges.get(stage, [])
        shown_node_ids = {edge["source"] for edge in edges} | {edge["target"] for edge in edges}
        top_nodes = ",".join(
            f"{node['node_id']}({node['delta']:+.2f})"
            for node in nodes
            if node["node_id"] in shown_node_ids
        )
        top_edges = ",".join(
            f"{edge['source']}->{edge['target']}({edge['selected_delta_r2']:.2f})"
            for edge in edges
        )
        lines.append(
            "\t".join(
                [
                    stage,
                    str(len(shown_node_ids)),
                    str(len(edges)),
                    top_nodes,
                    top_edges,
                ]
            )
        )
    return "\n".join(lines) + "\n"


def build_node_table(stage_order: list[str], stage_edges: dict[str, list[dict]], nodes_by_id: dict[str, dict]) -> str:
    lines = ["stage\tnode_id\tcell_type\tsignature\tdelta\tabs_delta\tfirst_altered_stage"]
    for stage in stage_order:
        node_ids = {edge["source"] for edge in stage_edges.get(stage, [])} | {edge["target"] for edge in stage_edges.get(stage, [])}
        for node_id in sorted(node_ids):
            node = nodes_by_id[node_id]
            lines.append(
                "\t".join(
                    [
                        stage,
                        node_id,
                        node["cell_type"],
                        node["signature"],
                        f"{node['delta']:.6f}",
                        f"{node['abs_delta']:.6f}",
                        node["first_altered_stage"],
                    ]
                )
            )
    return "\n".join(lines) + "\n"


def build_edge_table(stage_order: list[str], stage_edges: dict[str, list[dict]]) -> str:
    lines = ["stage\tsource\ttarget\tselected_delta_r2\tselected_empirical_p\tselected_coeff\tstage_consistency\tselected_edge_stage_count"]
    for stage in stage_order:
        for edge in stage_edges.get(stage, []):
            lines.append(
                "\t".join(
                    [
                        stage,
                        edge["source"],
                        edge["target"],
                        f"{edge['selected_delta_r2']:.6f}",
                        f"{edge['selected_empirical_p']:.6f}",
                        f"{edge['selected_coeff']:.6f}",
                        edge["stage_consistency"],
                        str(edge["selected_edge_stage_count"]),
                    ]
                )
            )
    return "\n".join(lines) + "\n"


def build_markdown(graph_title: str, mermaid: str, stage_summary_path: Path, node_table_path: Path, edge_table_path: Path) -> str:
    return (
        f"# {graph_title}\n\n"
        "```mermaid\n"
        f"{mermaid}"
        "```\n\n"
        f"- Stage summary: `{stage_summary_path}`\n"
        f"- Nodes shown: `{node_table_path}`\n"
        f"- Edges shown: `{edge_table_path}`\n"
    )


def xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def build_svg(stage_order: list[str], nodes_by_id: dict[str, dict], stage_edges: dict[str, list[dict]]) -> str:
    stage_node_ids = {
        stage: sorted(
            {edge["source"] for edge in stage_edges.get(stage, [])} | {edge["target"] for edge in stage_edges.get(stage, [])},
            key=lambda node_id: (
                nodes_by_id[node_id]["cell_type"],
                -nodes_by_id[node_id]["abs_delta"],
                node_id,
            ),
        )
        for stage in stage_order
    }
    column_width = 360
    margin_x = 40
    margin_y = 70
    node_width = 250
    node_height = 40
    stage_gap = 32
    node_gap = 18
    legend_height = 150
    max_nodes = max((len(node_ids) for node_ids in stage_node_ids.values()), default=1)
    height = margin_y * 2 + legend_height + max_nodes * (node_height + node_gap) + 80
    width = margin_x * 2 + len(stage_order) * column_width

    node_positions: dict[tuple[str, str], tuple[float, float]] = {}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs>',
        '<marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">',
        '<path d="M0,0 L0,6 L9,3 z" fill="#666"/>',
        '</marker>',
        '</defs>',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fcfcfb"/>',
        '<text x="40" y="34" font-size="24" font-family="Arial, Helvetica, sans-serif" fill="#222">SCP2154 Stagewise Network</text>',
        '<text x="40" y="56" font-size="13" font-family="Arial, Helvetica, sans-serif" fill="#555">Filtered directional edges by stage; node color = cell type, edge label = delta R^2 / p-value</text>',
    ]

    legend_x = 40
    legend_y = 76
    for idx, (cell_type, color) in enumerate(CELL_TYPE_COLORS.items()):
        x = legend_x + idx * 125
        parts.append(f'<rect x="{x}" y="{legend_y}" width="18" height="18" rx="3" fill="{color}" stroke="#333" stroke-width="1"/>')
        parts.append(
            f'<text x="{x + 26}" y="{legend_y + 14}" font-size="12" font-family="Arial, Helvetica, sans-serif" fill="#333">{xml_escape(cell_type)}</text>'
        )

    edge_legend_y = legend_y + 34
    parts.append('<text x="40" y="130" font-size="12" font-family="Arial, Helvetica, sans-serif" fill="#444">Arrow meaning:</text>')
    parts.append('<path d="M140,126 C185,126 185,126 230,126" fill="none" stroke="#1f77b4" stroke-width="3.0" marker-end="url(#arrow)"/>')
    parts.append('<text x="240" y="130" font-size="12" font-family="Arial, Helvetica, sans-serif" fill="#333">Blue = positive coefficient</text>')
    parts.append('<path d="M430,126 C475,126 475,126 520,126" fill="none" stroke="#d62728" stroke-width="3.0" marker-end="url(#arrow)"/>')
    parts.append('<text x="530" y="130" font-size="12" font-family="Arial, Helvetica, sans-serif" fill="#333">Red = negative coefficient</text>')
    parts.append('<path d="M140,150 C185,150 185,150 230,150" fill="none" stroke="#666" stroke-width="1.8" marker-end="url(#arrow)"/>')
    parts.append('<path d="M430,150 C475,150 475,150 520,150" fill="none" stroke="#666" stroke-width="4.0" marker-end="url(#arrow)"/>')
    parts.append('<text x="240" y="154" font-size="12" font-family="Arial, Helvetica, sans-serif" fill="#333">Thin = weaker held-out gain</text>')
    parts.append('<text x="530" y="154" font-size="12" font-family="Arial, Helvetica, sans-serif" fill="#333">Thick = stronger held-out gain</text>')
    parts.append('<line x1="820" y1="126" x2="900" y2="126" stroke="#9aa0a6" stroke-width="2" stroke-dasharray="6 4"/>')
    parts.append('<text x="910" y="130" font-size="12" font-family="Arial, Helvetica, sans-serif" fill="#333">Gray dashed = same node across stages</text>')
    parts.append('<text x="40" y="170" font-size="12" font-family="Arial, Helvetica, sans-serif" fill="#666">Colored arrows are within-stage directional edges, not direct cross-stage transitions.</text>')

    for stage_idx, stage in enumerate(stage_order):
        x0 = margin_x + stage_idx * column_width
        y0 = margin_y + legend_height
        panel_height = height - y0 - margin_y
        parts.append(f'<rect x="{x0}" y="{y0}" width="{column_width - stage_gap}" height="{panel_height}" rx="10" fill="#f1f3f4" stroke="#d9dde1" stroke-width="1"/>')
        parts.append(
            f'<text x="{x0 + 16}" y="{y0 + 28}" font-size="20" font-family="Arial, Helvetica, sans-serif" fill="#222">{xml_escape(stage)}</text>'
        )
        for node_idx, node_id in enumerate(stage_node_ids[stage]):
            node = nodes_by_id[node_id]
            box_x = x0 + 16
            box_y = y0 + 48 + node_idx * (node_height + node_gap)
            node_positions[(stage, node_id)] = (box_x, box_y)
            color = CELL_TYPE_COLORS.get(node["cell_type"], "#999999")
            parts.append(f'<rect x="{box_x}" y="{box_y}" width="{node_width}" height="{node_height}" rx="6" fill="{color}" fill-opacity="0.92" stroke="#333" stroke-width="1"/>')
            parts.append(
                f'<text x="{box_x + 10}" y="{box_y + 17}" font-size="12" font-family="Arial, Helvetica, sans-serif" font-weight="bold" fill="#111">{xml_escape(node["cell_type"])}</text>'
            )
            parts.append(
                f'<text x="{box_x + 10}" y="{box_y + 32}" font-size="12" font-family="Arial, Helvetica, sans-serif" fill="#111">{xml_escape(node["signature"])}</text>'
            )
            parts.append(
                f'<text x="{box_x + node_width - 12}" y="{box_y + 25}" text-anchor="end" font-size="11" font-family="Arial, Helvetica, sans-serif" fill="#111">Δ={node["delta"]:+.2f}</text>'
            )

    for stage in stage_order:
        for edge in stage_edges.get(stage, []):
            source_pos = node_positions.get((stage, edge["source"]))
            target_pos = node_positions.get((stage, edge["target"]))
            if source_pos is None or target_pos is None:
                continue
            source_x, source_y = source_pos
            target_x, target_y = target_pos
            x1 = source_x + node_width
            y1 = source_y + node_height / 2
            x2 = target_x
            y2 = target_y + node_height / 2
            mid_x = (x1 + x2) / 2
            coeff_color = "#1f77b4" if edge["selected_coeff"] >= 0 else "#d62728"
            stroke_width = 1.5 + min(edge["selected_delta_r2"], 1.5)
            parts.append(
                f'<path d="M{x1},{y1} C{mid_x},{y1} {mid_x},{y2} {x2},{y2}" fill="none" stroke="{coeff_color}" stroke-width="{stroke_width:.2f}" marker-end="url(#arrow)" opacity="0.9"/>'
            )
            label_x = mid_x
            label_y = min(y1, y2) - 6 if abs(y2 - y1) > 8 else y1 - 8
            label = f'{edge["selected_delta_r2"]:.2f} | p={edge["selected_empirical_p"]:.2f}'
            if edge["stage_consistency"] == "recurrent":
                label += " | recurrent"
            parts.append(f'<text x="{label_x}" y="{label_y}" text-anchor="middle" font-size="11" font-family="Arial, Helvetica, sans-serif" fill="#333">{xml_escape(label)}</text>')

    repeated_nodes = defaultdict(list)
    for stage, node_ids in stage_node_ids.items():
        for node_id in node_ids:
            repeated_nodes[node_id].append(stage)
    stage_rank = {stage: idx for idx, stage in enumerate(stage_order)}
    for node_id, stages in repeated_nodes.items():
        ordered_stages = sorted(stages, key=lambda stage: stage_rank[stage])
        if len(ordered_stages) < 2:
            continue
        for left_stage, right_stage in zip(ordered_stages, ordered_stages[1:]):
            left = node_positions.get((left_stage, node_id))
            right = node_positions.get((right_stage, node_id))
            if left is None or right is None:
                continue
            x1 = left[0] + node_width + 8
            y1 = left[1] + node_height / 2
            x2 = right[0] - 8
            y2 = right[1] + node_height / 2
            parts.append(
                f'<path d="M{x1},{y1} C{(x1+x2)/2},{y1} {(x1+x2)/2},{y2} {x2},{y2}" fill="none" stroke="#9aa0a6" stroke-width="2" stroke-dasharray="6 4" opacity="0.8"/>'
            )

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render stage-wise SCP2154 network as a compact graph view")
    parser.add_argument("--nodes", type=Path, default=Path("results/scp2154_stagewise_network/nodes.tsv"))
    parser.add_argument("--edges", type=Path, default=Path("results/scp2154_stagewise_network/stage_chain.tsv"))
    parser.add_argument("--stages", default="low_steatosis,cirrhotic,Tumor")
    parser.add_argument("--node-abs-delta-min", type=float, default=0.2)
    parser.add_argument("--edge-min-delta", type=float, default=0.3)
    parser.add_argument("--edge-max-p", type=float, default=0.12)
    parser.add_argument("--max-abs-delta", type=float, default=5.0)
    parser.add_argument("--top-edges-per-stage", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("results/scp2154_stagewise_network_graph"))
    args = parser.parse_args()

    stage_order = [item.strip() for item in args.stages.split(",") if item.strip()]
    node_rows = load_tsv(args.nodes)
    edge_rows = load_tsv(args.edges)
    nodes_by_id, stage_nodes = choose_nodes(node_rows, stage_order, args.node_abs_delta_min)
    stage_edges = choose_edges(
        edge_rows,
        stage_order,
        args.edge_min_delta,
        args.edge_max_p,
        args.max_abs_delta,
        args.top_edges_per_stage,
    )

    mermaid = build_mermaid(stage_order, nodes_by_id, stage_nodes, stage_edges)
    stage_summary = build_stage_summary(stage_order, stage_nodes, stage_edges)
    node_table = build_node_table(stage_order, stage_edges, nodes_by_id)
    edge_table = build_edge_table(stage_order, stage_edges)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mermaid_path = args.output_dir / "stagewise_network.mmd"
    svg_path = args.output_dir / "stagewise_network.svg"
    markdown_path = args.output_dir / "stagewise_network.md"
    stage_summary_path = args.output_dir / "stage_summary.tsv"
    node_table_path = args.output_dir / "graph_nodes.tsv"
    edge_table_path = args.output_dir / "graph_edges.tsv"

    save_text(mermaid_path, mermaid)
    save_text(svg_path, build_svg(stage_order, nodes_by_id, stage_edges))
    save_text(stage_summary_path, stage_summary)
    save_text(node_table_path, node_table)
    save_text(edge_table_path, edge_table)
    save_text(
        markdown_path,
        build_markdown("SCP2154 Stagewise Network Graph", mermaid, stage_summary_path, node_table_path, edge_table_path),
    )

    print("Rendered SCP2154 stagewise network graph", flush=True)
    print(f"  {mermaid_path}", flush=True)
    print(f"  {svg_path}", flush=True)
    print(f"  {markdown_path}", flush=True)
    print(f"  {stage_summary_path}", flush=True)
    print(f"  {node_table_path}", flush=True)
    print(f"  {edge_table_path}", flush=True)


if __name__ == "__main__":
    main()
