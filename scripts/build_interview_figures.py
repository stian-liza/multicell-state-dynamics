from __future__ import annotations

import csv
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT_DIR = RESULTS / "interview_figures"


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def copy_figure(source: Path, target_name: str) -> Path:
    destination = OUTPUT_DIR / target_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination


def make_directional_chain_svg(chain_hops: list[dict[str, str]], triplets: list[dict[str, str]]) -> str:
    width = 1360
    height = 820
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FAF8F4"/>',
        '<text x="72" y="56" font-size="30" font-family="Arial, Helvetica, sans-serif" fill="#15202B">Interview Figure 4. Candidate Directional Chain</text>',
        '<text x="72" y="88" font-size="17" font-family="Arial, Helvetica, sans-serif" fill="#52606D">This figure condenses the strongest local chain we found in SCP2154: immune input into hepatocyte stress, then stress toward malignant-like state.</text>',
    ]

    node_positions = [
        ("Myeloid.c1qc_macrophage", 120, 190, 250, 84, "#CFE8FF"),
        ("Myeloid.inflammatory_monocyte", 450, 190, 300, 84, "#FFD9B8"),
        ("Hepatocyte.secretory_stress", 830, 190, 260, 84, "#F8D7DA"),
        ("Hepatocyte.malignant_like", 1130, 190, 180, 84, "#E6D3F2"),
    ]
    lookup: dict[str, tuple[int, int, int, int]] = {}
    for label, x, y, w, h, fill in node_positions:
        lookup[label] = (x, y, w, h)
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="{fill}" stroke="#2F3A46" stroke-width="1.5"/>')
        lines = label.split(".")
        parts.append(
            f'<text x="{x + w/2:.1f}" y="{y + 34}" text-anchor="middle" font-size="18" font-family="Arial, Helvetica, sans-serif" fill="#15202B">{lines[0]}</text>'
        )
        parts.append(
            f'<text x="{x + w/2:.1f}" y="{y + 58}" text-anchor="middle" font-size="15" font-family="Arial, Helvetica, sans-serif" fill="#334E68">{lines[1]}</text>'
        )

    arrow_y = 232
    for hop in chain_hops:
        source = hop["source"]
        target = hop["target"]
        if source not in lookup or target not in lookup:
            continue
        sx, sy, sw, sh = lookup[source]
        tx, ty, tw, th = lookup[target]
        x1 = sx + sw
        x2 = tx
        forward = float(hop["forward_delta_r2"])
        reverse = float(hop["reverse_delta_r2"])
        winner_p = float(hop["winner_p"])
        stage = hop["stage"]
        stroke_w = 4.5 + max(0.0, forward) * 7.0
        parts.append(
            f'<line x1="{x1}" y1="{arrow_y}" x2="{x2}" y2="{arrow_y}" stroke="#245BDB" stroke-width="{stroke_w:.1f}" marker-end="url(#arrowBlue)" stroke-linecap="round"/>'
        )
        parts.append(
            f'<text x="{(x1 + x2)/2:.1f}" y="{arrow_y - 20}" text-anchor="middle" font-size="15" font-family="Arial, Helvetica, sans-serif" fill="#1F2933">{stage}</text>'
        )
        parts.append(
            f'<text x="{(x1 + x2)/2:.1f}" y="{arrow_y + 30}" text-anchor="middle" font-size="15" font-family="Arial, Helvetica, sans-serif" fill="#1F2933">forward ΔR²={forward:.3f} | reverse ΔR²={reverse:.3f}</text>'
        )
        parts.append(
            f'<text x="{(x1 + x2)/2:.1f}" y="{arrow_y + 53}" text-anchor="middle" font-size="14" font-family="Arial, Helvetica, sans-serif" fill="#52606D">winner p={winner_p:.4f}</text>'
        )

    parts.insert(
        1,
        '<defs><marker id="arrowBlue" markerWidth="12" markerHeight="12" refX="9" refY="6" orient="auto"><path d="M0,0 L12,6 L0,12 z" fill="#245BDB"/></marker></defs>',
    )

    parts.append('<text x="72" y="348" font-size="24" font-family="Arial, Helvetica, sans-serif" fill="#15202B">What This Means</text>')
    bullets = [
        "Hop 1 is a weaker cirrhotic-stage lead-in and should be presented as exploratory.",
        "Hop 2 and Hop 3 are both tumor-stage directional wins with held-out support.",
        "The strongest local story is inflammatory myeloid input feeding hepatocyte stress, then stress moving toward malignant-like state.",
    ]
    bullet_y = 386
    for bullet in bullets:
        parts.append(f'<circle cx="88" cy="{bullet_y - 6}" r="4" fill="#245BDB"/>')
        parts.append(
            f'<text x="104" y="{bullet_y}" font-size="17" font-family="Arial, Helvetica, sans-serif" fill="#243B53">{bullet}</text>'
        )
        bullet_y += 34

    if triplets:
        triplet = triplets[0]
        x0 = 72
        y0 = 520
        w = 1210
        h = 216
        parts.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" rx="22" fill="#FFFFFF" stroke="#D9E2EC" stroke-width="1.5"/>')
        parts.append(
            f'<text x="{x0 + 24}" y="{y0 + 38}" font-size="24" font-family="Arial, Helvetica, sans-serif" fill="#15202B">Triplet Mediation Check: {triplet["source"]} -&gt; {triplet["mediator"]} -&gt; {triplet["target"]}</text>'
        )
        parts.append(
            f'<text x="{x0 + 24}" y="{y0 + 68}" font-size="16" font-family="Arial, Helvetica, sans-serif" fill="#52606D">Status = {triplet["mediation_call"]}; donor-level joint model gain tests whether the middle hepatocyte stress state adds explanatory power.</text>'
        )

        metrics = [
            ("Source-only ΔR²", float(triplet["source_only_delta_r2"])),
            ("Mediator-only ΔR²", float(triplet["mediator_only_delta_r2"])),
            ("Joint ΔR²", float(triplet["joint_delta_r2"])),
            ("Joint gain", float(triplet["joint_gain_over_source"])),
            ("Joint gain p", float(triplet["joint_gain_p"])),
        ]
        card_x = x0 + 24
        card_y = y0 + 96
        for label, value in metrics:
            card_w = 214
            parts.append(f'<rect x="{card_x}" y="{card_y}" width="{card_w}" height="88" rx="14" fill="#F7FAFC" stroke="#E4E7EB" stroke-width="1"/>')
            parts.append(
                f'<text x="{card_x + 14}" y="{card_y + 30}" font-size="15" font-family="Arial, Helvetica, sans-serif" fill="#52606D">{label}</text>'
            )
            fmt = f"{value:.4f}"
            parts.append(
                f'<text x="{card_x + 14}" y="{card_y + 60}" font-size="28" font-family="Arial, Helvetica, sans-serif" fill="#102A43">{fmt}</text>'
            )
            card_x += 232

    parts.append("</svg>")
    return "".join(parts)


def make_loose_vs_strict_svg(loose_summary: dict[str, str], strict_summary: dict[str, str]) -> str:
    width = 1180
    height = 760
    margin = 92
    chart_bottom = 590
    chart_height = 330

    loose_bidirectional = int(loose_summary["bidirectional_edges"])
    loose_chain = int(loose_summary["chain_edges"])
    strict_bidirectional = int(strict_summary["bidirectional_edges"])
    strict_chain = int(strict_summary["chain_edges"])
    vmax = max(loose_bidirectional, strict_bidirectional, 1)

    def bar_height(value: int) -> float:
        return chart_height * value / vmax

    bars = [
        ("Loose\nnetwork", "#356AE6", loose_bidirectional, loose_chain),
        ("Strict\nnetwork", "#0F7B6C", strict_bidirectional, strict_chain),
    ]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FAF8F4"/>',
        '<text x="72" y="56" font-size="30" font-family="Arial, Helvetica, sans-serif" fill="#15202B">Interview Figure 5. Why The Model Is Not Just Drawing Arrows</text>',
        '<text x="72" y="88" font-size="17" font-family="Arial, Helvetica, sans-serif" fill="#52606D">A good interview point: the exploratory network finds candidates, but stricter donor and recurrence filters remove unstable stories.</text>',
        f'<line x1="{margin}" y1="{chart_bottom}" x2="{width - margin}" y2="{chart_bottom}" stroke="#7B8794" stroke-width="1.5"/>',
        f'<line x1="{margin}" y1="{chart_bottom - chart_height}" x2="{margin}" y2="{chart_bottom}" stroke="#7B8794" stroke-width="1.5"/>',
    ]

    x_positions = [260, 650]
    for idx, (label, fill, bidirectional, chain_edges) in enumerate(bars):
        x = x_positions[idx]
        bidirectional_h = bar_height(bidirectional)
        chain_h = bar_height(chain_edges)
        parts.append(
            f'<rect x="{x}" y="{chart_bottom - bidirectional_h:.1f}" width="126" height="{bidirectional_h:.1f}" rx="14" fill="{fill}" fill-opacity="0.88"/>'
        )
        parts.append(
            f'<rect x="{x + 160}" y="{chart_bottom - chain_h:.1f}" width="126" height="{chain_h:.1f}" rx="14" fill="{fill}" fill-opacity="0.40" stroke="{fill}" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{x + 63}" y="{chart_bottom - bidirectional_h - 14:.1f}" text-anchor="middle" font-size="24" font-family="Arial, Helvetica, sans-serif" fill="#102A43">{bidirectional}</text>'
        )
        chain_label_y = chart_bottom - chain_h - 14 if chain_h > 1 else chart_bottom - 14
        parts.append(
            f'<text x="{x + 223}" y="{chain_label_y:.1f}" text-anchor="middle" font-size="24" font-family="Arial, Helvetica, sans-serif" fill="#102A43">{chain_edges}</text>'
        )
        for line_idx, line in enumerate(label.split("\n")):
            parts.append(
                f'<text x="{x + 142}" y="{chart_bottom + 42 + line_idx * 22}" text-anchor="middle" font-size="18" font-family="Arial, Helvetica, sans-serif" fill="#243B53">{line}</text>'
            )

    parts.extend(
        [
            '<rect x="820" y="190" width="18" height="18" rx="4" fill="#356AE6" fill-opacity="0.88"/>',
            '<text x="846" y="204" font-size="15" font-family="Arial, Helvetica, sans-serif" fill="#243B53">All bidirectional pairs tested</text>',
            '<rect x="820" y="220" width="18" height="18" rx="4" fill="#356AE6" fill-opacity="0.40" stroke="#356AE6" stroke-width="2"/>',
            '<text x="846" y="234" font-size="15" font-family="Arial, Helvetica, sans-serif" fill="#243B53">Retained chain edges after temporal/directional screening</text>',
        ]
    )

    narrative_y = 660
    parts.append('<text x="72" y="646" font-size="22" font-family="Arial, Helvetica, sans-serif" fill="#15202B">Take-home message</text>')
    notes = [
        f"Exploratory mode keeps {loose_chain} chain edges across {loose_summary['chain_edges_by_stage']}.",
        "Strict mode raises donor, recurrence, and p-value requirements and collapses the retained chain set to zero.",
        "That shrinkage is good interview evidence that the framework can propose hypotheses without overclaiming.",
    ]
    for note in notes:
        parts.append(f'<circle cx="88" cy="{narrative_y - 6}" r="4" fill="#0F7B6C"/>')
        parts.append(
            f'<text x="104" y="{narrative_y}" font-size="17" font-family="Arial, Helvetica, sans-serif" fill="#243B53">{note}</text>'
        )
        narrative_y += 32

    parts.append("</svg>")
    return "".join(parts)


def build_notes(figures: dict[str, Path]) -> str:
    lines = [
        "# Interview Figure Set",
        "",
        "These are the four figures I would actually use in a PhD interview to support the first validation loop of the research plan.",
        "",
        "## Figure 1. Real-data prototype runs end-to-end",
        f"- File: `{figures['fig1']}`",
        "- Use this to say: the state/module abstraction is not only conceptual; it already runs on a real macrophage dataset and separates inflammatory versus non-inflammatory states.",
        "",
        "## Figure 2. Module-level state shifts are biologically interpretable",
        f"- File: `{figures['fig2']}`",
        "- Use this to say: after module compression, the inferred programs still retain recognizable biology, so the coarse-graining is meaningful rather than arbitrary.",
        "",
        "## Figure 3. Stage-wise multicellular directional network",
        f"- File: `{figures['fig3']}`",
        "- Use this to say: on the SCP2154 atlas, the framework can already organize disease-stage-specific sender/receiver structure instead of stopping at DEG or cell proportion changes.",
        "",
        "## Figure 4. Candidate local chain and mediation support",
        f"- File: `{figures['fig4']}`",
        "- Use this to say: the most interesting local chain is inflammatory myeloid input into hepatocyte secretory stress, then toward malignant-like hepatocyte state.",
        "",
        "## Figure 5. Strict filtering shrinks the network",
        f"- File: `{figures['fig5']}`",
        "- Use this to say: the model does not force every correlation into a mechanistic story; when the criteria become strict, the chain count collapses, which is exactly the behavior you want from an honest exploratory framework.",
        "",
        "## Recommended order in the interview",
        "1. Figure 1: show the pipeline is executable on real data.",
        "2. Figure 2: show the modules are interpretable.",
        "3. Figure 3: show the cross-cell, stage-wise network idea.",
        "4. Figure 4: zoom into one concrete candidate chain.",
        "5. Figure 5: explain why the framework is disciplined rather than over-claiming.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    source_figures = {
        "fig1": RESULTS / "gse185477_demo" / "c41_macrophage_pca.svg",
        "fig2": RESULTS / "gse185477_demo" / "module_activity_shifts.svg",
        "fig3": RESULTS / "scp2154_stagewise_network_graph" / "stagewise_network.svg",
    }
    for path in source_figures.values():
        if not path.exists():
            raise SystemExit(f"Missing source figure: {path}")

    chain_hops = read_tsv(RESULTS / "scp2154_chain_directionality" / "chain_hops.tsv")
    triplets = read_tsv(RESULTS / "scp2154_chain_directionality" / "chain_triplets.tsv")
    loose_summary_rows = read_tsv(RESULTS / "scp2154_stagewise_network" / "network_summary.tsv")
    strict_summary_rows = read_tsv(RESULTS / "scp2154_stagewise_network_strict" / "network_summary.tsv")
    loose_summary = {row["metric"]: row["value"] for row in loose_summary_rows}
    strict_summary = {row["metric"]: row["value"] for row in strict_summary_rows}

    figures = {
        "fig1": copy_figure(source_figures["fig1"], "figure1_real_data_pca.svg"),
        "fig2": copy_figure(source_figures["fig2"], "figure2_module_activity_shifts.svg"),
        "fig3": copy_figure(source_figures["fig3"], "figure3_stagewise_network.svg"),
        "fig4": OUTPUT_DIR / "figure4_candidate_directional_chain.svg",
        "fig5": OUTPUT_DIR / "figure5_loose_vs_strict_network.svg",
    }

    save_text(figures["fig4"], make_directional_chain_svg(chain_hops, triplets))
    save_text(figures["fig5"], make_loose_vs_strict_svg(loose_summary, strict_summary))
    save_text(OUTPUT_DIR / "interview_figure_notes.md", build_notes(figures))

    print("Interview figures written to:")
    for key in ["fig1", "fig2", "fig3", "fig4", "fig5"]:
        print(f"  {figures[key]}")
    print(f"  {OUTPUT_DIR / 'interview_figure_notes.md'}")


if __name__ == "__main__":
    main()
