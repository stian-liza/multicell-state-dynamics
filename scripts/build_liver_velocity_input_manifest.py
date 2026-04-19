from __future__ import annotations

import argparse
import csv
from pathlib import Path


def dataset_file_type(path: Path) -> str:
    name = path.name
    if name.endswith(".h5"):
        return "filtered_h5"
    if name.endswith(".mtx.gz"):
        return "matrix_mtx"
    if name.endswith("barcodes.tsv.gz"):
        return "barcodes_tsv"
    if name.endswith("genes.tsv.gz"):
        return "genes_tsv"
    if name.endswith("features.tsv.gz"):
        return "features_tsv"
    return "other"


def collect_rows(interim_root: Path) -> list[dict[str, str]]:
    rows = []
    for dataset_dir in sorted(interim_root.glob("gse*/supplementary_raw")):
        accession = dataset_dir.parent.name.upper()
        for path in sorted(dataset_dir.rglob("*")):
            if not path.is_file():
                continue
            rows.append(
                {
                    "accession": accession,
                    "relative_path": str(path.relative_to(interim_root)),
                    "file_type": dataset_file_type(path),
                    "size_bytes": str(path.stat().st_size),
                }
            )
    return rows


def write_rows(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["accession", "relative_path", "file_type", "size_bytes"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a manifest of extracted liver velocity input files")
    parser.add_argument("--interim-root", type=Path, default=Path("data/interim"))
    parser.add_argument("--output", type=Path, default=Path("data/interim/liver_velocity_input_manifest.tsv"))
    args = parser.parse_args()

    rows = collect_rows(args.interim_root)
    write_rows(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
