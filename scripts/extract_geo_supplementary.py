from __future__ import annotations

import argparse
import inspect
import tarfile
from pathlib import Path


def archive_path_for_accession(raw_root: Path, accession: str) -> Path:
    return raw_root / accession.lower() / f"{accession}_RAW.tar"


def safe_members(archive: tarfile.TarFile, destination: Path) -> list[tarfile.TarInfo]:
    allowed = []
    dest_root = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if not str(target).startswith(str(dest_root)):
            raise ValueError(f"Unsafe archive member path: {member.name}")
        allowed.append(member)
    return allowed


def extract_archive(archive_path: Path, output_dir: Path) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r") as archive:
        members = safe_members(archive, output_dir)
        extract_kwargs = {}
        if "filter" in inspect.signature(archive.extractall).parameters:
            extract_kwargs["filter"] = "data"
        archive.extractall(output_dir, members=members, **extract_kwargs)

    summary = {
        "h5": len(list(output_dir.rglob("*.h5"))),
        "mtx": len(list(output_dir.rglob("*.mtx.gz"))),
        "barcodes": len(list(output_dir.rglob("*barcodes.tsv.gz"))),
        "genes": len(list(output_dir.rglob("*genes.tsv.gz"))),
        "features": len(list(output_dir.rglob("*features.tsv.gz"))),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a GEO supplementary raw archive into a reusable workspace")
    parser.add_argument("accession", help="GEO accession, for example GSE136103")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-root", type=Path, default=Path("data/interim"))
    args = parser.parse_args()

    archive_path = archive_path_for_accession(args.raw_root, args.accession)
    if not archive_path.exists():
        raise SystemExit(f"Archive not found: {archive_path}")

    output_dir = args.output_root / args.accession.lower() / "supplementary_raw"
    summary = extract_archive(archive_path, output_dir)
    print(f"Extracted {args.accession} to {output_dir}", flush=True)
    for key, value in summary.items():
        print(f"  {key}: {value}", flush=True)


if __name__ == "__main__":
    main()
