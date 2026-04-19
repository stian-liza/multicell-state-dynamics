from __future__ import annotations

import argparse
import csv
import shutil
import ssl
import sys
from pathlib import Path
from urllib.request import urlopen


DEFAULT_MANIFEST = Path("docs/liver_velocity_dataset_manifest.tsv")


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return {row["accession"]: row for row in reader}


def geo_download_url(accession: str) -> str:
    return f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={accession}&format=file"


def download_file(url: str, output_path: Path, insecure: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ssl_context = None
    if insecure:
        ssl_context = ssl._create_unverified_context()
    with urlopen(url, context=ssl_context) as response, output_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def expected_filename(accession: str) -> str:
    return f"{accession}_RAW.tar"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GEO supplementary raw archive for a listed accession")
    parser.add_argument("accession", help="GEO accession, for example GSE186343")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--insecure", action="store_true", help="Disable SSL verification for environments with local TLS interception")
    args = parser.parse_args()

    manifest = read_manifest(args.manifest)
    if args.accession not in manifest:
        raise SystemExit(f"{args.accession} not found in manifest {args.manifest}")

    row = manifest[args.accession]
    if row["raw_support"] == "processed_only":
        raise SystemExit(f"{args.accession} is marked processed_only in the manifest and should not be used for raw download.")

    output_dir = args.output_dir / args.accession.lower()
    output_path = output_dir / expected_filename(args.accession)
    url = geo_download_url(args.accession)
    print(f"Downloading {args.accession} supplementary archive from {url}", flush=True)
    download_file(url, output_path, insecure=args.insecure)
    print(f"Saved to {output_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
