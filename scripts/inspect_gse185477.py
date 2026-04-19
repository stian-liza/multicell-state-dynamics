from __future__ import annotations

from pathlib import Path

from multicell_dynamics.real_data import metadata_columns, read_metadata_table


def main() -> None:
    metadata_path = Path("data/raw/gse185477/GSE185477_Final_Metadata.txt.gz")
    if not metadata_path.exists():
        raise SystemExit(
            "Missing metadata file. Download GSE185477 metadata to "
            "data/raw/gse185477/GSE185477_Final_Metadata.txt.gz first."
        )
    columns = metadata_columns(metadata_path)
    print("Metadata columns:")
    for name in columns:
        print(" ", name)
    print("\nFirst rows:")
    for row in read_metadata_table(metadata_path, max_rows=3):
        print(row)


if __name__ == "__main__":
    main()
