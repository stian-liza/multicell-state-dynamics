from __future__ import annotations

import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_liver_velocity_input_manifest import collect_rows, dataset_file_type
from extract_geo_supplementary import extract_archive


class CloudDataToolsTest(unittest.TestCase):
    def test_dataset_file_type(self) -> None:
        self.assertEqual(dataset_file_type(Path("x.h5")), "filtered_h5")
        self.assertEqual(dataset_file_type(Path("x.mtx.gz")), "matrix_mtx")
        self.assertEqual(dataset_file_type(Path("x_barcodes.tsv.gz")), "barcodes_tsv")
        self.assertEqual(dataset_file_type(Path("x_genes.tsv.gz")), "genes_tsv")

    def test_extract_archive_and_collect_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            archive_path = tmp / "GSE000001_RAW.tar"
            extract_root = tmp / "data" / "interim" / "gse000001" / "supplementary_raw"
            extract_root.parent.mkdir(parents=True, exist_ok=True)

            input_dir = tmp / "input"
            input_dir.mkdir()
            (input_dir / "sample_filtered_feature_bc_matrix.h5").write_bytes(b"h5")
            (input_dir / "sample_matrix.mtx.gz").write_bytes(b"mtx")

            with tarfile.open(archive_path, "w") as archive:
                archive.add(input_dir / "sample_filtered_feature_bc_matrix.h5", arcname="sample_filtered_feature_bc_matrix.h5")
                archive.add(input_dir / "sample_matrix.mtx.gz", arcname="sample_matrix.mtx.gz")

            summary = extract_archive(archive_path, extract_root)
            self.assertEqual(summary["h5"], 1)
            self.assertEqual(summary["mtx"], 1)

            rows = collect_rows(tmp / "data" / "interim")
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["file_type"] for row in rows}, {"filtered_h5", "matrix_mtx"})


if __name__ == "__main__":
    unittest.main()
