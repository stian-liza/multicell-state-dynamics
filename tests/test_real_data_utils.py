from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from multicell_dynamics import (
    local_direction_from_pseudotime,
    log1p_library_normalize,
    pca_embedding,
    pseudotime_from_embedding,
    read_10x_triplet_from_zip,
    select_highly_variable_genes,
)


class RealDataUtilsTest(unittest.TestCase):
    def test_parse_minimal_10x_zip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "tiny_10x.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("matrix.mtx", "%%MatrixMarket matrix coordinate integer general\n%\n3 2 4\n1 1 1\n2 1 2\n2 2 3\n3 2 4\n")
                archive.writestr("barcodes.tsv", "cellA\ncellB\n")
                archive.writestr("features.tsv", "g1\tGene1\n g2\tGene2\n g3\tGene3\n")
            parsed = read_10x_triplet_from_zip(zip_path)
            self.assertEqual(parsed.matrix.shape, (2, 3))
            self.assertEqual(parsed.cell_barcodes.tolist(), ["cellA", "cellB"])
            self.assertEqual(parsed.gene_names.tolist(), ["Gene1", "Gene2", "Gene3"])

    def test_basic_preprocessing_and_direction(self) -> None:
        x = np.array([[1.0, 0.0, 4.0], [2.0, 1.0, 8.0], [3.0, 1.0, 16.0]])
        norm = log1p_library_normalize(x)
        reduced, genes = select_highly_variable_genes(norm, np.array(["a", "b", "c"], dtype=object), top_k=2)
        embedding = pca_embedding(reduced, n_components=2)
        pseudotime = pseudotime_from_embedding(embedding)
        direction = local_direction_from_pseudotime(reduced, pseudotime, k_neighbors=1)
        self.assertEqual(reduced.shape, (3, 2))
        self.assertEqual(len(genes), 2)
        self.assertEqual(embedding.shape[0], 3)
        self.assertEqual(direction.shape, reduced.shape)


if __name__ == "__main__":
    unittest.main()
