# SCP2154 Liver Phenotype Classification Baseline

This baseline is intentionally conservative.

It does not claim disease progression or causality. It only tests whether liver disease phenotypes in the SCP2154 atlas can be separated under a donor-held-out evaluation design.

## Design

- keep liver cells only, if a tissue/organ column is available
- choose 2-3 clear phenotype groups
- sample by phenotype and donor
- split train/validation/test by donor, not by cell
- use log-normalized HVG expression
- train a simple softmax logistic regression baseline

## Expected Input Files

Place downloaded SCP2154 files under:

```text
data/raw/scp2154/
├── metadata.tsv.gz
└── counts.tsv.gz
```

The expression matrix should be a dense gene-by-cell table:

```text
cell_1  cell_2  cell_3 ...
gene_A  0       1       0
gene_B  2       0       3
```

If SCP uses different column names, pass them explicitly:

```bash
python scripts/run_scp2154_phenotype_baseline.py \
  --metadata data/raw/scp2154/metadata.tsv.gz \
  --matrix data/raw/scp2154/counts.tsv.gz \
  --cell-col cell_id \
  --donor-col donor \
  --phenotype-col phenotype \
  --tissue-col tissue \
  --tissue-value liver
```

## Interpretation

This is a workflow sanity check:

- If donor-held-out performance is poor, the phenotype labels may not be separable from expression alone at this scale.
- If donor-held-out performance is strong, the next step is not to claim progression, but to inspect which cell types and modules drive separability.
- If cell-random performance is much higher than donor-held-out performance, the model is likely exploiting donor/study effects.
