# Cloud Velocity Runbook

## Purpose

This runbook turns the repository into a reproducible cloud workflow for liver disease progression modeling with velocity-aware preprocessing and donor-level integration.

## 1. Provision a cloud machine

Recommended starting point:

- `16+ vCPU`
- `64+ GB RAM`
- `500+ GB` fast disk

If FASTQ alignment will be run on the same node, prefer:

- `32 vCPU`
- `128 GB RAM`
- `1+ TB` fast disk

## 2. Clone the repository

```bash
git clone https://github.com/stian-liza/multicell-state-dynamics.git
cd multicell-state-dynamics
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 3. Download public supplementary raw packages

```bash
python scripts/download_geo_supplementary.py GSE136103 --output-dir data/raw
python scripts/download_geo_supplementary.py GSE186343 --output-dir data/raw
```

If the cloud environment uses TLS interception, add `--insecure`.

## 4. Extract archives into a stable workspace

```bash
python scripts/extract_geo_supplementary.py GSE136103
python scripts/extract_geo_supplementary.py GSE186343
python scripts/build_liver_velocity_input_manifest.py
```

This produces:

- `data/interim/gse136103/supplementary_raw/...`
- `data/interim/gse186343/supplementary_raw/...`
- `data/interim/liver_velocity_input_manifest.tsv`

## 5. Add true velocity-grade raw reads

Supplementary H5/MTX inputs are good enough for:

- cohort organization
- metadata harmonization
- donor-aware integration prototypes

They are not sufficient for full spliced/unspliced velocity.

For velocity, fetch SRA/FASTQ for:

- `GSE136103` -> `SRP218975`
- `GSE186343` -> `PRJNA773427`
- `GSE156625` -> `SRP278381`

Preferred preprocessing targets:

- `STARsolo --soloFeatures Gene Velocyto`
- or `kb-python` / `alevin-fry` if chemistry handling is cleaner

Expected outputs per dataset:

- spliced count matrix
- unspliced count matrix
- filtered donor/sample metadata
- per-sample QC summary

## 6. Modeling roadmap

### Phase A: supplementary-input integration

Goal:

- harmonize donors
- harmonize cell-type labels
- define healthy-reference initial quantities

### Phase B: velocity-aware per-dataset preprocessing

Goal:

- compute cell-level spliced/unspliced dynamics
- aggregate gene-level velocity to module-level velocity

### Phase C: shared multicellular coupling model

Goal:

- integrate donors across liver disease stages
- fit hepatocyte-centered strict networks
- test recurrence across datasets

## 7. Current first cloud milestone

The first milestone is:

- `GSE136103` extracted
- `GSE186343` extracted
- input manifest built
- one hepatocyte-centered integrated cohort definition created

Only after that should full FASTQ-level velocity preprocessing begin.
