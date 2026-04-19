# Multicell State Dynamics

An MVP research repository for coarse-grained multicellular state evolution modeling.

Chinese overview: [README.zh-CN.md](README.zh-CN.md)

This project translates a PhD-style research plan into a runnable GitHub prototype. It focuses on the smallest credible loop:

- define interpretable module variables from high-dimensional cell features
- fit local module-velocity dynamics
- add cross-population coupling terms
- rank candidate driver modules from a sparse linear dynamical system

## Why This Repo Exists

The full research plan is scientifically strong, but too broad to execute all at once. The practical strategy is:

1. Start with one disease context.
2. Focus on one major cell population and one transition of interest.
3. Build a module-level dynamical model before adding heavy evidence integration.
4. Treat literature and database reasoning as a later ranking layer, not the core model.

This repository implements that first closed loop.

## Feasibility Assessment

The plan is feasible if it is scoped as an incremental program rather than a fully parallel three-stage system.

What is realistic:

- module-level dynamics instead of gene-level mechanistic ODEs
- local velocity supervision instead of long-horizon forecasting
- sparse linear or weakly nonlinear coupling as the first model family
- one disease and one transition as the first case study

What is risky if attempted too early:

- simultaneous integration of scRNA, spatial, genetics, literature priors, structure, and drug design
- strong causal claims from trajectory reconstruction alone
- full intervention design before the dynamical model is stable

## Repository Layout

```text
multicell-state-dynamics/
├── docs/
│   └── roadmap.md
├── scripts/
│   └── run_synthetic_demo.py
├── src/
│   └── multicell_dynamics/
│       ├── __init__.py
│       ├── coupling.py
│       ├── dynamics.py
│       ├── module_learning.py
│       └── synthetic.py
├── tests/
│   └── test_synthetic_pipeline.py
├── .gitignore
└── pyproject.toml
```

## Quick Start

```bash
cd "/Users/einstian/Documents/New project/multicell-state-dynamics"
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/run_synthetic_demo.py
python -m unittest discover -s tests
```

If your shell is currently inside another Python environment such as Conda `base`, the most reliable option is:

```bash
cd "/Users/einstian/Documents/New project/multicell-state-dynamics"
python3 -m venv --system-site-packages .venv
./.venv/bin/python -m pip install -e . --no-build-isolation
./.venv/bin/python scripts/run_synthetic_demo.py
./.venv/bin/python -m unittest discover -s tests
```

The demo generates synthetic cell states, learns module variables with NMF, fits sparse dynamics, and prints the highest-magnitude inferred edges.

## Real Data Prototype: GSE185477

This repository now includes a first-pass real-data prototype built around `GSE185477`, using:

- one sample: `C41`
- one cell group: `Macrophage`
- one direction proxy: `pseudotime` plus local neighborhood differences

This is intentionally a `velocity-free` first pass. It does not claim RNA velocity. Instead, it estimates a local direction field from the state embedding and uses that as a weak supervision target for coarse-grained dynamics.

### Downloaded files expected locally

```text
data/raw/gse185477/
├── GSE185477_Final_Metadata.txt.gz
└── GSE185477_GSM3178784_C41_SC_raw_counts.zip
```

### Inspect metadata

```bash
./.venv/bin/python scripts/inspect_gse185477.py
```

### Run the real-data demo

```bash
./.venv/bin/python scripts/run_gse185477_demo.py
```

The real-data demo:

- selects `C41` macrophage cells from metadata
- separates `InfMac` and `NonInfMac`
- normalizes counts and keeps highly variable genes
- learns module variables
- builds a PCA state embedding
- orients pseudotime so `NonInfMac -> InfMac`
- estimates local module-direction targets from neighborhood differences
- fits a sparse module-level dynamics model

Expected behavior on real data:

- metrics will be much lower than the synthetic demo
- this is normal and desirable
- the goal is to establish a credible real-data first pass, not an artificially perfect fit

## SCP2154 Liver Phenotype Baseline

The SCP2154 atlas can be used as a conservative phenotype classification baseline. The first pass should not claim disease progression or causality. It should only test whether liver phenotypes are separable under donor-held-out evaluation.

See [docs/scp2154_phenotype_baseline.md](docs/scp2154_phenotype_baseline.md).

```bash
python scripts/run_scp2154_phenotype_baseline.py \
  --metadata data/raw/scp2154/metadata.tsv.gz \
  --matrix data/raw/scp2154/counts.tsv.gz
```

## Cloud-Scale Next Step

The next planned expansion is a cloud-scale liver disease progression program with:

- raw-read or raw spliced/unspliced capable datasets
- velocity-aware preprocessing
- donor-aware multi-dataset integration
- shared module/state variables across liver disease stages
- hepatocyte-centered strict driver-chain analysis

Design notes:

- [docs/liver_velocity_cloud_plan.md](docs/liver_velocity_cloud_plan.md)
- [docs/liver_velocity_dataset_manifest.tsv](docs/liver_velocity_dataset_manifest.tsv)

## Suggested Next Steps

- replace the synthetic generator with a small real benchmark dataset
- add RNA velocity or pseudotime-derived local direction supervision
- split by cell population and estimate one model per population
- add spatial neighbor weights for cross-cell coupling
- build a separate evidence-ranking layer for literature and genetics

## Good First Real Dataset

For a first real case, use one disease dataset with:

- a clear malignant-to-resistant or inflamed-to-exhausted transition
- enough cells per major population
- optional spatial coordinates, but not required for the first pass

## License

MIT
