# Multicell State Dynamics

A research prototype for coarse-grained modeling of multicellular state transitions from single-cell data.

Chinese version: [README.zh-CN.md](README.zh-CN.md)

## Overview

This repository explores a simple but ambitious question:

> Can we move from static single-cell association analysis toward a directional, module-level view of multicellular state change?

The current codebase focuses on three increasingly realistic steps:

1. **Synthetic dynamics demo**  
   Verify that module learning and sparse dynamics fitting work end to end.
2. **Small real-data prototype (`GSE185477`)**  
   Test whether pseudotime-derived local direction signals can support module-level dynamics on real single-cell data.
3. **Stage-resolved liver disease prototype (`SCP2154`)**  
   Use disease-stage-stratified single-cell data to compare `module state -> module velocity` in a donor-held-out, bidirectional framework.

The goal is not to claim causality. The goal is to build a credible modeling loop that can generate interpretable, testable hypotheses about state transitions and cross-cell coupling.

## Current Main Prototype

The current main line of this repository is the **SCP2154 stagewise velocity-coupling prototype**.

It uses liver single-cell data stratified by disease stage and asks whether:

- one cell population's module state can predict another population's module velocity
- that directional relationship is stronger than the reverse direction
- the signal survives donor-held-out evaluation and permutation-based filtering

The current implementation includes:

- fixed, interpretable cell-type-specific module signatures
- pseudotime-neighborhood differences as a local velocity proxy
- donor-aware pairing within each disease stage
- bidirectional testing:
  - `A score -> dB/dt`
  - `B score -> dA/dt`
- candidate stage-linked directional chains

Project summary:

- [docs/project2_velocity_coupling_summary.zh-CN.md](docs/project2_velocity_coupling_summary.zh-CN.md)

Interview-style explanation:

- [docs/scp2154_velocity_coupling_interview_summary.zh-CN.md](docs/scp2154_velocity_coupling_interview_summary.zh-CN.md)

Tracked figure assets:

- workflow: [docs/assets/project2/workflow_figure.svg](docs/assets/project2/workflow_figure.svg)
- conclusion chain: [docs/assets/project2/conclusion_chain_figure.svg](docs/assets/project2/conclusion_chain_figure.svg)

## Representative Result

Under a stricter single-direction criterion, the current prototype retains a candidate chain:

`Endothelial.inflammatory_endothelial -> Myeloid.interferon_myeloid -> Stromal.inflammatory_caf -> Hepatocyte.malignant_like`

This should be interpreted as a **candidate directional structure**, not a causal proof.

In plain terms, the current model suggests that terminal hepatocyte malignant-like acceleration may be linked to earlier microenvironment remodeling rather than appearing in isolation.

## Repository Structure

```text
multicell-state-dynamics/
├── docs/                         notes, summaries, and tracked figure assets
├── scripts/                      runnable prototypes and analysis scripts
├── src/multicell_dynamics/       core modeling utilities
├── tests/                        unit tests
├── README.md
├── README.zh-CN.md
├── pyproject.toml
└── LICENSE
```

## Key Scripts

### 1. Synthetic demo

```bash
python scripts/run_synthetic_demo.py
```

### 2. Real-data macrophage prototype (`GSE185477`)

```bash
python scripts/run_gse185477_demo.py
```

### 3. SCP2154 phenotype baseline

```bash
python scripts/run_scp2154_phenotype_baseline.py \
  --metadata data/raw/scp2154/metadata.tsv.gz \
  --matrix data/raw/scp2154/counts.tsv.gz
```

### 4. SCP2154 stagewise velocity coupling

```bash
python scripts/run_scp2154_stagewise_velocity_coupling.py
```

## Quick Start

```bash
cd "/Users/einstian/Documents/New project/multicell-state-dynamics"
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m unittest discover -s tests
```

If you are already inside another Python environment and want a more robust local setup:

```bash
cd "/Users/einstian/Documents/New project/multicell-state-dynamics"
python3 -m venv --system-site-packages .venv
./.venv/bin/python -m pip install -e . --no-build-isolation
./.venv/bin/python -m unittest discover -s tests
```

## Notes on Interpretation

This repository is deliberately conservative about claims.

- The velocity signal used here is **not RNA velocity**.
- The stagewise chains are **not longitudinal proof** across the same donors.
- The retained edges are best understood as **candidate directional hypotheses**.

That boundary is part of the design philosophy of the repo.

## Related Notes

- phenotype baseline: [docs/scp2154_phenotype_baseline.md](docs/scp2154_phenotype_baseline.md)
- process/code guide: [docs/prototype_process_code_guide.zh-CN.md](docs/prototype_process_code_guide.zh-CN.md)
- risk table: [docs/pre_experiment_assessment_and_risk_table.zh-CN.md](docs/pre_experiment_assessment_and_risk_table.zh-CN.md)

## License

MIT
