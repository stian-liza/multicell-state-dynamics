# Liver Velocity Cloud Plan

## Goal

Turn the current SCP2154/GSE prototypes into a cloud-scale, donor-aware liver disease progression program with:

- raw-read or raw-spliced/unspliced capable datasets
- per-dataset velocity preprocessing
- cross-dataset donor integration
- cell-type-specific initial state definitions
- module-level multicellular coupling dynamics

## Why Redesign

The current repository already proves a small closed loop:

- fixed or learned cell-state/module variables can be defined
- donor-held-out directional comparisons can be run
- strict filtering can collapse weak network claims

What it does not yet provide is a strong velocity-constrained, multi-dataset, multi-donor progression model. The cloud redesign addresses that gap.

## Recommended Dataset Stack

### Tier 1: fibrosis / inflammation / premalignant liver

- `GSE136103`
  - human healthy and cirrhotic liver
  - SRA available: `SRP218975`
  - useful for healthy-to-fibrotic niche remodeling

- `GSE186343`
  - chronic HBV liver inflammation progression
  - SRA available: `SRP342568`
  - supplementary raw H5 archive available
  - useful for early inflammatory progression

### Tier 2: HCC / late-stage disease

- `GSE156625`
  - healthy normal liver plus HCC
  - SRA available: `SRP278381`
  - useful for tumor-stage microenvironment and hepatocyte malignant-like states

### Tier 3: processed-only validation cohorts

- `GSE149614`
  - strong HCC ecosystem atlas
  - processed matrices available, raw data not directly available in GEO
  - use as external validation, not primary velocity training set

## Cloud Pipeline

### Step 1: raw preprocessing

For each Tier 1/2 dataset:

- download SRA / FASTQ or raw H5 inputs
- align with a velocity-aware pipeline
- preferred options:
  - `STARsolo --soloFeatures Gene Velocyto`
  - or `kb-python` / `alevin-fry` if chemistry support is cleaner
- output:
  - spliced matrix
  - unspliced matrix
  - basic QC metrics
  - donor/sample metadata table

### Step 2: per-dataset cell typing

- initial cell labels from marker-guided annotation
- retain major compartments:
  - `Hepatocyte`
  - `Myeloid`
  - `Stromal`
  - `Endothelial`
  - `TNKcell`
  - `Bcell`

### Step 3: generalized initial quantity per cell type

Define a reusable "initial quantity" for each cell type as the healthy-like or least-remodeled reference projection.

Recommended operational definition:

- for each cell type, build a healthy/non-tumor donor reference centroid
- define `q0` as similarity to that centroid
- define a disease-axis score `qd` as projection onto the dominant healthy-to-disease contrast
- use:
  - `q0` as initial-state anchor
  - `qd` as coarse disease progression coordinate

This is more transferable than hard-coding a unique root cell for each dataset.

### Step 4: shared state and module variables

Use a two-layer representation:

- dataset-local velocity estimation
  - compute cell-level velocity within each dataset separately
- shared module/state space
  - map all datasets into a common donor-aware latent space
  - candidate integration tools:
    - `scVI` for expression integration
    - `scANVI` if label refinement is needed

For each cell type:

- learn consensus modules from pooled non-tumor donors first
- project each dataset into the same module basis
- compute module velocity by aggregating gene-level spliced/unspliced dynamics into module space

## Dynamical Model

For each receiver cell type:

`dm/dt = A m + B z + C e + D q0 + b`

Where:

- `m`: module activity
- `z`: shared low-dimensional state
- `e`: donor-level external inputs from other cell-type modules
- `q0`: initial-state anchor or healthy-reference quantity

## What To Optimize Dynamically

Do not optimize for one global score only. Use staged tuning:

### Representation tuning

- number of HVGs
- latent dimension
- module count per cell type
- consensus module stability across donors

### Velocity tuning

- neighbor count
- moments / smoothing parameters
- minimum donor cell count
- filtering of low-information genes and cells

### Coupling tuning

- donor-held-out `delta R^2`
- sign agreement of module velocity
- edge stability under bootstrap / permutation
- recurrence across datasets and disease stages

## Success Criteria

The cloud redesign should be considered successful only if it achieves all of the following:

- at least one hepatocyte-centered chain survives strict filtering in more than one dataset
- donor-held-out directionality remains after raising permutation counts
- the same sender/receiver relation recurs across at least two disease contexts
- velocity-based module dynamics outperform pseudotime-only local difference baselines

## Immediate Repository Needs

- keep raw data out of Git
- keep manifests, configs, and scripts in Git
- add reproducible dataset manifests
- add cloud runner entrypoints for preprocessing and training

## Practical Next Move

1. Create or connect a GitHub remote for this repository.
2. Add a dataset manifest for Tier 1/2 liver cohorts.
3. Download one Tier 1 dataset and one Tier 2 dataset on cloud storage.
4. Implement velocity-aware preprocessing on cloud.
5. Rebuild the hepatocyte-centered strict network using true module velocity.
