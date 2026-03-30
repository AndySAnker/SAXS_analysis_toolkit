# Configuration files

YAML files under `configs/` in the repository root drive scripts and pipelines. Paths below are relative to that `configs/` directory.

Paths in YAML (data files, model paths) are typically **relative to the repository root** (`SAXS_analysis.ROOT_DIR`), as resolved by `load_config` and the scripts.

## Simulated data (HDF5)

Training configs usually point at HDF5 files produced by `simulate_sas.py`. Common datasets:

| Dataset name | Role |
|----------------|------|
| `SAXS_dataset` | Intensity rows (quotient-normalized length may be `len(q)-1`) |
| `q` | Scattering vector (optional; reconstructed if missing) |
| `formfactor` | Label per row |
| `parameters_formfactor` | Serialized parameter dicts for regression |
| `SAXS_dataset_uncertainty` | Uncertainty channel (simulation output) |

## Simulation

| File | Purpose |
|------|---------|
| `simulation/simulation_config.yaml` | Default simulation dataset generation (`scripts/simulate_sas.py`). |
| `simulation/simulation_config_large.yaml` | Larger / alternate simulation preset. |

## Training (machine learning)

| File | Purpose |
|------|---------|
| `training/classification_config.yaml` | Train form-factor classifier (`scripts/train_formfactor_classifier.py`). |
| `training/regression_config.yaml` | XGBoost parameter regression (`scripts/train_parameter_regressor.py`). |
| `training/forward_ann_config.yaml` | Forward ANN (`scripts/train_ann_regressor.py`). |
| `training/inverse_ann_config.yaml` | Inverse ANN (`scripts/train_inverse_ann_regressor.py`). |

## Analysis

| File | Purpose |
|------|---------|
| `analysis/fit_config.yaml` | Physics-based fitting (`scripts/fit_sas.py`). |
| `analysis/experimental_SAS_analysis.yaml` | Full experimental pipeline: preprocess → classify → regress → MCMC (`scripts/experimental_SAS_analysis.py`). |
| `analysis/mcmc_config.yaml` | MCMC-focused analysis (`scripts/mcmc.py`). |

All of these are loaded with `SAXS_analysis.utils.configs.load_config`, which reads paths relative to the **repository root** (see that function in the **Python API** section).
