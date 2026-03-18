[COOL icons]

[ABSTRACT]


## Getting Started

### Create an environment and install this as a package

If you e.g. use conda, we recommend creating a conda environment for this project:

```bash
conda create -n SAXS_analysis python=3.10
conda activate SAXS_analysis
python -m pip install -e .
```

Otherwise, if you use `uv`, you can install this package by running

```bash
uv sync
source .venv/bin/activate
```

You can make sure you're all set by printing the version:

```bash
python scripts/see_version.py
```

## Structure of the repository

```bash
├── configs
├── data
│   ├── experimental
│   ├── simulated
├── scripts
│   ├── experimental_SAS_analysis.py
│   ├── fit_sas.py
│   ├── simulate_sas.py
│   ├── train_formfactor_classifier.py
│   ├── train_parameter_regressor.py
├── src
│   └── SAXS_analysis
│       ├── classification 
│       ├── data_processing  
│       ├── fitting
│       ├── regression
│       ├── simulation
│       ├── utils
│       ├── visualization
```

## How to use SAXS_analysis

In the scripts folder, we share examples of how to:
- Simulated SAXS data
- Fit SAXS data
- Train a formfactor classification ML model
- Train a regression model to estimate model parameters such as size, polydispersity, etc.
- Analyse an experimental dataset using the above tools

All scripts are driven by YAML config files in `configs/`, grouped by purpose:

- `configs/simulation/`
- `configs/training/`
- `configs/analysis/`

For example:

```bash
python scripts/train_formfactor_classifier.py --config configs/training/classification_config.yaml
```

### Simulating data

Simulate SAXS datasets to an HDF5 file in `data/simulated/`:

```bash
python scripts/simulate_sas.py --config configs/simulation/simulation_config.yaml
```

**Changing which parameters are simulated**:

- **Parameter bounds / distributions** (used when sampling parameters for simulation and for fitting): `src/SAXS_analysis/utils/parameter_ranges.py` (`param_ranges`).
- **Which parameters belong to each form factor model**: `src/SAXS_analysis/utils/formfactors.py` (`formfactor_params`).

### Training models

1) **Train a form-factor classifier (XGBoost)**

```bash
python scripts/train_formfactor_classifier.py --config configs/training/classification_config.yaml
```

2) **Train a parameter regressor**

- XGBoost regressor:

```bash
python scripts/train_parameter_regressor.py --config configs/training/regression_config.yaml
```

- Forward ANN regressor (PyTorch):

```bash
python scripts/train_ann_regressor.py --config configs/training/forward_ann_config.yaml
```

- Inverse ANN regressor (PyTorch):

```bash
python scripts/train_inverse_ann_regressor.py --config configs/training/inverse_ann_config.yaml
```

The training scripts save models under `models/` and may also write scalers under `scalers/` and plots under `plots/`.

**Changing which parameters the regressors learn**:

- The regressors pull their target parameter list from `src/SAXS_analysis/utils/formfactors.py` (`formfactor_params`). If you add/remove parameters for a form factor, update that mapping (and ensure the parameter exists in `param_ranges`).

### Analysing data

There are two main analysis entry points:

1) **Physics-based fitting of (simulated) datasets** using sasmodels/bumps:

```bash
python scripts/fit_sas.py --config configs/analysis/fit_config.yaml
```

2) **Experimental data analysis** (preprocess → classify → regress → MCMC inference):

```bash
python scripts/experimental_SAS_analysis.py --config configs/analysis/experimental_SAS_analysis.yaml
```

This pipeline expects the model/scaler paths in `configs/analysis/experimental_SAS_analysis.yaml` to point to existing files (i.e. trained models and saved scalers).

## License

This project is licensed under [...].

## Cite us!

As an APA reference: ...

Or using `bibtex`:

```bibtex
...
```

# Contributing to the software

We welcome contributions to our software! To contribute, please follow these steps:

1. Fork the repository.
2. Make your changes in a new branch.
3. Submit a pull request.

We'll review your changes and merge them if they meet our quality standards, including passing all unit tests. To ensure that your changes pass the unit tests, please run the tests locally before submitting your pull request. You can also view the test results on our GitHub repository using GitHub Actions.

## Reporting issues

If you encounter any issues or problems with our software, please report them by opening an issue on our GitHub repository. Please include as much detail as possible, including steps to reproduce the issue and any error messages you received.

## Seeking support

If you need help using our software, please reach out to us on our GitHub repository. We'll do our best to assist you and answer any questions you have.
