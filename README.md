[COOL icons]

[ABSTRACT]


## Getting Started

### Create an environment and install this as a package

If you e.g. use conda, we recommend creating a conda environment for this project:

```bash
conda create -n SAXS_analysis python=3.10
conda activate SAXS_analysis
pip install -e .
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

These scripts work with the accompanying config file in the ```configs``` folder in the following way:
```
python scripts/train_formfactor_classifier.py --config configs/classification_config.yaml
```

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
