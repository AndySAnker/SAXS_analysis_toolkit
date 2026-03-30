"""SAXS Analysis Toolkit — small-angle scattering simulation, fitting, and ML-assisted analysis.

Subpackages include ``simulation`` (sasmodels datasets), ``data_processing``, ``classification`` /
``regression`` (XGBoost), ``forward_ann`` / ``inverse_ann`` (PyTorch), ``fitting`` (bumps/sasmodels),
``mcmc`` (emcee + neural surrogates), and ``visualization``.

``ROOT_DIR`` is the repository root (parent of ``src/``), used for resolving config paths and data.
"""

from pathlib import Path

__version__ = "0.0.1"
ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
