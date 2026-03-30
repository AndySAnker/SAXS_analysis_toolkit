# Sphinx configuration for SAXS Analysis Toolkit (HTML + DOCX via docxbuilder).

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

# Repository root (parent of docs/) and package source (src/)
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for p in (SRC, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

project = "SAXS Analysis Toolkit"
author = "Andy Sode Anker"
copyright = f"{datetime.now(tz=timezone.utc).year}, {author}"

try:
    from SAXS_analysis import __version__ as release
except ImportError:
    release = "0.0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "docxbuilder",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# Allow `sphinx-build` without installing all runtime deps (torch, sasmodels, …).
autodoc_mock_imports = [
    "bayes_opt",
    "bumps",
    "corner",
    "emcee",
    "h5py",
    "joblib",
    "matplotlib",
    "matplotlib.pyplot",
    "pandas",
    "sasmodels",
    "sasmodels.bumps_model",
    "sasmodels.core",
    "sasmodels.data",
    "sasmodels.direct_model",
    "sasmodels.resolution",
    "seaborn",
    "sklearn",
    "sklearn.model_selection",
    "sklearn.preprocessing",
    "torch",
    "torch.nn",
    "tqdm",
    "xgboost",
    "xgboost.core",
    "yaml",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_static_path = ["_static"]

# myst_parser registers ``.md`` → ``markdown``; keep ``.rst`` for API reference (autodoc).
source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

myst_enable_extensions = ["colon_fence", "deflist"]

# docxbuilder: Word output (https://github.com/amedama41/docxbuilder)
# Build the README-backed user guide, not the full API (DOCX is best-effort for large autodoc).
docx_documents = [
    (
        "user_guide",
        "SAXS_Analysis_Toolkit.docx",
        {
            "title": project,
            "creator": author,
            "subject": "SAXS Analysis Toolkit user guide",
        },
        True,
    ),
]
