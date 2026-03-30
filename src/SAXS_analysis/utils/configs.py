"""Load YAML configuration files relative to the repository root."""

import yaml

import SAXS_analysis

ROOT_DIR = SAXS_analysis.ROOT_DIR


def load_config(config_path):
    """Load a YAML config file.

    Args:
        config_path: Path relative to ``ROOT_DIR`` (e.g. ``configs/simulation/simulation_config.yaml``).

    Returns:
        Parsed mapping from ``yaml.safe_load``.
    """
    with open(ROOT_DIR / config_path, "r") as file:
        return yaml.safe_load(file)
