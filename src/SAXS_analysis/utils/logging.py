"""Logging setup for scripts driven by YAML configs."""

import logging
import os

import SAXS_analysis

ROOT_DIR = SAXS_analysis.ROOT_DIR


def setup_logging(config):
    """Configure the root logger from a loaded config dict.

    Expects keys ``experiment_name``, ``experiment_id``, ``verbose`` (0=warnings only,
    1=+stdout info, 2+=debug), and optional ``verbose`` > 0 to also log to the console.

    Writes to ``logs/<experiment_name> - <experiment_id>/<experiment_id>.log`` under ``ROOT_DIR``.

    Args:
        config: Mapping from ``load_config`` (see training/analysis YAML files).

    Returns:
        Logger for this module.
    """
    log_dir = os.path.join(
        ROOT_DIR / "logs", f"{config['experiment_name']} - {config['experiment_id']}"
    )
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"{config['experiment_id']}.log")

    if config["verbose"] == 0:
        log_level = logging.WARNING
    elif config["verbose"] == 1:
        log_level = logging.INFO
    else:
        log_level = logging.DEBUG

    handlers = [logging.FileHandler(log_file)]
    if config["verbose"] > 0:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )
    return logging.getLogger(__name__)
