"""Hydra config-loading helper.

``load_config`` initialises Hydra, composes the base config with optional
overrides, and returns an ``OmegaConf`` object.
"""

import sys
from pathlib import Path
from typing import List, Optional
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf


def _config_dir() -> Path:
    """Find bundled configs, falling back to the source-checkout config tree."""
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    if not config_dir.is_dir():
        config_dir = Path(__file__).resolve().parents[3] / "configs"
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Hydra config directory not found: {config_dir}")
    return config_dir


def load_config(
    config_name: str = "config", overrides: Optional[List[str]] = None
) -> OmegaConf:
    """Compose and return a Hydra config.

    Parameters
    ----------
    config_name : str
        Name of the config group/entry to compose (default ``"config"``).
    overrides : Optional[List[str]]
        Optional list of dotted ``key=value`` override strings.

    Returns
    -------
    omegaconf.DictConfig
        The composed configuration object.
    """
    with initialize_config_dir(
        version_base=None, config_dir=str(_config_dir())
    ):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    return cfg
