"""Hydra config-loading helper.

``load_config`` initialises Hydra, composes the base config with optional
overrides, and returns an ``OmegaConf`` object.
"""

import sys
from typing import List, Optional
from hydra import initialize, compose
from omegaconf import OmegaConf


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
    with initialize(version_base=None, config_path="../../../configs"):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    return cfg
