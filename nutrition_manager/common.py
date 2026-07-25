"""Shared paths and config loading for the nutrition model.

All dataset/asset paths in the config are interpreted relative to this
directory (`nutrition_manager/`), so the scripts work regardless of the
current working directory.
"""
from pathlib import Path

import yaml

# nutrition_manager/  (the directory containing this file)
PROJECT = Path(__file__).resolve().parent


def load_config(path: str = "configs/config.yaml") -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT / p
    with open(p) as f:
        return yaml.safe_load(f)


def resolve(*parts) -> Path:
    """Join path parts under the project dir (nutrition_manager/)."""
    return PROJECT.joinpath(*parts)
