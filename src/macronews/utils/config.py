"""
Shared helpers for loading asset universe configuration.
"""

from pathlib import Path

import yaml

from macronews.config.paths import ASSET_UNIVERSE_YAML


def load_asset_universe(yaml_path: Path = ASSET_UNIVERSE_YAML) -> dict:
    """Load asset universe YAML and return the full dict."""
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def get_asset_symbols(yaml_path: Path = ASSET_UNIVERSE_YAML) -> list[str]:
    """Return asset symbol keys from the universe YAML."""
    return list(load_asset_universe(yaml_path).keys())
