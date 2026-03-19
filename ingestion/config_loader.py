"""
Config loader with COLAB_MODE path resolution.

All scripts import get_paths() from here to ensure consistent
path handling between local dev and Google Colab environments.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml


def load_config(config_name: str, config_dir: str | None = None) -> dict:
    """Load a YAML config file from the configs/ directory."""
    if config_dir is None:
        # Walk up until we find configs/ (works from any subdirectory)
        here = Path(__file__).resolve().parent
        for candidate in [here, here.parent, here.parent.parent]:
            if (candidate / "configs" / config_name).exists():
                config_dir = str(candidate / "configs")
                break
        if config_dir is None:
            raise FileNotFoundError(
                f"Could not locate configs/{config_name}. "
                "Run from the project root or set config_dir explicitly."
            )

    config_path = Path(config_dir) / config_name
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_paths(colab_mode: bool = False) -> dict[str, Path]:
    """
    Return all data paths as Path objects, resolved for the current environment.

    Parameters
    ----------
    colab_mode:
        If True, prepend the Google Drive mount path to all data paths.
        Set this flag at the top of each training/ingestion script.
        Example usage in a Colab notebook or script:
            COLAB_MODE = True  # ← flip this when running on Colab
            paths = get_paths(COLAB_MODE)

    Returns
    -------
    dict mapping path keys (e.g. "raw_lidar") to resolved Path objects.
    """
    cfg = load_config("data_sources.yaml")
    path_cfg = cfg["paths"]

    root = (
        Path(path_cfg["colab_root"])
        if colab_mode
        else Path(path_cfg["local_root"]).resolve()
    )

    def resolve(rel: str) -> Path:
        p = root / rel
        return p

    return {
        "root": root,
        # Raw inputs
        "raw_lidar": resolve(path_cfg["raw"]["lidar"]),
        "raw_imagery": resolve(path_cfg["raw"]["imagery"]),
        "raw_landfire": resolve(path_cfg["raw"]["landfire"]),
        "raw_dem": resolve(path_cfg["raw"]["dem"]),
        "raw_weather": resolve(path_cfg["raw"]["weather"]),
        "raw_parcels": resolve(path_cfg["raw"]["parcels"]),
        # Processed outputs
        "processed": resolve(path_cfg["processed"]["base"]),
        "processed_terrain": resolve(path_cfg["processed"]["terrain"]),
        "processed_vegetation": resolve(path_cfg["processed"]["vegetation"]),
        "processed_structure": resolve(path_cfg["processed"]["structure"]),
        "processed_flood": resolve(path_cfg["processed"]["flood"]),
        "processed_twins": resolve(path_cfg["processed"]["twins"]),
        "processed_simulations": resolve(path_cfg["processed"]["simulations"]),
        # Validation
        "validation": resolve(path_cfg["validation"]["base"]),
    }


def ensure_dirs(paths: dict[str, Path]) -> None:
    """Create all directories in the paths dict if they don't exist."""
    for p in paths.values():
        if not p.suffix:  # Only create directories, not files
            p.mkdir(parents=True, exist_ok=True)


def get_study_area(config: dict | None = None) -> dict:
    """Return study area bbox and CRS from config."""
    cfg = config or load_config("data_sources.yaml")
    return cfg["study_area"]
