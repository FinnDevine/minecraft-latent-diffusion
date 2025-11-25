"""Voxelisation utilities for PlanetMinecraft schematics."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .schem_io import schem_to_voxel


def load_pmc_csv(csv_path: str) -> pd.DataFrame:
    """Load the PlanetMinecraft CSV with minimal preprocessing."""
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV not found at {csv_file}")
    return pd.read_csv(csv_file)


def _parse_schem_paths(paths: str) -> List[str]:
    """Parse the stringified list of schem paths from the CSV."""
    if isinstance(paths, str):
        try:
            parsed = ast.literal_eval(paths)
            return [str(p) for p in parsed]
        except (ValueError, SyntaxError):
            return []
    return []


def build_voxel_dataset_from_schems(
    csv_path: str, schem_root: str, resolution: int
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Voxelise schems listed in the PMC CSV and build a global palette.

    Args:
        csv_path: Path to ``processed_build_dataframe.csv``.
        schem_root: Directory containing processed ``.schem`` files.
        resolution: Target cubic resolution for voxel grids.

    Returns:
        Tuple containing the voxel tensor of shape ``(N, resolution, resolution, resolution)``
        and the global palette mapping.
    """
    df = load_pmc_csv(csv_path)
    schem_dir = Path(schem_root)

    block2id: Dict[str, int] = {"minecraft:air": 0}
    voxels = []

    for _, row in df.iterrows():
        schem_paths = _parse_schem_paths(row.get("PROCESSED_PATHS"))
        for schem_name in schem_paths:
            schem_path = schem_dir / schem_name
            if not schem_path.exists():
                continue
            voxel = schem_to_voxel(str(schem_path), resolution, block2id)
            voxels.append(voxel)

    voxel_array = (
        np.stack(voxels).astype(np.int16)
        if voxels
        else np.empty((0, resolution, resolution, resolution), dtype=np.int16)
    )

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "all_voxels.npy", voxel_array)
    with open(data_dir / "block_palette.json", "w", encoding="utf-8") as f:
        json.dump(block2id, f, indent=2)

    return voxel_array, block2id
