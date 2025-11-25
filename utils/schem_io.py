"""Utilities for reading and writing Sponge .schem files."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import nbtlib
import numpy as np

from .palette import normalize_block_name

DATA_VERSION = 3120  # Default DataVersion; aligns with Minecraft 1.16.x.


def _resize_axis(arr: np.ndarray, target: int, axis: int) -> np.ndarray:
    """Downscale an array along a single axis via nearest-neighbor sampling."""
    if arr.shape[axis] == target:
        return arr
    indices = np.linspace(0, arr.shape[axis] - 1, num=target, dtype=int)
    return np.take(arr, indices, axis=axis)


def _center_grid(grid: np.ndarray, resolution: int) -> np.ndarray:
    """Center a (X, Y, Z) voxel grid inside a resolution^3 cube with padding."""
    target = np.zeros((resolution, resolution, resolution), dtype=grid.dtype)
    x, y, z = grid.shape
    x_offset = (resolution - x) // 2
    y_offset = (resolution - y) // 2
    z_offset = (resolution - z) // 2
    target[x_offset : x_offset + x, y_offset : y_offset + y, z_offset : z_offset + z] = grid
    return target


def schem_to_voxel(path: str, resolution: int, block2id: Dict[str, int]) -> np.ndarray:
    """Load a .schem file and convert it to a voxel grid with a global palette.

    Args:
        path: Path to the ``.schem`` file.
        resolution: Target cubic resolution for the voxel grid.
        block2id: Global palette mapping (block string → ID). This is mutated in-place
            to add new blocks encountered in the schematic.

    Returns:
        ``np.ndarray`` of shape ``(resolution, resolution, resolution)`` containing
        integer block IDs.
    """
    nbt = nbtlib.load(path)
    root = nbt.root

    width = int(root["Width"])
    height = int(root["Height"])
    length = int(root["Length"])

    local_palette = {int(v): normalize_block_name(k) for k, v in root["Palette"].items()}
    block_data = np.array(root["BlockData"], dtype=np.int32)
    grid = block_data.reshape((height, length, width))  # (Y, Z, X)
    grid = np.transpose(grid, (2, 0, 1))  # (X, Y, Z)

    global_grid = np.empty_like(grid, dtype=np.int16)
    for local_id in np.unique(grid):
        block_name = local_palette.get(int(local_id), "minecraft:air")
        if block_name not in block2id:
            block2id[block_name] = len(block2id)
        global_id = block2id[block_name]
        global_grid[grid == local_id] = global_id

    if any(dim > resolution for dim in global_grid.shape):
        resized = global_grid
        for axis in range(3):
            if resized.shape[axis] > resolution:
                resized = _resize_axis(resized, resolution, axis)
        global_grid = resized

    padded = _center_grid(global_grid, resolution)
    return padded.astype(np.int16)


def save_voxel_as_schem(voxel_ids: np.ndarray, id2block: Dict[int, str], out_path: str) -> None:
    """Export a voxel grid to a Sponge v2 ``.schem`` file.

    Args:
        voxel_ids: Array of shape ``(X, Y, Z)`` containing global block IDs.
        id2block: Mapping from integer ID to block name.
        out_path: Destination path for the schem file.
    """
    voxel_ids = np.asarray(voxel_ids)
    width, height, length = voxel_ids.shape

    block_names = [id2block.get(int(i), "minecraft:stone") for i in np.unique(voxel_ids)]
    palette = {name: idx for idx, name in enumerate(block_names)}

    block_entities = nbtlib.List[nbtlib.Compound]([])
    entities = nbtlib.List[nbtlib.Compound]([])

    # Map global IDs to local palette indices
    local_grid = np.empty_like(voxel_ids, dtype=np.int32)
    for global_id, block_name in id2block.items():
        if block_name not in palette:
            continue
        local_grid[voxel_ids == global_id] = palette[block_name]

    yzx = np.transpose(local_grid, (1, 2, 0))  # (Y, Z, X)
    block_data = nbtlib.ByteArray(yzx.flatten().tolist())

    root = nbtlib.Compound(
        {
            "Width": nbtlib.Short(width),
            "Height": nbtlib.Short(height),
            "Length": nbtlib.Short(length),
            "Version": nbtlib.Int(2),
            "DataVersion": nbtlib.Int(DATA_VERSION),
            "PaletteMax": nbtlib.Int(len(palette) - 1),
            "Palette": nbtlib.Compound({k: nbtlib.Int(v) for k, v in palette.items()}),
            "BlockData": block_data,
            "BlockEntities": block_entities,
            "Entities": entities,
            "Offset": nbtlib.List[nbtlib.Int]([0, 0, 0]),
        }
    )

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    schem = nbtlib.File(root)
    schem.save(out_file, gzipped=True)
