"""Palette utilities for mapping Minecraft block IDs to integer classes."""
from __future__ import annotations

from typing import Dict


def invert_palette(block2id: Dict[str, int]) -> Dict[int, str]:
    """Invert a block name to ID mapping.

    Args:
        block2id: Mapping from block string (e.g., "minecraft:stone") to integer ID.

    Returns:
        Reverse mapping from integer ID to block string.
    """
    return {idx: block for block, idx in block2id.items()}


def normalize_block_name(block: str) -> str:
    """Normalize a block name for palette consistency.

    Currently strips redundant whitespace and lowercases the string. The
    function is intentionally simple and assumes incoming palette entries
    already follow vanilla naming conventions such as ``minecraft:oak_planks``.
    """
    return block.strip().lower()
