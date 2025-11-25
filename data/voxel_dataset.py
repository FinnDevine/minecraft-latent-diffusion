"""Dataset wrappers for voxel and latent tensors."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class VoxelDataset(Dataset):
    """Dataset for voxel grids stored as a numpy archive."""

    def __init__(self, voxel_path: str):
        self.voxels = np.load(voxel_path)

    def __len__(self) -> int:  # type: ignore[override]
        return self.voxels.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:  # type: ignore[override]
        v = self.voxels[idx]
        return torch.from_numpy(v).long()


class LatentDataset(Dataset):
    """Dataset for precomputed latent tensors."""

    def __init__(self, latent_path: str):
        self.latents = np.load(latent_path)

    def __len__(self) -> int:  # type: ignore[override]
        return self.latents.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:  # type: ignore[override]
        z = self.latents[idx]
        return torch.from_numpy(z).float()
