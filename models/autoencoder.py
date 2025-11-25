"""3D voxel autoencoder for Minecraft structures."""
from __future__ import annotations

import torch
import torch.nn as nn


class VoxelAutoencoder(nn.Module):
    def __init__(self, num_classes: int, resolution: int, embed_dim: int = 8):
        super().__init__()
        self.num_classes = num_classes
        self.resolution = resolution

        self.embedding = nn.Embedding(num_classes, embed_dim)

        self.encoder = nn.Sequential(
            nn.Conv3d(embed_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, num_classes, kernel_size=1),
        )

    def encode(self, x_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(x_ids)  # (B, D, H, W, E)
        embeddings = embeddings.permute(0, 4, 1, 2, 3)  # (B, E, D, H, W)
        latent = self.encoder(embeddings)
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, x_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x_ids)
        logits = self.decode(latent)
        return logits, latent
