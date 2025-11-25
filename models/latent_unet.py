"""3D UNet backbone for latent diffusion."""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal timestep embeddings."""
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, device=device, dtype=torch.float32) / half
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

        self.shortcut = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(time_emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.shortcut(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_dim)
        self.down = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.res(x, time_emb)
        return self.down(h), h


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.up(x)
        h = torch.cat([h, skip], dim=1)
        return self.res(h, time_emb)


class LatentUNet3D(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 64, time_dim: int = 256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.in_block = ResidualBlock(in_channels, base_channels, time_dim)
        self.down1 = DownBlock(base_channels, base_channels * 2, time_dim)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, time_dim)

        self.mid = ResidualBlock(base_channels * 4, base_channels * 4, time_dim)

        self.up1 = UpBlock(base_channels * 4, base_channels * 2, time_dim)
        self.up2 = UpBlock(base_channels * 2, base_channels, time_dim)

        self.out_conv = nn.Conv3d(base_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = timestep_embedding(timesteps, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)

        h1 = self.in_block(x, t_emb)
        h2, skip1 = self.down1(h1, t_emb)
        h3, skip2 = self.down2(h2, t_emb)

        h_mid = self.mid(h3, t_emb)

        h = self.up1(h_mid, skip2, t_emb)
        h = self.up2(h, skip1, t_emb)
        return self.out_conv(h)
