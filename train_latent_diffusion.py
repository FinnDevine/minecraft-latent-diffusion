"""Train a latent diffusion model over autoencoder latents."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from data.voxel_dataset import LatentDataset, VoxelDataset
from models.autoencoder import VoxelAutoencoder
from models.latent_unet import LatentUNet3D


def prepare_noise_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    betas = torch.linspace(beta_start, beta_end, timesteps)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def maybe_precompute_latents(
    autoencoder: VoxelAutoencoder,
    voxel_path: str,
    latent_path: str,
    batch_size: int,
    device: torch.device,
) -> None:
    if Path(latent_path).exists():
        return

    dataset = VoxelDataset(voxel_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    latents = []
    autoencoder.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            z = autoencoder.encode(batch)
            latents.append(z.cpu().numpy())

    Path(latent_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(latent_path, np.concatenate(latents, axis=0))


def train_latent_diffusion(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    with open(args.palette_path, "r", encoding="utf-8") as f:
        block2id = json.load(f)

    autoencoder = VoxelAutoencoder(num_classes=len(block2id), resolution=args.resolution).to(device)
    autoencoder.load_state_dict(torch.load(args.autoencoder_path, map_location=device))
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad = False

    maybe_precompute_latents(
        autoencoder, args.voxel_path, args.latent_path, args.batch_size, device
    )

    latent_dataset = LatentDataset(args.latent_path)
    latent_loader = DataLoader(latent_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    if len(latent_dataset) == 0:
        raise RuntimeError("Latent dataset is empty. Ensure voxel data has been encoded.")

    with torch.no_grad():
        dummy = torch.zeros(
            1, args.resolution, args.resolution, args.resolution, dtype=torch.long, device=device
        )
        latent_shape = autoencoder.encode(dummy).shape
    in_channels = latent_shape[1]

    unet = LatentUNet3D(in_channels=in_channels, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr)

    betas, alphas, alpha_bars = prepare_noise_schedule(args.timesteps)
    betas = betas.to(device)
    alphas = alphas.to(device)
    alpha_bars = alpha_bars.to(device)

    Path("models").mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        unet.train()
        total_loss = 0.0

        for z0 in latent_loader:
            z0 = z0.to(device)
            noise = torch.randn_like(z0)
            t = torch.randint(0, args.timesteps, (z0.size(0),), device=device)
            alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1, 1)

            zt = torch.sqrt(alpha_bar_t) * z0 + torch.sqrt(1 - alpha_bar_t) * noise
            noise_pred = unet(zt, t.float())

            loss = F.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(latent_loader)
        print(f"Epoch {epoch}: loss={avg_loss:.6f}")

    torch.save(unet.state_dict(), Path("models") / "latent_unet.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train latent diffusion model")
    parser.add_argument("--voxel_path", default="data/all_voxels.npy")
    parser.add_argument("--latent_path", default="data/all_latents.npy")
    parser.add_argument("--palette_path", default="data/block_palette.json")
    parser.add_argument("--autoencoder_path", default="models/ae_full.pt")
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()
    train_latent_diffusion(args)
