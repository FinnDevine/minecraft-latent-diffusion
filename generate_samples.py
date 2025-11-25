"""Sample voxel structures from the trained latent diffusion model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from models.autoencoder import VoxelAutoencoder
from models.latent_unet import LatentUNet3D
from train_latent_diffusion import prepare_noise_schedule
from utils.palette import invert_palette
from utils.schem_io import save_voxel_as_schem


def load_models(args: argparse.Namespace, device: torch.device):
    with open(args.palette_path, "r", encoding="utf-8") as f:
        block2id = json.load(f)

    autoencoder = VoxelAutoencoder(num_classes=len(block2id), resolution=args.resolution).to(device)
    autoencoder.load_state_dict(torch.load(args.autoencoder_path, map_location=device))
    autoencoder.eval()

    with torch.no_grad():
        dummy = torch.zeros(
            1, args.resolution, args.resolution, args.resolution, dtype=torch.long, device=device
        )
        latent_shape = autoencoder.encode(dummy).shape

    unet = LatentUNet3D(in_channels=latent_shape[1], base_channels=args.base_channels).to(device)
    unet.load_state_dict(torch.load(args.unet_path, map_location=device))
    unet.eval()

    return autoencoder, unet, block2id, latent_shape


def sample_latents(unet: LatentUNet3D, shape: torch.Size, timesteps: int, device: torch.device):
    betas, alphas, alpha_bars = prepare_noise_schedule(timesteps)
    betas, alphas, alpha_bars = betas.to(device), alphas.to(device), alpha_bars.to(device)

    z = torch.randn(shape, device=device)

    for t in reversed(range(timesteps)):
        t_batch = torch.full((shape[0],), float(t), device=device)
        eps = unet(z, t_batch)
        alpha_t = alphas[t]
        alpha_bar_t = alpha_bars[t]
        beta_t = betas[t]

        z = (z - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps) / torch.sqrt(alpha_t)
        if t > 0:
            z = z + torch.sqrt(beta_t) * torch.randn_like(z)
    return z


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    autoencoder, unet, block2id, latent_shape = load_models(args, device)
    id2block = invert_palette(block2id)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_shape = torch.Size((1, latent_shape[1], latent_shape[2], latent_shape[3], latent_shape[4]))

    for i in range(args.num_samples):
        with torch.no_grad():
            latent = sample_latents(unet, sample_shape, args.timesteps, device)
            logits = autoencoder.decode(latent)
            voxels = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        np.save(output_dir / f"sample_{i}.npy", voxels)
        save_voxel_as_schem(voxels, id2block, output_dir / f"sample_{i}.schem")
        print(f"Saved sample {i} to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples from latent diffusion model")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--autoencoder_path", default="models/ae_full.pt")
    parser.add_argument("--unet_path", default="models/latent_unet.pt")
    parser.add_argument("--palette_path", default="data/block_palette.json")
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()
    main(args)
