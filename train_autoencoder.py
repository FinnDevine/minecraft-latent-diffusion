"""Training script for the voxel autoencoder."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.voxel_dataset import VoxelDataset
from models.autoencoder import VoxelAutoencoder
from utils.voxelize import build_voxel_dataset_from_schems


def _hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def _schem_tree_signature(schem_root: Path) -> str:
    hasher = hashlib.sha256()
    if not schem_root.exists():
        return hasher.hexdigest()

    for schem_path in sorted(schem_root.rglob("*.schem")):
        stat = schem_path.stat()
        hasher.update(str(schem_path.relative_to(schem_root)).encode())
        hasher.update(str(stat.st_mtime_ns).encode())
        hasher.update(str(stat.st_size).encode())
    return hasher.hexdigest()


def _current_voxel_metadata(csv_path: Path, schem_root: Path, resolution: int) -> Dict[str, Any]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at {csv_path}")
    return {
        "csv_path": str(csv_path.resolve()),
        "schem_root": str(schem_root.resolve()),
        "resolution": resolution,
        "csv_hash": _hash_file(csv_path),
        "schem_signature": _schem_tree_signature(schem_root),
    }


def ensure_voxel_data(csv_path: str, schem_root: str, resolution: int) -> tuple[Path, dict]:
    vox_path = Path("data/all_voxels.npy")
    palette_path = Path("data/block_palette.json")
    metadata_path = Path("data/voxel_metadata.json")

    requested_meta = _current_voxel_metadata(Path(csv_path), Path(schem_root), resolution)
    existing_meta = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            existing_meta = json.load(f)

    should_rebuild = not (vox_path.exists() and palette_path.exists() and requested_meta == existing_meta)
    if should_rebuild:
        voxels, block2id = build_voxel_dataset_from_schems(csv_path, schem_root, resolution)
        if voxels.size == 0:
            raise RuntimeError("No voxel data generated from schematics.")
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(requested_meta, f, indent=2)
    else:
        with open(palette_path, "r", encoding="utf-8") as f:
            block2id = json.load(f)
    return vox_path, block2id


def train_autoencoder(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    vox_path, block2id = ensure_voxel_data(args.csv_path, args.schem_root, args.resolution)
    dataset = VoxelDataset(str(vox_path))
    if len(dataset) == 0:
        raise RuntimeError("Voxel dataset is empty. Ensure schem files are available.")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = VoxelAutoencoder(num_classes=len(block2id), resolution=args.resolution).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    Path("models").mkdir(parents=True, exist_ok=True)
    Path("outputs").mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_voxels = 0

        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits, _ = model(batch)
            loss = F.cross_entropy(logits, batch)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = logits.argmax(dim=1)
                total_correct += (preds == batch).sum().item()
                total_voxels += batch.numel()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        acc = total_correct / total_voxels
        print(f"Epoch {epoch}: loss={avg_loss:.4f} acc={acc:.4f}")

        # Save a few reconstructions for inspection
        model.eval()
        with torch.no_grad():
            sample_batch = next(iter(dataloader)).to(device)
            recon_logits, _ = model(sample_batch)
            recon = recon_logits.argmax(dim=1).cpu().numpy()
            np.save(Path("outputs") / f"recon_epoch_{epoch}.npy", recon)

    torch.save(model.state_dict(), Path("models") / "ae_full.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the voxel autoencoder")
    parser.add_argument("--csv_path", default="data/processed_build_dataframe.csv")
    parser.add_argument("--schem_root", default="data/schems")
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()
    train_autoencoder(args)
