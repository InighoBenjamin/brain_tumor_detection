"""
dataset.py — BraTS 2020 H5 Dataset Loading & Preprocessing
============================================================
Loads pre-processed BraTS 2020 data from HDF5 (.h5) slices.

Expected data format (Kaggle BraTS2020 pre-processed):
    data/
        volume_0_slice_0.h5
        volume_0_slice_1.h5
        ...
        meta_data.csv          (columns: slice_path, target, volume, slice)

    Each .h5 file contains:
        image : (240, 240, 4)  float64 — Z-score normalized (T1, T1ce, T2, FLAIR)
        mask  : (240, 240, 3)  uint8   — 3 non-overlapping binary channels:
                                          ch0 = NCR/NET (Necrotic / Non-Enhancing Tumor)
                                          ch1 = Edema   (Peritumoral Edema)
                                          ch2 = ET      (GD-Enhancing Tumor)

Hardware target: NVIDIA RTX 4050 (6 GB VRAM)
"""

import os
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# Class configuration
# ---------------------------------------------------------------------------
# The H5 mask has 3 binary channels. We convert them to a single multi-class
# label map:  0=Background, 1=NCR/NET, 2=Edema, 3=Enhancing Tumor
NUM_CLASSES = 4

CLASS_NAMES = {
    0: "Background",
    1: "Necrotic / Non-Enhancing Tumor (NCR/NET)",
    2: "Peritumoral Edema (ED)",
    3: "GD-Enhancing Tumor (ET)",
}


def mask_channels_to_class(mask_3ch: np.ndarray) -> np.ndarray:
    """
    Convert 3-channel binary mask (H, W, 3) → single-channel class map (H, W).
    Channels: ch0=NCR/NET(1), ch1=Edema(2), ch2=ET(3), else=Background(0).
    """
    out = np.zeros(mask_3ch.shape[:2], dtype=np.uint8)
    out[mask_3ch[:, :, 0] == 1] = 1   # NCR/NET
    out[mask_3ch[:, :, 1] == 1] = 2   # Edema
    out[mask_3ch[:, :, 2] == 1] = 3   # Enhancing Tumor
    return out


def class_to_mask_channels(class_map: np.ndarray) -> np.ndarray:
    """
    Inverse: single-channel class map (H, W) → 3-channel binary mask (H, W, 3).
    """
    h, w = class_map.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:, :, 0] = (class_map == 1).astype(np.uint8)
    out[:, :, 1] = (class_map == 2).astype(np.uint8)
    out[:, :, 2] = (class_map == 3).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Slice-level H5 Dataset
# ---------------------------------------------------------------------------
class BraTSDataset(Dataset):
    """
    PyTorch Dataset that loads individual 2D slices from pre-processed
    BraTS HDF5 files.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing .h5 files and meta_data.csv.
    split : str
        'train' or 'val'. Controls augmentation and data split.
    tumor_only : bool
        If True, only include slices where tumor is present (target=1).
    augment : bool
        Apply geometric + intensity augmentation (train only).
    val_ratio : float
        Fraction of patients reserved for validation.
    seed : int
        Random seed for reproducible train/val split.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        tumor_only: bool = True,
        augment: bool = True,
        val_ratio: float = 0.15,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.augment = augment and (split == "train")

        # Load metadata
        meta_path = os.path.join(data_dir, "meta_data.csv")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"meta_data.csv not found in {data_dir}. "
                "Expected the Kaggle BraTS2020 pre-processed dataset."
            )
        meta = pd.read_csv(meta_path)

        # Filter to tumor-containing slices only (if requested)
        if tumor_only:
            meta = meta[meta["target"] == 1].reset_index(drop=True)

        # Split by PATIENT (volume) to avoid data leakage
        all_volumes = sorted(meta["volume"].unique())
        rng = random.Random(seed)
        rng.shuffle(all_volumes)
        n_val = max(1, int(len(all_volumes) * val_ratio))

        if split == "val":
            selected_volumes = set(all_volumes[:n_val])
        else:
            selected_volumes = set(all_volumes[n_val:])

        meta = meta[meta["volume"].isin(selected_volumes)].reset_index(drop=True)

        # Build file list
        self.samples: List[Dict] = []
        for _, row in meta.iterrows():
            # The CSV stores paths like '/content/data/volume_X_slice_Y.h5'
            # We just need the filename
            filename = os.path.basename(row["slice_path"])
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                self.samples.append({
                    "path": filepath,
                    "volume": int(row["volume"]),
                    "slice_idx": int(row["slice"]),
                    "has_tumor": bool(row["target"]),
                })

        print(
            f"[{split.upper()}] {len(self.samples)} slices from "
            f"{len(selected_volumes)} patients"
            f" (tumor_only={tumor_only})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load H5 file
        with h5py.File(sample["path"], "r") as f:
            image = f["image"][:].astype(np.float32)  # (240, 240, 4)
            mask_3ch = f["mask"][:].astype(np.uint8)   # (240, 240, 3)

        # Convert 3-channel binary mask → single-channel class map
        mask = mask_channels_to_class(mask_3ch)  # (240, 240) with values 0-3

        # Transpose image to channels-first: (4, 240, 240)
        image = np.transpose(image, (2, 0, 1))

        # Convert to tensors
        image = torch.from_numpy(image.copy()).float()
        mask = torch.from_numpy(mask.copy()).long()

        # Data augmentation (train only)
        if self.augment:
            image, mask = self._augment(image, mask)

        return {
            "image": image,                          # (4, H, W) float32
            "mask": mask,                            # (H, W)    int64
            "volume": sample["volume"],
            "slice_idx": sample["slice_idx"],
        }

    # -------------------------------------------------------------------
    # MRI-appropriate augmentation
    # -------------------------------------------------------------------
    def _augment(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies random augmentations suitable for MRI:
          - Random horizontal flip
          - Random vertical flip
          - Random rotation (±15°)
          - Random intensity scaling (brightness jitter)
        Note: Color jitter / hue shifts are physically meaningless for MRI
        and are NOT applied.
        """
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask.unsqueeze(0)).squeeze(0)

        # Random vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask.unsqueeze(0)).squeeze(0)

        # Random rotation (±15°)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            image = TF.rotate(
                image, angle, interpolation=TF.InterpolationMode.BILINEAR
            )
            mask = TF.rotate(
                mask.unsqueeze(0).float(), angle,
                interpolation=TF.InterpolationMode.NEAREST,
            ).squeeze(0).long()

        # Random intensity scaling (per-channel)
        if random.random() > 0.5:
            scale = random.uniform(0.9, 1.1)
            image = image * scale

        return image, mask


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------
def get_dataloaders(
    data_dir: str,
    batch_size: int = 2,
    num_workers: int = 4,
    tumor_only: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader) ready for the training script.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing .h5 files and meta_data.csv.
    batch_size : int
        Actual batch size per GPU step.  Keep ≤ 2 for 6 GB VRAM.
    num_workers : int
        DataLoader worker processes.
    tumor_only : bool
        Only load slices containing tumor.
    """
    train_ds = BraTSDataset(
        data_dir, split="train", tumor_only=tumor_only, augment=True,
    )
    val_ds = BraTSDataset(
        data_dir, split="val", tumor_only=tumor_only, augment=False,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to BraTS H5 data directory"
    )
    args = parser.parse_args()

    train_loader, val_loader = get_dataloaders(args.data_dir, batch_size=2)
    batch = next(iter(train_loader))
    print("Image shape :", batch["image"].shape)   # (B, 4, 240, 240)
    print("Mask shape  :", batch["mask"].shape)     # (B, 240, 240)
    print("Mask classes :", torch.unique(batch["mask"]))
    print("Volumes     :", batch["volume"])
    print("Slice IDs   :", batch["slice_idx"])
