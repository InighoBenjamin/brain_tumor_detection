"""
inference.py — Prediction, Visualization & Tumor Analysis
==========================================================
Loads a trained Attention U-Net checkpoint, runs inference on BraTS H5 slices,
and produces:
    1. Predicted segmentation mask
    2. Visualization: MRI slice with colored mask overlay + contour
    3. Tumor report: size in pixels, affected brain region, class areas

Usage (single slice):
    python inference.py \
        --checkpoint output/checkpoints/best_model.pth \
        --h5_file "C:/path/to/data/volume_100_slice_80.h5" \
        --output_dir output/predictions

Usage (all tumor slices for a patient volume):
    python inference.py \
        --checkpoint output/checkpoints/best_model.pth \
        --data_dir "C:/path/to/data" \
        --volume_id 100 \
        --output_dir output/predictions
"""

import os
import argparse
import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import h5py
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (works on headless servers)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2

from model import AttentionUNet, NUM_CLASSES
from dataset import mask_channels_to_class, CLASS_NAMES


# ======================================================================
# Color palette for the 4 classes
# ======================================================================
CLASS_COLORS = {
    0: (0, 0, 0, 0),           # Background — transparent
    1: (255, 0, 0, 140),       # NCR/NET   — red
    2: (0, 255, 0, 140),       # Edema     — green
    3: (255, 255, 0, 140),     # Enhancing — yellow
}

# Rough brain region mapping based on centroid position (axial view)
REGION_MAP = {
    "frontal": "Frontal Lobe",
    "parietal": "Parietal Lobe",
    "temporal": "Temporal Lobe",
    "occipital": "Occipital Lobe",
    "central": "Central / Deep Brain",
}


# ======================================================================
# Loading utilities
# ======================================================================

def load_model(checkpoint_path: str, device: torch.device) -> AttentionUNet:
    """Load trained Attention U-Net from a checkpoint file."""
    model = AttentionUNet(in_channels=4, out_channels=NUM_CLASSES).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    if "epoch" in ckpt:
        print(f"  Checkpoint epoch : {ckpt['epoch']}")
    if "val_dice" in ckpt:
        print(f"  Val Dice scores  : {ckpt['val_dice']}")
    return model


def load_h5_slice(h5_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a single 2D slice from a BraTS H5 file.

    Returns
    -------
    image : (4, H, W) float32  — 4-modality MRI (channels first)
    gt_mask : (H, W) uint8     — ground-truth class map (0-3)
    """
    with h5py.File(h5_path, "r") as f:
        image = f["image"][:].astype(np.float32)  # (240, 240, 4)
        mask_3ch = f["mask"][:].astype(np.uint8)   # (240, 240, 3)

    image = np.transpose(image, (2, 0, 1))         # (4, 240, 240)
    gt_mask = mask_channels_to_class(mask_3ch)      # (240, 240)
    return image, gt_mask


# ======================================================================
# Inference
# ======================================================================

@torch.no_grad()
def predict_slice(
    model: AttentionUNet,
    image: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    Run the model on a single 2D slice.

    Parameters
    ----------
    model  : trained AttentionUNet
    image  : (4, H, W) float32 array
    device : torch device

    Returns
    -------
    pred_mask : (H, W) uint8 — predicted class labels (0–3)
    """
    tensor = torch.from_numpy(image).unsqueeze(0).to(device)  # (1, 4, H, W)

    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        logits = model(tensor)  # (1, C, H, W)

    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred


# ======================================================================
# Tumor Analysis Report
# ======================================================================

def analyze_tumor(pred_mask: np.ndarray) -> Dict:
    """
    Generate a structured report from a predicted segmentation mask.

    Since the H5 dataset doesn't include voxel spacing metadata, we report
    area in pixels. BraTS native resolution is 1mm × 1mm, so pixel count
    approximately equals area in mm².

    Returns
    -------
    dict with:
        - total_tumor_pixels    : int
        - total_tumor_area_mm2  : float (assuming 1mm × 1mm spacing)
        - per_class             : dict of class → {pixels, area_mm2}
        - tumor_centroid        : (row, col) or None
        - estimated_region      : str — rough brain region
        - whole_tumor_present   : bool
    """
    # BraTS standard voxel spacing is 1mm × 1mm × 1mm
    pixel_area_mm2 = 1.0

    report = {
        "total_tumor_pixels": 0,
        "total_tumor_area_mm2": 0.0,
        "per_class": {},
        "tumor_centroid": None,
        "estimated_region": "N/A",
        "whole_tumor_present": False,
    }

    tumor_mask = pred_mask > 0
    total_pixels = int(tumor_mask.sum())
    report["total_tumor_pixels"] = total_pixels
    report["total_tumor_area_mm2"] = round(total_pixels * pixel_area_mm2, 2)
    report["whole_tumor_present"] = total_pixels > 0

    # Per-class breakdown
    for cls_id in [1, 2, 3]:
        px = int((pred_mask == cls_id).sum())
        report["per_class"][CLASS_NAMES[cls_id]] = {
            "pixels": px,
            "area_mm2": round(px * pixel_area_mm2, 2),
        }

    # Centroid & rough region estimation
    if total_pixels > 0:
        ys, xs = np.where(tumor_mask)
        cy, cx = int(ys.mean()), int(xs.mean())
        report["tumor_centroid"] = (cy, cx)

        h, w = pred_mask.shape
        rel_y, rel_x = cy / h, cx / w
        report["estimated_region"] = _estimate_region(rel_y, rel_x)

    return report


def _estimate_region(rel_y: float, rel_x: float) -> str:
    """
    Heuristic brain region estimation from normalized centroid position
    in an axial MRI slice.  This is a rough approximation — for precise
    localization, atlas-based registration (e.g., MNI152) is needed.
    """
    if rel_y < 0.35:
        return REGION_MAP["frontal"]
    elif rel_y > 0.70:
        return REGION_MAP["occipital"]
    elif rel_x < 0.30 or rel_x > 0.70:
        return REGION_MAP["temporal"]
    elif rel_y < 0.55:
        return REGION_MAP["parietal"]
    else:
        return REGION_MAP["central"]


# ======================================================================
# Visualization
# ======================================================================

def visualize_prediction(
    image: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: Optional[np.ndarray] = None,
    report: Optional[Dict] = None,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Create a publication-quality figure showing:
        Col 1: FLAIR modality (grayscale)
        Col 2: Predicted mask contour overlay on FLAIR
        Col 3: Ground-truth overlay (if available)

    Tumor contours are drawn using OpenCV for sharp outlines.
    """
    # Use FLAIR channel (index 3) — most informative for tumor visualization
    flair = image[3]
    flair_norm = _normalize_for_display(flair)

    n_cols = 3 if gt_mask is not None and gt_mask.max() > 0 else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))

    # --- Col 1: Raw FLAIR ---
    axes[0].imshow(flair_norm, cmap="gray")
    axes[0].set_title("FLAIR MRI", fontsize=14)
    axes[0].axis("off")

    # --- Col 2: Prediction overlay with contours ---
    contour_pred = _draw_contours(flair_norm, pred_mask)
    axes[1].imshow(contour_pred)
    axes[1].set_title("Predicted Segmentation", fontsize=14)
    axes[1].axis("off")

    # --- Col 3: Ground truth overlay (optional) ---
    if n_cols == 3:
        contour_gt = _draw_contours(flair_norm, gt_mask)
        axes[2].imshow(contour_gt)
        axes[2].set_title("Ground Truth", fontsize=14)
        axes[2].axis("off")

    # Legend
    patches = [
        mpatches.Patch(color=np.array(CLASS_COLORS[c][:3]) / 255,
                       label=CLASS_NAMES[c])
        for c in [1, 2, 3]
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=11,
               frameon=True, fancybox=True)

    # Report text
    if report:
        text = (
            f"Total Tumor: {report['total_tumor_area_mm2']:.1f} mm²  "
            f"({report['total_tumor_pixels']} px)\n"
            f"Region: {report['estimated_region']}"
        )
        fig.suptitle(text, fontsize=12, y=0.02)

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def _normalize_for_display(img: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 255] uint8 for display."""
    img = img.copy()
    lo, hi = np.percentile(img[img > 0], [1, 99]) if img.max() > 0 else (0, 1)
    img = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)
    return (img * 255).astype(np.uint8)


def _draw_contours(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Draw colored contours + semi-transparent fill for each tumor class."""
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    overlay = rgb.copy()

    for cls_id in [1, 2, 3]:
        binary = (mask == cls_id).astype(np.uint8)
        color = CLASS_COLORS[cls_id][:3]

        # Semi-transparent fill
        overlay[binary == 1] = color

        # Sharp contour outline
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(rgb, contours, -1, color, thickness=2)

    # Blend fill + original
    blended = cv2.addWeighted(overlay, 0.3, rgb, 0.7, 0)
    return blended


# ======================================================================
# Main CLI
# ======================================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    # Collect H5 files to process
    h5_files = []
    if args.h5_file:
        h5_files = [args.h5_file]
    elif args.data_dir and args.volume_id is not None:
        pattern = os.path.join(args.data_dir, f"volume_{args.volume_id}_slice_*.h5")
        h5_files = sorted(glob.glob(pattern))
        print(f"Found {len(h5_files)} slices for volume {args.volume_id}")
    else:
        print("ERROR: Provide either --h5_file or (--data_dir + --volume_id)")
        return []

    all_reports = []

    for h5_path in h5_files:
        filename = os.path.basename(h5_path)
        image, gt_mask = load_h5_slice(h5_path)
        pred_mask = predict_slice(model, image, device)
        report = analyze_tumor(pred_mask)
        report["file"] = filename
        all_reports.append(report)

        if report["whole_tumor_present"]:
            save_path = os.path.join(
                args.output_dir, filename.replace(".h5", ".png")
            )
            report["save_path"] = os.path.abspath(save_path)
            visualize_prediction(
                image, pred_mask,
                gt_mask=gt_mask if gt_mask.max() > 0 else None,
                report=report, save_path=save_path,
            )

    # Print summary report
    print("\n" + "=" * 65)
    print("TUMOR ANALYSIS REPORT")
    print("=" * 65)
    tumor_reports = [r for r in all_reports if r["whole_tumor_present"]]
    for r in tumor_reports:
        print(
            f"  {r['file']:40s}  |  "
            f"Area: {r['total_tumor_area_mm2']:8.1f} mm²  |  "
            f"Region: {r['estimated_region']}"
        )
    if not tumor_reports:
        print("  No tumor detected in any slice.")
    print("=" * 65)

    # Structured output for Java / external callers (--output_json flag)
    if args.output_json and all_reports:
        # Aggregate across all slices: sum areas, pick dominant region
        total_area = sum(r["total_tumor_area_mm2"] for r in all_reports)
        detected   = any(r["whole_tumor_present"] for r in all_reports)
        regions    = [r["estimated_region"] for r in all_reports if r["whole_tumor_present"]]
        region     = max(set(regions), key=regions.count) if regions else "N/A"

        # Best wt/tc/et dice from saved reports (printed per-slice if available)
        wt = all_reports[0].get("wt_dice", 0.0)
        tc = all_reports[0].get("tc_dice", 0.0)
        et = all_reports[0].get("et_dice", 0.0)

        # Find the saved mask/heatmap paths (first tumor slice)
        mask_path    = ""
        heatmap_path = ""
        for r in all_reports:
            if r["whole_tumor_present"] and r.get("save_path"):
                mask_path    = r["save_path"]
                heatmap_path = r["save_path"]  # same file; split if needed
                break

        print(f"WT_DICE={wt:.4f}")
        print(f"TC_DICE={tc:.4f}")
        print(f"ET_DICE={et:.4f}")
        print(f"TUMOR_DETECTED={1 if detected else 0}")
        print(f"TUMOR_AREA_MM2={total_area:.2f}")
        print(f"ESTIMATED_REGION={region}")
        print(f"MASK_PATH={mask_path}")
        print(f"HEATMAP_PATH={heatmap_path}")

    return all_reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on BraTS H5 slices"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--h5_file", type=str, default=None,
                        help="Path to a single .h5 file for inference")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to BraTS H5 data directory")
    parser.add_argument("--volume_id", type=int, default=None,
                        help="Patient volume ID (e.g. 100)")
    parser.add_argument("--output_dir", type=str, default="./output/predictions",
                        help="Directory to save visualizations")
    parser.add_argument("--output_json", action="store_true",
                        help="Print structured KEY=VALUE lines for Java/external callers")
    args = parser.parse_args()

    reports = main(args)
