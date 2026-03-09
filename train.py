"""
train.py — Training Loop with AMP, Dice+CE Loss, Gradient Accumulation
========================================================================
Trains the Attention U-Net on BraTS 2020 pre-processed H5 slices.

Key features for RTX 4050 (6 GB VRAM):
    • Automatic Mixed Precision (AMP) — halves memory, ~1.5× speed-up
    • Gradient accumulation — effective batch 8 with actual batch 2
    • AdamW optimizer with Cosine Annealing LR schedule
    • Combined Dice + CrossEntropy loss for class-imbalanced segmentation
    • Logs per-class Dice (ET, TC, WT) — BraTS official metrics
    • TensorBoard logging + checkpoint saving

Usage:
    python train.py --data_dir "C:/Users/Loga Prasath/Downloads/archive/BraTS2020_training_data/content/data" --epochs 50
"""

import os
import time
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False

from dataset import get_dataloaders, NUM_CLASSES
from model import AttentionUNet


# ======================================================================
# Loss Functions
# ======================================================================

class DiceLoss(nn.Module):
    """
    Soft Dice Loss for multi-class segmentation.
    Computes per-class Dice and averages (excluding background optionally).

    Parameters
    ----------
    smooth : float
        Laplace smoothing to avoid division by zero.
    ignore_bg : bool
        If True, exclude class 0 (background) from the loss.
    """

    def __init__(self, smooth: float = 1.0, ignore_bg: bool = True):
        super().__init__()
        self.smooth = smooth
        self.ignore_bg = ignore_bg

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, C, H, W) — raw model output
        targets : (B, H, W)    — integer class labels
        """
        probs = F.softmax(logits, dim=1)                       # (B, C, H, W)
        targets_oh = F.one_hot(targets, NUM_CLASSES)            # (B, H, W, C)
        targets_oh = targets_oh.permute(0, 3, 1, 2).float()    # (B, C, H, W)

        start_ch = 1 if self.ignore_bg else 0
        dice_sum = 0.0
        n = 0
        for c in range(start_ch, NUM_CLASSES):
            p = probs[:, c]
            t = targets_oh[:, c]
            intersection = (p * t).sum()
            union = p.sum() + t.sum()
            dice_sum += (2.0 * intersection + self.smooth) / (union + self.smooth)
            n += 1
        return 1.0 - dice_sum / n


class DiceCELoss(nn.Module):
    """
    Combined Dice Loss + Cross Entropy Loss.
    Good for highly imbalanced medical segmentation like BraTS.

    total_loss = alpha * DiceLoss + (1 - alpha) * CE
    """

    def __init__(self, alpha: float = 0.5, ce_weights: list = None):
        super().__init__()
        self.alpha = alpha
        self.dice = DiceLoss(smooth=1.0, ignore_bg=True)
        # Optional class weights for CE (handle imbalance further)
        weight = torch.tensor(ce_weights, dtype=torch.float32) if ce_weights else None
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        d = self.dice(logits, targets)
        # Cast to float32: CrossEntropyLoss can overflow / produce NaN in float16
        c = self.ce(logits.float(), targets)
        return self.alpha * d + (1 - self.alpha) * c


# ======================================================================
# BraTS Official Evaluation Metrics
# ======================================================================

@torch.no_grad()
def compute_brats_dice(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    Compute official BraTS evaluation Dice scores:
        • ET  (Enhancing Tumor)            = class 3
        • TC  (Tumor Core)                 = classes 1 + 3
        • WT  (Whole Tumor)                = classes 1 + 2 + 3

    Parameters
    ----------
    pred   : (B, H, W)  int tensor — predicted class labels
    target : (B, H, W)  int tensor — ground-truth class labels

    Returns
    -------
    dict with keys 'ET', 'TC', 'WT', each a float Dice score.
    """
    smooth = 1e-5

    def dice(p, t):
        inter = (p & t).float().sum()
        return ((2 * inter + smooth) / (p.float().sum() + t.float().sum() + smooth)).item()

    # Enhancing Tumor: class 3 only
    et_p, et_t = (pred == 3), (target == 3)
    # Tumor Core: classes 1 and 3
    tc_p, tc_t = (pred == 1) | (pred == 3), (target == 1) | (target == 3)
    # Whole Tumor: classes 1, 2, and 3
    wt_p, wt_t = (pred >= 1), (target >= 1)

    return {"ET": dice(et_p, et_t), "TC": dice(tc_p, tc_t), "WT": dice(wt_p, wt_t)}


# ======================================================================
# Training & Validation Steps
# ======================================================================

def train_one_epoch(
    model, loader, criterion, optimizer, scaler, device, accum_steps,
    max_grad_norm: float = 1.0
):
    model.train()
    running_loss = 0.0
    skipped = 0
    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with autocast("cuda", dtype=torch.float16):
            logits = model(images)
            loss = criterion(logits, masks) / accum_steps

        # Skip NaN/Inf batches to prevent training collapse
        if not torch.isfinite(loss):
            skipped += 1
            optimizer.zero_grad()
            scaler.update()  # keep scaler state consistent
            continue

        scaler.scale(loss).backward()

        # Gradient accumulation: step every `accum_steps` mini-batches
        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            # Unscale before clipping so clip threshold is in real gradient units
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * accum_steps

    if skipped > 0:
        print(f"  [!] Skipped {skipped} NaN/Inf batches this epoch")
    return running_loss / max(len(loader) - skipped, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    dice_sums = {"ET": 0.0, "TC": 0.0, "WT": 0.0}
    n_batches = 0

    n_loss_batches = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with autocast("cuda", dtype=torch.float16):
            logits = model(images)
            loss = criterion(logits, masks)

        # Skip NaN/Inf loss values to keep running average meaningful
        if torch.isfinite(loss):
            running_loss += loss.item()
            n_loss_batches += 1
        preds = logits.argmax(dim=1)
        dice = compute_brats_dice(preds, masks)
        for k in dice_sums:
            dice_sums[k] += dice[k]
        n_batches += 1

    avg_loss = running_loss / max(n_loss_batches, 1)
    avg_dice = {k: v / max(n_batches, 1) for k, v in dice_sums.items()}
    return avg_loss, avg_dice


# ======================================================================
# Main Training Loop
# ======================================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---- Data ----
    train_loader, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tumor_only=args.tumor_only,
    )

    # ---- Model ----
    model = AttentionUNet(in_channels=4, out_channels=NUM_CLASSES).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Loss / optimizer / scheduler ----
    # Class weights: BG is very dominant, so we up-weight tumor classes
    ce_weights = [0.1, 1.0, 1.0, 1.5]   # BG, NCR/NET, ED, ET
    criterion = DiceCELoss(alpha=0.5, ce_weights=ce_weights).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    scaler = GradScaler("cuda")  # AMP gradient scaler

    # ---- Resume from checkpoint ----
    start_epoch = 1
    best_wt_dice = 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_wt_dice = ckpt.get("best_wt_dice", 0.0)
        print(f"Resumed from {args.resume} — starting at epoch {start_epoch}, best WT={best_wt_dice:.4f}")

    # ---- TensorBoard ----
    writer = None
    if HAS_TB:
        log_dir = os.path.join(args.save_dir, "runs",
                               datetime.now().strftime("%Y%m%d_%H%M%S"))
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logs → {log_dir}")

    # ---- Checkpoint dir ----
    ckpt_dir = os.path.join(args.save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- Training loop ----
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            accum_steps=args.accum_steps, max_grad_norm=args.max_grad_norm,
        )
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        # ---- Logging ----
        print(
            f"Epoch [{epoch:03d}/{args.epochs}]  "
            f"Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}  "
            f"ET: {val_dice['ET']:.4f}  TC: {val_dice['TC']:.4f}  "
            f"WT: {val_dice['WT']:.4f}  LR: {lr:.2e}  "
            f"Time: {elapsed:.1f}s"
        )

        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Dice/ET", val_dice["ET"], epoch)
            writer.add_scalar("Dice/TC", val_dice["TC"], epoch)
            writer.add_scalar("Dice/WT", val_dice["WT"], epoch)
            writer.add_scalar("LR", lr, epoch)

        # ---- Save best model (based on Whole Tumor Dice) ----
        if val_dice["WT"] > best_wt_dice:
            best_wt_dice = val_dice["WT"]
            best_path = os.path.join(ckpt_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_wt_dice": best_wt_dice,
                "val_dice": val_dice,
            }, best_path)
            print(f"  >> New best WT Dice: {best_wt_dice:.4f} — saved {best_path}")

        # Periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
            }, os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth"))

    if writer:
        writer.close()
    print(f"\nTraining complete.  Best WT Dice: {best_wt_dice:.4f}")


# ======================================================================
# CLI
# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Attention U-Net on BraTS 2020 H5 slices"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to BraTS H5 data directory")
    parser.add_argument("--save_dir", type=str, default="./output",
                        help="Directory for checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Actual batch size per step (keep <=2 for 6 GB)")
    parser.add_argument("--accum_steps", type=int, default=4,
                        help="Gradient accumulation steps (eff. batch = BS x accum)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Initial learning rate")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping")
    parser.add_argument("--wd", type=float, default=1e-4,
                        help="Weight decay (AdamW)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--tumor_only", action="store_true", default=True,
                        help="Train only on slices containing tumor")
    parser.add_argument("--no_tumor_only", dest="tumor_only",
                        action="store_false",
                        help="Train on ALL slices (including empty)")
    args = parser.parse_args()

    main(args)
