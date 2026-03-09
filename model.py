"""
model.py — Attention U-Net for Brain Tumor Segmentation
========================================================
A 2D Attention U-Net that accepts 4-channel MRI input (T1, T1ce, T2, FLAIR)
and outputs a multi-class segmentation mask with 4 classes:
    0 — Background
    1 — Necrotic / Non-Enhancing Tumor Core (NCR/NET)
    2 — Peritumoral Edema (ED)
    3 — GD-Enhancing Tumor (ET)

Architecture highlights:
    • Encoder: 4 downsampling blocks with double-conv + BatchNorm + ReLU
    • Bottleneck: Double-conv at the lowest resolution
    • Decoder: 4 upsampling blocks with **Attention Gates** on skip connections
    • Final: 1×1 convolution → NUM_CLASSES channels

Reference:
    Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas",
    MIDL 2018.  arXiv:1804.03999

Hardware target: RTX 4050 (6 GB).  The model is ~8.7 M params (<35 MB fp32)
so it comfortably fits in VRAM alongside batch_size=2 at 240×240.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 4  # must match dataset.py


# ======================================================================
# Building blocks
# ======================================================================

class DoubleConv(nn.Module):
    """(Conv2d → BN → ReLU) × 2"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    """
    Additive Attention Gate.

    Takes the gating signal *g* (from the decoder) and the skip
    connection *x* (from the encoder), computes an attention coefficient
    map α ∈ [0, 1], and returns x * α.

    Parameters
    ----------
    F_g : int – channels in the gating signal
    F_l : int – channels in the skip connection (encoder feature)
    F_int : int – intermediate channel count (typically F_l // 2)
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        # 1×1 convs to project g and x to same channel dim
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        # ψ produces scalar attention map
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        g : gating signal  (B, F_g, H, W)  — from decoder (coarser)
        x : skip connection (B, F_l, H, W)  — from encoder (finer)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # Align spatial dims (g may be slightly smaller after up-conv)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear",
                               align_corners=True)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)          # (B, 1, H, W) attention coefficients
        return x * psi               # element-wise re-weighting


class DownBlock(nn.Module):
    """MaxPool2d → DoubleConv"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class UpBlock(nn.Module):
    """Transpose-Conv upsample → Attention Gate → Concat skip → DoubleConv"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # Upsample the decoder feature map
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        # Attention gate: g has out_ch channels (after up), x has out_ch
        self.attn = AttentionGate(F_g=out_ch, F_l=out_ch, F_int=out_ch // 2)
        # After concat: out_ch (up) + out_ch (skip) = 2*out_ch
        self.conv = DoubleConv(out_ch * 2, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if sizes don't match exactly (odd input dims)
        diff_h = skip.shape[2] - x.shape[2]
        diff_w = skip.shape[3] - x.shape[3]
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                       diff_h // 2, diff_h - diff_h // 2])
        skip = self.attn(g=x, x=skip)      # attention-gated skip
        x = torch.cat([x, skip], dim=1)     # (B, 2*out_ch, H, W)
        return self.conv(x)


# ======================================================================
# Full Attention U-Net
# ======================================================================

class AttentionUNet(nn.Module):
    """
    2D Attention U-Net for multi-class brain tumor segmentation.

    Channel progression (default):
        Encoder: 4→64→128→256→512
        Bottleneck: 512→1024
        Decoder: 1024→512→256→128→64
        Head: 64→NUM_CLASSES

    Parameters
    ----------
    in_channels : int
        Number of input channels (4 for BraTS: T1, T1ce, T2, FLAIR).
    out_channels : int
        Number of output classes (4: BG, NCR/NET, ED, ET).
    features : list[int]
        Feature map sizes at each encoder level.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = NUM_CLASSES,
        features: list = None,
    ):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        # ---- Encoder ----
        self.enc1 = DoubleConv(in_channels, features[0])
        self.enc2 = DownBlock(features[0], features[1])
        self.enc3 = DownBlock(features[1], features[2])
        self.enc4 = DownBlock(features[2], features[3])

        # ---- Bottleneck ----
        self.bottleneck = DownBlock(features[3], features[3] * 2)

        # ---- Decoder (with attention) ----
        self.dec4 = UpBlock(features[3] * 2, features[3])
        self.dec3 = UpBlock(features[3], features[2])
        self.dec2 = UpBlock(features[2], features[1])
        self.dec1 = UpBlock(features[1], features[0])

        # ---- Segmentation head ----
        self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)

        # Weight init
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 4, H, W) — 4-channel MRI input

        Returns
        -------
        logits : (B, NUM_CLASSES, H, W) — raw class logits
        """
        # Encoder
        s1 = self.enc1(x)    # (B, 64,  H,   W)
        s2 = self.enc2(s1)   # (B, 128, H/2, W/2)
        s3 = self.enc3(s2)   # (B, 256, H/4, W/4)
        s4 = self.enc4(s3)   # (B, 512, H/8, W/8)

        # Bottleneck
        b = self.bottleneck(s4)  # (B, 1024, H/16, W/16)

        # Decoder + attention skip connections
        d4 = self.dec4(b, s4)    # (B, 512, H/8,  W/8)
        d3 = self.dec3(d4, s3)   # (B, 256, H/4,  W/4)
        d2 = self.dec2(d3, s2)   # (B, 128, H/2,  W/2)
        d1 = self.dec1(d2, s1)   # (B, 64,  H,    W)

        return self.head(d1)     # (B, NUM_CLASSES, H, W)


# ======================================================================
# Utility: param count & quick test
# ======================================================================
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = AttentionUNet(in_channels=4, out_channels=NUM_CLASSES)
    print(f"Attention U-Net — {count_parameters(model):,} trainable parameters")

    # Dummy forward pass (simulate one BraTS slice, 4 modalities)
    dummy = torch.randn(2, 4, 240, 240)
    out = model(dummy)
    print(f"Input  shape: {dummy.shape}")
    print(f"Output shape: {out.shape}")    # expect (2, 4, 240, 240)
    assert out.shape == (2, NUM_CLASSES, 240, 240), "Shape mismatch!"
    print("✓ Forward pass OK")
