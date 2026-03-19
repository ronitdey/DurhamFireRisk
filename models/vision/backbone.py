"""
Hybrid CNN-ViT backbone for structural feature extraction.

Adapted from the DISTRACT architecture: a CNN stem captures local texture
(roof material, vegetation type) while a Vision Transformer body captures
spatial context (vegetation-to-structure proximity, neighbor relationships).

This architecture is used for:
    1. Roof material classification (8 classes)
    2. Vegetation type segmentation (40 FBFM40 classes)

Colab training note:
    Set COLAB_MODE = True at the top of each training script (train_roof.py,
    train_veg.py) to load data from Google Drive paths.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MBConvBlock(nn.Module):
    """
    EfficientNet-style Mobile Inverted Bottleneck Conv block.
    Expand → Depthwise → Squeeze → Project.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: int = 4,
        stride: int = 1,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        mid = in_channels * expand_ratio
        self.use_skip = stride == 1 and in_channels == out_channels

        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(mid, mid, 3, stride=stride, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(),
        )
        self.se = SqueezeExcitation(mid, reduction=4)
        self.project = nn.Sequential(
            nn.Conv2d(mid, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.project(self.se(self.depthwise(self.expand(x))))
        if self.use_skip:
            out = self.drop_path(out) + x
        return out


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.SiLU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.se(x).view(x.shape[0], x.shape[1], 1, 1)
        return x * scale


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        return x * (random_tensor < keep).float() / keep


class EfficientNetStem(nn.Module):
    """
    CNN stem: 3 MBConv stages that reduce 256×256 → 16×16 feature map.
    Captures local texture: roof material grain, vegetation color, surface type.
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            # Initial projection: (B, 4, 256, 256) → (B, 32, 128, 128)
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            # Stage 1: (B, 32, 128, 128) → (B, 64, 64, 64)
            MBConvBlock(32, 64, stride=2),
            MBConvBlock(64, 64),
            # Stage 2: (B, 64, 64, 64) → (B, 128, 32, 32)
            MBConvBlock(64, 128, stride=2),
            MBConvBlock(128, 128),
            # Stage 3: (B, 128, 32, 32) → (B, out, 16, 16)
            MBConvBlock(128, out_channels, stride=2),
            MBConvBlock(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class VisionTransformerEncoder(nn.Module):
    """
    ViT encoder body operating on the 16×16 feature grid from the CNN stem.
    Each 1×1 spatial location becomes a token (256 tokens total).
    Captures spatial relationships: distance from structure to vegetation,
    neighboring buildings, terrain context.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Learnable position embeddings for 16×16 grid
        self.pos_embed = nn.Parameter(torch.randn(1, 16 * 16, embed_dim) * 0.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) from CNN stem
        B, C, H, W = x.shape
        # Flatten spatial → token sequence
        tokens = rearrange(x, "b c h w -> b (h w) c")
        tokens = tokens + self.pos_embed[:, : H * W, :]
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        # Reshape back to (B, C, H, W)
        return rearrange(tokens, "b (h w) c -> b c h w", h=H, w=W)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        drop_path_rate: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ClimateRiskBackbone(nn.Module):
    """
    Full hybrid CNN-ViT backbone for climate risk feature extraction.

    Input:
        4-band NAIP patch (B, 4, 256, 256) @ 0.6m resolution
        → covers ~150m × 150m centered on each building footprint.

    Output:
        dict with task-specific predictions:
            "roof": (B, num_classes_roof) logits
            "vegetation": (B, num_classes_veg, 16, 16) segmentation logits
    """

    def __init__(
        self,
        in_channels: int = 4,
        embed_dim: int = 256,
        vit_depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        num_classes_roof: int = 8,
        num_classes_veg: int = 40,
    ):
        super().__init__()
        self.cnn_stem = EfficientNetStem(in_channels=in_channels, out_channels=embed_dim)
        self.vit_body = VisionTransformerEncoder(
            embed_dim=embed_dim,
            depth=vit_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.roof_head = RoofClassificationHead(embed_dim, num_classes_roof)
        self.veg_head = VegetationSegmentationHead(embed_dim, num_classes_veg)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.cnn_stem(x)        # (B, 256, 16, 16)
        features = self.vit_body(features)  # (B, 256, 16, 16)
        return {
            "roof": self.roof_head(features),        # (B, 8)
            "vegetation": self.veg_head(features),   # (B, 40, 16, 16)
        }

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the raw feature map (B, 256, 16, 16) for downstream tasks."""
        return self.vit_body(self.cnn_stem(x))

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RoofClassificationHead(nn.Module):
    """
    Classification head predicting roof material category.

    Classes (8):
        0: Metal standing seam (fire resistant)
        1: Metal corrugated    (fire resistant)
        2: Concrete/clay tile  (fire resistant)
        3: Asphalt shingles    (moderate risk)
        4: Wood shingles/shake (high risk)
        5: Built-up/tar+gravel (moderate risk)
        6: Membrane/flat       (moderate risk)
        7: Unknown/occluded
    """

    def __init__(self, embed_dim: int = 256, num_classes: int = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(x))


class VegetationSegmentationHead(nn.Module):
    """
    Dense segmentation head predicting Scott-Burgan fuel model per pixel.
    Output is at 16×16 resolution (each pixel = ~9m at 0.6m input resolution).
    """

    def __init__(self, embed_dim: int = 256, num_classes: int = 40):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


def build_backbone(model_config: dict) -> ClimateRiskBackbone:
    """Instantiate backbone from model_config.yaml dict."""
    cfg = model_config.get("backbone", {})
    rc = model_config.get("roof_classifier", {})
    vs = model_config.get("veg_segmenter", {})
    return ClimateRiskBackbone(
        embed_dim=cfg.get("vit_embed_dim", 256),
        vit_depth=cfg.get("vit_depth", 6),
        num_heads=cfg.get("vit_num_heads", 8),
        mlp_ratio=cfg.get("vit_mlp_ratio", 4.0),
        dropout=cfg.get("dropout", 0.1),
        num_classes_roof=rc.get("num_classes", 8),
        num_classes_veg=vs.get("num_classes", 40),
    )
