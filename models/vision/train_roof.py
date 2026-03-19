"""
Roof material classification training script.

Trains the CNN-ViT backbone + roof classification head on labeled NAIP patches.

Usage:
    Local:   python models/vision/train_roof.py
    Colab:   Set COLAB_MODE = True; upload this file to Colab and run.

Colab setup:
    from google.colab import drive
    drive.mount('/content/drive')
    # Then set COLAB_MODE = True below
"""

from __future__ import annotations

# ── Colab mode flag ──────────────────────────────────────────────────────────
COLAB_MODE = False  # ← Set to True when running on Google Colab
# ─────────────────────────────────────────────────────────────────────────────

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as T
from loguru import logger

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ingestion.config_loader import get_paths, load_config
from models.vision.backbone import build_backbone


# ── Dataset ──────────────────────────────────────────────────────────────────

class RoofPatchDataset(Dataset):
    """
    Dataset of 256×256 NAIP patches labeled by roof material class.

    Expects the following directory structure under processed_structure:
        roof_patches/
            metal_standing_seam/   (*.npy files, shape 4×256×256)
            metal_corrugated/
            asphalt_shingles/
            wood_shingles_shake/
            ...

    Labels are derived from directory name → class index mapping.
    """

    CLASS_NAMES = [
        "metal_standing_seam",
        "metal_corrugated",
        "concrete_clay_tile",
        "asphalt_shingles",
        "wood_shingles_shake",
        "built_up_tar_gravel",
        "membrane_flat",
        "unknown_occluded",
    ]

    def __init__(
        self,
        data_dir: Path,
        augment: bool = True,
        mean: list[float] = (0.485, 0.456, 0.406, 0.4),
        std: list[float] = (0.229, 0.224, 0.225, 0.15),
    ):
        self.data_dir = data_dir
        self.augment = augment
        self.mean = torch.tensor(mean).view(4, 1, 1)
        self.std = torch.tensor(std).view(4, 1, 1)
        self.samples: list[tuple[Path, int]] = []

        for class_idx, class_name in enumerate(self.CLASS_NAMES):
            class_dir = data_dir / "roof_patches" / class_name
            if class_dir.exists():
                for f in class_dir.glob("*.npy"):
                    self.samples.append((f, class_idx))

        if not self.samples:
            logger.warning(
                f"No training samples found in {data_dir / 'roof_patches'}. "
                "Generate patches with scripts/generate_roof_patches.py or "
                "use the synthetic fallback below."
            )
            self._add_synthetic_samples()

        logger.info(
            f"RoofPatchDataset: {len(self.samples)} samples across "
            f"{len({s[1] for s in self.samples})} classes"
        )

    def _add_synthetic_samples(self) -> None:
        """Add synthetic samples for development/testing when real data is absent."""
        for i in range(200):
            class_idx = i % len(self.CLASS_NAMES)
            self.samples.append((Path(f"synthetic_{i}"), class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]

        if not path.exists():
            # Synthetic fallback: random patch
            patch = torch.randn(4, 256, 256)
        else:
            arr = np.load(path).astype("float32") / 255.0
            patch = torch.from_numpy(arr)

        # Normalize
        patch = (patch - self.mean) / self.std

        # Augmentation
        if self.augment and self.training:
            patch = self._augment(patch)

        return patch, label

    def _augment(self, patch: torch.Tensor) -> torch.Tensor:
        """Apply random flips and 90° rotations (rotation-invariant roof patterns)."""
        if torch.rand(1) > 0.5:
            patch = torch.flip(patch, dims=[2])  # horizontal flip
        if torch.rand(1) > 0.5:
            patch = torch.flip(patch, dims=[1])  # vertical flip
        k = int(torch.randint(0, 4, (1,)))
        patch = torch.rot90(patch, k=k, dims=[1, 2])
        return patch


# ── Training loop ─────────────────────────────────────────────────────────────

def train(colab_mode: bool = False) -> None:
    paths = get_paths(colab_mode)
    cfg = load_config("model_config.yaml")
    train_cfg = cfg["training"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device} | COLAB_MODE={colab_mode}")

    # Batch size: smaller on Colab to respect GPU memory limits
    batch_size = train_cfg["batch_size_colab"] if colab_mode else train_cfg["batch_size"]

    # Dataset
    data_dir = paths["processed_structure"]
    dataset = RoofPatchDataset(data_dir, augment=True)

    n_val = max(1, int(0.15 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    # Disable augmentation for validation split
    val_ds.dataset.training = False

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # Model
    model = build_backbone(cfg).to(device)
    logger.info(f"Model parameters: {model.num_parameters:,}")

    # Class-weighted loss (wood shake is rare but critical)
    class_weights = torch.tensor(
        cfg["roof_classifier"].get("class_weights", [1.0] * 8)
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer with cosine schedule
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"], eta_min=1e-6
    )

    # Checkpoint directory
    ckpt_dir = paths["processed"] / "checkpoints" / "roof"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    patience = train_cfg["early_stopping_patience"]
    no_improve = 0

    for epoch in range(1, train_cfg["epochs"] + 1):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for patches, labels in train_loader:
            patches, labels = patches.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(patches)["roof"]
            loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip_norm"])
            optimizer.step()

            train_loss += loss.item() * len(labels)
            train_correct += (out.argmax(1) == labels).sum().item()
            train_total += len(labels)

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for patches, labels in val_loader:
                patches, labels = patches.to(device), labels.to(device)
                out = model(patches)["roof"]
                val_loss += criterion(out, labels).item() * len(labels)
                val_correct += (out.argmax(1) == labels).sum().item()
                val_total += len(labels)

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        scheduler.step()

        logger.info(
            f"Epoch {epoch:3d}/{train_cfg['epochs']} | "
            f"train loss={train_loss / train_total:.4f} acc={train_acc:.3f} | "
            f"val loss={val_loss / val_total:.4f} acc={val_acc:.3f}"
        )

        # Checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "val_acc": val_acc, "config": cfg},
                ckpt_dir / "best_roof_model.pt",
            )
            logger.info(f"  → Saved best model (val_acc={val_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    logger.info(f"Training complete. Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    train(colab_mode=COLAB_MODE)
