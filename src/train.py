"""
Train a MobileNetV2 baseline model for face-mask detection.

- Loads train/val datasets with torchvision ImageFolder
- Applies ImageNet-style preprocessing (resize, flip, normalize)
- Builds model head for 2 classes
- Trains and saves best weights to models/
"""

from __future__ import annotations

from pathlib import Path
import argparse
import random
import sys

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import datasets, transforms, models


# ---- config / paths ----
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TRAIN_DIR = DATA / "train"
VAL_DIR = DATA / "val"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["with_mask", "without_mask"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224

SEED = 42


# ---- utils ----
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---- data ----
def build_transforms(img_size: int = IMG_SIZE):
    """
    MobileNetV2-style preprocessing.
    Train: light augmentation (flip) + normalize.
    Val: deterministic transform + normalize.
    """
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tfms, val_tfms


def build_loaders(
        train_dir: Path,
        val_dir: Path,
        batch: int,
        workers: int,
        dev: torch.device,
):
    train_tfms, val_tfms = build_transforms()

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tfms)

    # On CUDA, pin_memory can speed up host-->GPU copies
    pin = (dev.type == "cuda")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch,
        shuffle=True,
        num_workers=workers,    # On windows: start with 0-2
        pin_memory=pin,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
    )

    return train_loader, val_loader


# ---- model ----
def build_model(num_classes: int = 2, freeze_backbone: bool = False) -> torch.nn.Module:
    """
    Load pretrained MobileNetV2 and replace the classifier head for num_classes.
    Optionally freeze the backbone (feature extractor).
    """
    # weights are name depends on torchvision version; this works for 0.13+
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    #in_features = last layer input size (1280 for MobileNetV2)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False
    
    return model


# ---- argparse ----
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Train baseline face-mask classifier")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num-workers", type=int, default=0)  # safe default on Windows
    p.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    p.add_argument("--val-dir", type=Path, default=VAL_DIR)
    p.add_argument("--out", type=Path, default=MODELS_DIR / "best_mobilenetv2.pt")
    p.add_argument("--freeze-backbone", action="store_true")
    return p


# ---- main ----
def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    assert args.train_dir.exists(), f"missing: {args.train_dir}"
    assert args.val_dir.exists(), f"missing: {args.val_dir}"

    set_seed(SEED)
    dev = device()
    print(f"device: {dev.type} | cuda: {torch.cuda.is_available()}")

    # step 2.1: data transforms + dataloaders
    train_loader, val_loader = build_loaders(
        args.train_dir, args.val_dir,
        batch=args.batch_size,
        workers=args.num_workers,
        dev=dev,
    )
    print(f"batches | train: {len(train_loader)} val: {len(val_loader)}")

    # step 2.2: model (mobilenetv2), optionally freeze backbone
    model = build_model(num_classes=len(CLASSES), freeze_backbone=args.freeze_backbone)
    print(model.classifier[-1])  # print new head

    # step 2.3: loss/optim/scheduler
    # step 2.4: training loop (track acc/F1, save best)
    # step 2.5: final summary

    return 0


if __name__ == "__main__":
    sys.exit(main())
