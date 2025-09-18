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
import numpy as np
import time

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

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


# ---- optim ----
def build_criterion() -> nn.Module:
    """ Mulit-class classification loss. """
    return nn.CrossEntropyLoss()


def build_optimizer(model: nn.Module, lr: float) -> Adam:
    """ Adam over trainable parameters only. """
    params = (p for p in model.parameters() if p.requires_grad)
    return Adam(params, lr=lr)


def build_scheduler(optimizer: Adam) -> ReduceLROnPlateau:
    """
    Reduce LR when a metric has stopped improving.
    Call scheduler.step(val_f1) after each epoch.
    """
    return ReduceLROnPlateau(
        optimizer,
        mode="max",     # maximizing F1
        factor=0.5,     # halves LR on plateau
        patience=2,     # wait 2 epochs without improvement
        min_lr=1e-6,
    )


def current_lr(optimizer: Adam) -> float:
    """ Get current learning rate from optimizer. """
    return optimizer.param_groups[0]["lr"]


# ---- metrics ----
def batch_preds(logits: torch.Tensor) -> torch.Tensor:
    """ logits: [B, 2] -> class indices [B]."""
    return logits.argmax(dim=1)


def update_running_counts(preds, targets, correct, total):
    """ Update running counts for accuracy calculation. """
    correct += (preds == targets).sum().item()
    total += targets.numel()
    return correct, total


def f1_macro_from_counts(y_true: list[int], y_pred: list[int]) -> float:
    """ Compute macro F1 from lists of true and predicted class indices."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    f1s = []
    for c in [0, 1]:
        tp = ((y_true == c) & (y_pred == c)).sum()
        fp = ((y_true != c) & y_pred == c).sum()
        fn = ((y_true == c) & (y_pred != c)).sum()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2*prec*rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return float(sum(f1s) / len(f1s))


# ---- loops ----
def train_one_epoch(
        model: torch.nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: Adam,
        device: torch.device,
) -> dict:
    model.train()   # enable dropout/batchnorm updates
    loss_sum = 0.0
    correct, total = 0, 0
    y_true, y_pred = [], []

    for xb, yb in loader:
        # move batch to device (GPU/CPU)
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        # clear previous gradients
        optimizer.zero_grad(set_to_none=True)

        # forward pass: logits [B, 2]
        logits = model(xb)

        # compute scalar loss for the batch
        loss = criterion(logits, yb)

        # backprop
        loss.backward()

        # update trainable parameters
        optimizer.step()

        # accumulate sum of per-sample losses
        loss_sum += loss.item() * yb.size(0)

        # predictions & running accuracy
        preds = batch_preds(logits)
        correct, total = update_running_counts(preds, yb, correct, total)

        # collect for F1
        y_true.extend(yb.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    # epoch-level averages/metrics
    avg_loss = loss_sum / max(total, 1)
    acc = correct / max(total, 1)
    f1 = f1_macro_from_counts(y_true, y_pred)
    return {"loss": avg_loss, "acc": acc, "f1": f1}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> dict:
    model.eval()  # disable dropout/batchnorm updates
    loss_sum = 0.0
    correct, total = 0, 0
    y_true, y_pred = [], []

    for xb, yb in loader:
        # move batch to device (GPU/CPU)
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        # forward pass: logits [B, 2]
        logits = model(xb)

        # compute scalar loss for the batch
        loss = criterion(logits, yb)

        # accumulate sum of per-sample losses
        loss_sum += loss.item() * yb.size(0)

        # predictions & running accuracy
        preds = batch_preds(logits)
        correct, total = update_running_counts(preds, yb, correct, total)

        # collect for F1
        y_true.extend(yb.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    # epoch-level averages/metrics
    avg_loss = loss_sum / max(total, 1)
    acc = correct / max(total, 1)
    f1 = f1_macro_from_counts(y_true, y_pred)
    return {"loss": avg_loss, "acc": acc, "f1": f1}


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

    mode = model.to(dev)
    print("model device:", next(model.parameters()).device)

    # step 2.3: loss/optim/scheduler
    criterion = build_criterion()
    optimizer = build_optimizer(model, lr=args.lr)
    scheduler = build_scheduler(optimizer)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"params | trainable: {n_trainable:,} / total: {n_total:,}")

    # step 2.4: training loop (track acc/F1, save best)
    best_f1 = -1.0
    for epoch in range(args.epochs):
        start = time.time() # start timer
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, dev)
        val_metrics = evaluate(model, val_loader, criterion, dev)
        elapsed = time.time() - start # end timer

    # scheduler uses val F1
    scheduler.step(val_metrics["f1"])
    print(f"lr now: {current_lr(optimizer):.2e}")

    # save best
    if val_metrics["f1"] > best_f1:
        best_f1 = val_metrics["f1"]
        torch.save(model.state_dict(), args.out)

    print(
        f"epcoch {epoch+1:03d}/{args.epochs} | "
        f"train loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.3f} f1 {train_metrics['f1']:.3f} | "
        f"val_loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.3f} f1 {val_metrics['f1']:.3f} | "
        f"time {elapsed:.1f}s"
    )

    # step 2.5: final summary

    return 0


if __name__ == "__main__":
    sys.exit(main())
