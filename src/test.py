from __future__ import annotations
import argparse
from pathlib import Path
import sys
import random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from PIL import Image

from models.factory import build_model
from transforms import build_eval_transform
from config import (
    CLASSES,
    IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SEED,
    DEFAULT_WEIGHTS_PATH,
)
from utils import (
    set_seed,
    device,
    load_weights,
    batch_preds,
    update_running_counts,
    f1_macro_from_counts,
)

import warnings; warnings.filterwarnings("ignore", message="Palette images with Transparency")

# ---- config / paths ----
ROOT = Path(__file__).resolve().parents[1]  # repo root
DATA = ROOT / "data"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---- data ----
def build_loader(
        data_dir: Path,
        batch: int,
        workers: int,
        dev: torch.device,
) -> DataLoader:
    """ Build DataLoader for eval/inference. """
    tfms = build_eval_transform(img_size=IMG_SIZE)
    ds = datasets.ImageFolder(data_dir, transform=tfms)
    pin = (dev.type == "cuda")
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=pin)

class ImageFolderFlat(torch.utils.data.Dataset):
    """
    Walks all image files under a root directory (recursively) and returns:
        (tensor_image, path_str)
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root: Path, transform):
        self.root = Path(root)
        self.transform = transform
        self.files = [p for p in self.root.rglob("*") if p.suffix.lower() in self.exts]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path)
        img = self.transform(img)
        return img, str(path)

def build_infer_loader(images_dir: Path, batch: int, workers: int, dev: torch.device) -> DataLoader:
    tfms = build_eval_transform(img_size=IMG_SIZE)
    ds = ImageFolderFlat(images_dir, transform=tfms)
    pin = (dev.type == "cuda")
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=pin)

# ---- loops ----
@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, dev: torch.device) -> dict:
    model.eval()
    correct, total = 0, 0
    y_true, y_pred = [], []
    for xb, yb in loader:
        xb = xb.to(dev, non_blocking=True)
        yb = yb.to(dev, non_blocking=True)

        logits = model(xb)
        preds = batch_preds(logits)

        correct += (preds == yb).sum().item()
        total   += yb.numel()
        y_true.extend(yb.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    acc = correct / max(total, 1)
    f1  = f1_macro_from_counts(y_true, y_pred)  # import from utils or sklearn
    return {"n": total, "acc": acc, "f1": f1}

@torch.no_grad()
def infer_images(model: torch.nn.Module, loader: DataLoader, dev: torch.device, class_names: list[str]) -> None:
    model.eval()
    for xb, paths in loader:
        xb = xb.to(dev, non_blocking=True)
        logits = model(xb)                  # [B, C]
        preds = logits.argmax(dim=1).tolist()
        for p, path in zip(preds, paths):
            print(f"{path} -> {class_names[p]}")

# ---- argparse ----
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Evaluate/Infer face-mask classifier")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--test-dir", type=Path, help="Labeled test/val directory")
    src.add_argument("--images", type=Path, help="Unlabeled images folder for inference")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS_PATH, help="Path to model weights (.pt/.ckpt)")
    p.add_argument("--classes", nargs="+", default=CLASSES ,help="Class names in training order")
    p.add_argument("--img-size", type=int, default=IMG_SIZE)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    return p

# ---- main ----
def main(argv: list[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)

    # path checks
    if args.test_dir is not None and not args.test_dir.exists():
        parser.error(f"--test-dir path does not exist: {args.test_dir}")
    if args.images is not None and not args.images.exists():
        parser.error(f"--images path does not exist: {args.images}")
    if not args.weights.exists():
        parser.error(f"--weights path does not exist: {args.weights}")

    # seed + device
    set_seed(SEED)
    dev = device()
    print(f"device: {dev.type}")

    # model + weights
    num_classes = len(args.classes)
    model = build_model(num_classes=num_classes).to(dev)
    load_weights(model, args.weights, map_location=dev)
    print(f"loaded weights from: {args.weights}")

    # head sanity
    head = getattr(model, "classifier", None)
    if head is not None:
        print("model head:", head[-1])
    print(f"num classes: {num_classes}")

    # ----- labeled evaluation path -----
    if args.test_dir is not None:
        loader = build_loader(args.test_dir, args.batch_size, args.num_workers, dev)
        ds = loader.dataset  # ImageFolder

        # class-order check (dataset vs CLI)
        ds_classes = list(ds.classes)
        if ds_classes != list(args.classes):
            print("[warn] Dataset class order != --classes order", file=sys.stderr)
        print(f"       dataset:   {ds_classes}", file=sys.stderr)
        print(f"       --classes: {list(args.classes)}", file=sys.stderr)

        # optional one-batch dry pass
        try:
            xb, yb = next(iter(loader))
            print(f"test loader: {len(ds)} images | batch={args.batch_size} | xb={tuple(xb.shape)} yb={tuple(yb.shape)}")
        except StopIteration:
            print("[warn] test loader is empty (no images found).", file=sys.stderr)

        # run evaluation
        metrics = evaluate(model, loader, dev)
        print(f"eval: n={metrics['n']} | acc={metrics['acc']:.3f} | f1_macro={metrics['f1']:.3f}")

    # ----- unlabeled inference path -----
    if args.images is not None:
        infer_loader = build_infer_loader(args.images, args.batch_size, args.num_workers, dev)
        n = len(infer_loader.dataset)
        if n == 0:
            print(f"[warn] no images found under: {args.images}", file=sys.stderr)
        else:
            print(f"infer: {n} images found under {args.images}")
            infer_images(model, infer_loader, dev, list(args.classes))

    return 0

if __name__ == "__main__":
    sys.exit(main())