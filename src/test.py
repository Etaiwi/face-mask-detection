from __future__ import annotations
import argparse
from pathlib import Path
import sys
import random
import torch

from models.factory import build_model


# ---- config / paths ----
ROOT = Path(__file__).resolve().parents[1]  # repo root
DATA = ROOT / "data"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["with_mask", "without_mask"]
IMG_SIZE = 224
SEED = 42


# ---- runtime helpers ----
def set_seed(seed: int = SEED) -> None:
    """ Set RNG seeds for reproducibility"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    """ Get available device"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---- argparse ----
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Evaluate/Infer face-mask classifier")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--test-dir", type=Path, help="Labeled test/val directory")
    src.add_argument("--images", type=Path, help="Unlabeled images folder for inference")
    p.add_argument("--weights", type=Path, default=MODELS_DIR / "best_mobilenetv2.pt", help="Path to model weights (.pt/.ckpt)")
    p.add_argument("--classes", nargs="+", default=CLASSES ,help="Class names in training order")
    p.add_argument("--img-size", type=int, default=IMG_SIZE)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)

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

    # instantiate model
    num_classes = len(args.classes)
    model = build_model(num_classes=num_classes).to(dev)

    # Optional, quick confidence print (keeps output minimal)
    head = getattr(model, "classifier", None)
    if head is not None:
        print("model head:", head[-1])
    print(f"num classes: {num_classes}")

    return 0


if __name__ == "__main__":
    sys.exit(main())