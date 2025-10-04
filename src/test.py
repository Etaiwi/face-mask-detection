from __future__ import annotations
import argparse
from pathlib import Path
import sys

# ---- config / paths ----
ROOT =Path(__file__).resolve().parents[1]  # repo root
DATA = ROOT / "data"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["with_mask", "without_mask"]
IMG_SIZE = 224


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
    args = build_argparser().parse_args(argv)
    
    # debug echo, remove at final version
    for k, v in vars(args).items():
        print(f"{k} = {v}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())