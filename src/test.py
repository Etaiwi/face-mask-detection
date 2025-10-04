from __future__ import annotations
import argparse
from pathlib import Path
import sys


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Evaluate/Infer face-mask classifier")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--test-dir", type=Path, help="Labeled test/val directory")
    src.add_argument("--images", type=Path, help="Unlabeled images folder for inference")
    p.add_argument("--weights", type=Path, required=True, help="Path to model weights (.pt/.ckpt)")
    p.add_argument("--classes", nargs="+", required=True, help="Class names in training order")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    
    # debug echo, remove at final version
    for k, v in vars(args).items():
        print(f"{k} = {v}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())