from __future__ import annotations
from pathlib import Path

# ---- Classes ----
CLASSES = ("with_mask", "without_mask")

# ---- Image / preprocessing ----
IMG_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---- Reproducibility ----
SEED = 42

# ---- Paths ----
ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
DEFAULT_WEIGHTS_PATH = MODELS_DIR / "best_mobilenetv2.pt"

# ---- Model weights download (GitHub Release) ----
WEIGHTS_URL = "https://github.com/Etaiwi/face-mask-detection/releases/download/v1.0.0/best_mobilenetv2.pt"
WEIGHTS_SHA256 = None  # optional: add later for integrity
