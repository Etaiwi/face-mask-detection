from __future__ import annotations

from pathlib import Path
import random
from typing import Tuple, List

import numpy as np
import torch

from config import SEED

def set_seed(seed: int = SEED) -> None:
    """Set RNG seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    """Return the available device (CUDA if present, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_weights(
    model: torch.nn.Module,
    pt_path: Path,
    map_location: str | torch.device = "cpu",
) -> None:
    """
    Load a state_dict from a .pt/.ckpt file.

    - Supports raw state_dict or wrapped {"state_dict": ...}
    - Strips "module." prefixes (e.g. if trained with DataParallel)
    - Uses strict=False and prints any missing/unexpected keys.
    """
    state = torch.load(pt_path, map_location=map_location)

    # Unwrap checkpoints that store {"state_dict": ...}
    if isinstance(state, dict):
        inner = state.get("state_dict", state)
    else:
        inner = state

    # Strip "module." prefixes if present
    cleaned = {k.replace("module.", "", 1): v for k, v in inner.items()}
    result = model.load_state_dict(cleaned, strict=False)

    missing = getattr(result, "missing_keys", [])
    unexpected = getattr(result, "unexpected_keys", [])
    if missing:
        print(f"[warn] missing keys: {len(missing)} (first 5): {missing[:5]}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)} (first 5): {unexpected[:5]}")


def batch_preds(logits: torch.Tensor) -> torch.Tensor:
    """Convert model logits [B, C] to predicted class indices [B]."""
    return logits.argmax(dim=1)


def update_running_counts(
    preds: torch.Tensor,
    targets: torch.Tensor,
    correct: int,
    total: int,
) -> Tuple[int, int]:
    """Update running correct/total counts for accuracy calculation."""
    correct += (preds == targets).sum().item()
    total += targets.numel()
    return correct, total


def f1_macro_from_counts(y_true: List[int], y_pred: List[int]) -> float:
    """
    Compute macro F1 from lists of true and predicted class indices.

    Uses all unique labels in y_true (so it generalizes beyond 2 classes).
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    classes = np.unique(y_true_arr)
    if classes.size == 0:
        return 0.0

    f1s: list[float] = []
    for c in classes:
        tp = ((y_true_arr == c) & (y_pred_arr == c)).sum()
        fp = ((y_true_arr != c) & (y_pred_arr == c)).sum()
        fn = ((y_true_arr == c) & (y_pred_arr != c)).sum()

        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)

    return float(sum(f1s) / len(f1s))
