"""
Reusable data helpers for Face Mask Detection.
- iterate images
- integrity check
- inventory counts
- print summaries
"""

from __future__ import annotations
from pathlib import Path  # path handling (platform-safe, nicer than raw strings)
from typing import Iterable, Dict, Tuple  # type hints for clarity & IDE help
from PIL import Image  # image integrity check (Pillow is robust for open/verify)
import hashlib


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def iter_images(root: Path) -> Iterable[Path]:
    """
    Yield image file paths under 'root' (recursive) with allowed extensions.
    """
    root = Path(root) # in case user passes a string instead of Path
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def check_image_ok(p: Path) -> bool:
    """
    Return True if PIL can open/verify and fully decode the image; False otherwise.
    """
    try:
        with Image.open(p) as im:
            im.verify()     # quick header check

        with Image.open(p) as im:
            _ = im.size     # force a real decode

        return True
    except Exception:
        # Any problem (unsupported format, truncated, corrupt) → not OK
        return False
    

def inventory(root: Path) -> dict[str, int]:
    """
    Count images by parent folder name under root.
    Returns a dict like {"WithMask": 1234, "WithoutMask": 1311}.
    """
    counts: dict[str, int] = {}
    for p in iter_images(root):
        cls = p.parent.name
        counts[cls] = counts.get(cls, 0) + 1
    return counts


def print_summary(title: str, counts: dict[str, int]) -> None:
    """
    Print a summary of counts with a title and total.
    """
    total = sum(counts.values())
    print(f"\n{title} (total={total})")
    for k in sorted(counts):
        print(f"  {k:>12}: {counts[k]}")


def file_hash(p: Path, algo: str = "md5", block_size: int = 1 << 20) -> str:
    """
    Return hex digest of file content using the given algorithm (default: md5).
    Reads file in chunks so it's memory-efficient.
    """
    h = hashlib.new(algo)
    with open(p, "rb") as f:
        while True:
            chunk = f.read(block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def find_duplicates(root: Path) -> dict[str, list[Path]]:
    """
    Return dict: {hash: [paths...]} for any hash that appears 2+ times.
    """
    groups: dict[str, list[Path]] = {}
    for p in iter_images(root):
        h = file_hash(p)
        groups.setdefault(h, []).append(p)

    # keep only groups with 2+ files
    dupes = {h: paths for h, paths in groups.items() if len(paths) > 1}
    return dupes


def size_stats(root: Path) -> Dict[str, Tuple[int, int, int, int]]:
    """
    Compute per-class size extremes: {class: (min_w, min_h, max_w, max_h)}.
    """
    stats: Dict[str, Tuple[int, int, int, int]] = {}
    # We'll store as mutable lists internally, then cast to tuples at the end
    tmp: Dict[str, list[int]] = {}

    for p in iter_images(root):
        cls = p.parent.name
        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception:
            # If an image is corrupt and slipped through, skip it
            # (We already have check_image_ok(), so this should be rare.)
            continue

        if cls not in tmp:
            tmp[cls] = [w, h, w, h]  # min_w, min_h, max_w, max_h
        else:
            tmp[cls][0] = min(tmp[cls][0], w)
            tmp[cls][1] = min(tmp[cls][1], h)
            tmp[cls][2] = max(tmp[cls][2], w)
            tmp[cls][3] = max(tmp[cls][3], h)

    # freeze to tuples
    stats = {cls: (v[0], v[1], v[2], v[3]) for cls, v in tmp.items()}
    return stats