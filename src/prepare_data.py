"""
Prepare dataset for Face Mask Detection.

Pipeline:
1) Scan data/raw/ → report class counts, corrupt images, duplicates, size stats.
2) Build data/working/ with only valid, deduplicated images.
3) Split data/working/ into train/val/test (80/10/10) with stable RNG seed.
4) Verify final counts.

Usage:
    python src/prepare_data.py
"""

from pathlib import Path
import shutil, random, sys
from data_utils import (
    iter_images, check_image_ok, file_hash,
    inventory, print_summary, find_duplicates, size_stats
)

RAW_DIR = Path("data/raw")
WORK_DIR = Path("data/working")
OUT_DIRS = {
    "train": Path("data/train"),
    "val": Path("data/val"),
    "test": Path("data/test"),
}
CLASSES = ["with_mask", "without_mask"]
RNG_SEED = 42


def build_working_copy(raw_root: Path, work_root: Path) -> None:
    """
    Copy only valid, unique images into data/working/<class>/.
    For duplicates (same MD5): keep the first seen file; skip the rest.
    """
    raw_root = Path(raw_root)
    work_root = Path(work_root)

    # 0) Ensure destination class directories exist
    for cls in CLASSES:
        (work_root / cls).mkdir(parents=True, exist_ok=True)

    seen_hashes: dict[str, Path] = {}   # md5 -> kept destination path
    n_total = n_ok = n_dupe = n_bad = n_outside = 0

    for src in iter_images(raw_root):
        n_total += 1
        cls = src.parent.name

        # 1) If file lives outside the expected classes, skip (but count it)
        if cls not in CLASSES:
            n_outside += 1
            continue

        # 2) Integrity check
        if not check_image_ok(src):
            n_bad += 1
            continue

        # 3) Content hash (used for dedup)
        h = file_hash(src)

        # 4) Duplicate? Skip
        if h in seen_hashes:
            n_dupe += 1
            continue

        # 5) Construct destination path (try to keep original name)
        dst_dir = work_root / cls
        dst = dst_dir / src.name

        # 6) Avoid name collisions inside the same class folder
        if dst.exists():
            stem = src.stem
            suffix = src.suffix.lower()
            dst = dst_dir / f"{h[:8]}_{stem}{suffix}"

        # 7) Copy and record hash
        shutil.copy2(src, dst)
        seen_hashes[h] = dst
        n_ok += 1

        # (optional) progress heartbeat
        if n_total % 1000 == 0:
            print(f"...processed {n_total:,} | kept={n_ok:,} dupes={n_dupe:,} bad={n_bad:,} outside={n_outside:,}")

    print(f"Done. Total={n_total:,} | kept={n_ok:,} | dupes={n_dupe:,} | bad={n_bad:,} | outside_class={n_outside:,}")


def split_sets(work_root: Path) -> dict[str, list[Path]]:
    """
    Random split into train/val/test (80/10/10), stratified by class.
    Returns dict: {split_name: [image_paths]}.
    """
    random.seed(RNG_SEED)
    splits: dict[str, list[Path]] = {"train": [], "val": [], "test": []}

    for cls in CLASSES:
        cls_dir = Path(work_root) / cls
        files = list(iter_images(cls_dir))
        # Reproducible shuffle per class keeps class balance in each split
        random.shuffle(files)

        n = len(files)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        n_test = n - (n_train + n_val)  # whatever remains

        splits["train"].extend(files[:n_train])
        splits["val"].extend(files[n_train:n_train + n_val])
        splits["test"].extend(files[n_train + n_val:])

    return splits


def materialize_splits(splits: dict[str, list[Path]], out_dirs: dict[str, Path]) -> None:
    """
    Copy files into train/val/test/<class>/ directories.
    Avoid overwriting by appending a short hash to the name on collision.
    """
    # Ensure destination dirs exist
    for split, out_root in out_dirs.items():
        for cls in CLASSES:
            (out_root / cls).mkdir(parents=True, exist_ok=True)

    for split, files in splits.items():
        out_root = out_dirs[split]
        for src in files:
            cls = src.parent.name
            dst = out_root / cls / src.name

            if dst.exists():
                # Disambiguate file name within the split/class folder
                stem, suffix = src.stem, src.suffix.lower()
                # Use a stable short hash derived from the full path string
                short_id = f"{abs(hash(str(src))) % (10**8):08d}"
                dst = out_root / cls / f"{stem}_{short_id}{suffix}"

            shutil.copy2(src, dst)


def main() -> int:
    assert RAW_DIR.exists(), f"Missing {RAW_DIR}. Place your raw dataset there."

    # 1) EDA
    print("== EDA: raw dataset ==")
    counts_raw = inventory(RAW_DIR)
    print_summary("Raw counts", counts_raw)

    dups = find_duplicates(RAW_DIR)
    print(f"Duplicate groups: {len(dups)}")
    sizes = size_stats(RAW_DIR)
    for cls, (min_w, min_h, max_w, max_h) in sizes.items():
        print(f"  {cls:>12}: min=({min_w}x{min_h})  max=({max_w}x{max_h})")

    # 2) Build working copy
    print("\n== Building working copy ==")
    build_working_copy(RAW_DIR, WORK_DIR)

    # 3) Split into train/val/test
    print("\n== Splitting train/val/test ==")
    splits = split_sets(WORK_DIR)
    materialize_splits(splits, OUT_DIRS)

    # 4) Verify
    for name, out_dir in OUT_DIRS.items():
        c = inventory(out_dir)
        print_summary(f"{name} counts", c)

    return 0


if __name__ == "__main__":
    sys.exit(main())
