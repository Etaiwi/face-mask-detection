from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path
from typing import Optional, Callable


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_weights(
    weights_path: Path,
    url: str,
    expected_sha256: Optional[str] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """
    Ensure model weights exist at weights_path.
    If missing, download from url.
    Optionally verifies SHA256 and reports progress (downloaded_bytes, total_bytes).
    """
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    if weights_path.exists() and weights_path.stat().st_size > 0:
        if expected_sha256:
            if _sha256(weights_path) != expected_sha256.lower():
                weights_path.unlink(missing_ok=True)
            else:
                return weights_path
        else:
            return weights_path

    # Download to temp then atomically move
    tmp_path = weights_path.with_suffix(weights_path.suffix + ".tmp")

    def reporthook(block_num: int, block_size: int, total_size: int):
        downloaded = block_num * block_size
        if progress_cb:
            progress_cb(downloaded, total_size)

    urllib.request.urlretrieve(url, tmp_path, reporthook=reporthook)

    # Basic sanity
    if tmp_path.stat().st_size == 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError("Downloaded weights file is empty.")

    tmp_path.replace(weights_path)

    if expected_sha256:
        actual = _sha256(weights_path)
        if actual != expected_sha256.lower():
            weights_path.unlink(missing_ok=True)
            raise RuntimeError("Weights SHA256 mismatch after download.")

    return weights_path
