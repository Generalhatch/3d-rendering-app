"""Persistent cache of voxel-downsampled clouds (Phase 4.5).

Why this exists
---------------
Reprocessing a job re-ingests the raw scan from scratch: for a 680 MB LAS
that is minutes of decompression plus the full assembly memory spike —
all to rebuild a ~3 M-point downsampled cloud that is byte-identical to the
one the previous run already computed.  The operator iterating on detector /
regularize knobs pays that cost on every single run.

After vectorize stage 1b downsamples the cloud, we persist the result keyed
by ``(scan content hash, requested voxel size, auto flag)``.  A reprocess
with the same key loads the cached cloud and skips raw ingest entirely —
the run resumes at floor detection.

Correctness requirements:

- Points are stored as **lossless float64** (``np.savez`` / optionally
  ``np.savez_compressed`` — exact round-trip).  A cache hit must produce
  results byte-identical to a cache miss; float32 PLY would move every
  coordinate.  Clouds above ``_COMPRESS_MAX_POINTS`` skip zlib: compressing
  tens of millions of float64 points can take many minutes with no
  progress events and freezes the UI after downsample.
- The cached cloud is captured AFTER unit normalization (metres) and AFTER
  downsampling, but BEFORE gravity re-leveling / any filtering — exactly
  the state the pipeline is in at the end of stage 1b.  Unit detection is
  therefore cached *with* the cloud; it never needs re-deriving.
- The original load's ``raw_points`` and downsample counts ride along in a
  meta sidecar so job metrics are identical either way.

The cache is an optimization tier: it participates in the size-capped LRU
eviction (see ``storage.enforce_storage_cap``) — loads touch the file mtime
so recently used entries survive eviction longest.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import open3d as o3d

from ..config import get_settings

CACHE_VERSION = 1

# zlib on ~80 M float64 points is multi-minute CPU with a silent UI.  Below
# this count, compressed npz is fine and saves disk; above it, write
# uncompressed (still lossless, typically only ~2× larger for float64).
_COMPRESS_MAX_POINTS = 5_000_000


def cache_dir() -> Path:
    p = get_settings().data_dir / "cache" / "downsample"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _key(scan_hash: str, voxel_m: float, auto: bool) -> str:
    # Voxel encoded at fixed precision so float repr drift can't split keys.
    return f"{scan_hash}_vox{voxel_m:.6f}_auto{int(bool(auto))}"


def cache_paths(scan_hash: str, voxel_m: float, auto: bool) -> tuple[Path, Path]:
    """(points .npz, meta .json) paths for one cache key.

    Extensions are appended (NOT ``with_suffix`` — the key itself contains
    dots from the voxel size, and with_suffix would truncate the key)."""
    key = _key(scan_hash, voxel_m, auto)
    d = cache_dir()
    return d / f"{key}.npz", d / f"{key}.json"


@dataclass
class CachedDownsample:
    points: np.ndarray   # (N, 3) float64, metres, post-downsample
    meta: dict

    def to_point_cloud(self) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        return pcd


def load_cache(
    scan_hash: str, voxel_m: float, auto: bool,
) -> CachedDownsample | None:
    """Return the cached downsample for this key, or None on a miss.

    A hit touches both files' mtimes so the LRU eviction in
    ``storage.enforce_storage_cap`` sees them as recently used.
    """
    npz_path, meta_path = cache_paths(scan_hash, voxel_m, auto)
    if not (npz_path.exists() and meta_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text())
        if meta.get("version") != CACHE_VERSION:
            return None
        with np.load(npz_path) as data:
            points = np.ascontiguousarray(data["points"], dtype=np.float64)
    except Exception:
        return None

    now = time.time()
    for p in (npz_path, meta_path):
        try:
            os.utime(p, (now, now))
        except OSError:
            pass
    return CachedDownsample(points=points, meta=meta)


def store_cache(
    scan_hash: str,
    voxel_m: float,
    auto: bool,
    pcd: o3d.geometry.PointCloud,
    *,
    effective_voxel_m: float,
    raw_points: int,
    n_before: int,
) -> Path:
    """Persist a downsampled cloud + meta.  Write is atomic (tmp + rename)
    so a crash mid-write can't leave a truncated entry that later poisons a
    reprocess."""
    npz_path, meta_path = cache_paths(scan_hash, voxel_m, auto)
    points = np.asarray(pcd.points, dtype=np.float64)

    tmp_path = npz_path.with_name(npz_path.name + ".tmp")
    saver = (
        np.savez_compressed
        if len(points) <= _COMPRESS_MAX_POINTS
        else np.savez
    )
    try:
        with open(tmp_path, "wb") as f:
            saver(f, points=points)
        tmp_path.rename(npz_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    meta = {
        "version": CACHE_VERSION,
        "scan_hash": scan_hash,
        "requested_voxel_m": float(voxel_m),
        "voxel_m": float(effective_voxel_m),
        "auto": bool(auto),
        "raw_points": int(raw_points),
        "n_before": int(n_before),
        "n_after": int(len(points)),
        "units": "metres",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return npz_path
