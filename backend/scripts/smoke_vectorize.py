"""Smoke test for the vectorize pipeline — runs end-to-end against a local file.

Usage (from the repo root):
    backend/.venv/bin/python backend/scripts/smoke_vectorize.py \\
        "source-files/180LJ2-RR-f8a435d6-cc0d-44a1-a603-d130e3e1f9f3.laz copy" \\
        --out data/smoke

Outputs into ``<out>/<run_id>/``:
  - slice.png           the raw raster
  - slice_cleaned.png   after preprocess
  - overlay.png         clean detected lines on the raster
  - overlay_raw.png     raw detector output for comparison
  - vectorized.dxf      the actual deliverable
  - result.json         metrics + params + paths
"""
from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scan", type=Path, help="path to .las/.laz/.ply/.e57")
    parser.add_argument("--out", type=Path, default=Path("data/smoke"))
    parser.add_argument("--elevation", type=float, default=None,
                        help="absolute elevation (m); default = auto-detect floor + 1.4")
    parser.add_argument("--slab", type=float, default=0.20)
    parser.add_argument("--resolution", type=float, default=0.01,
                        help="metres per pixel; 0.01 = 1 cm/px (default)")
    parser.add_argument("--detector", choices=["hough", "fld", "both"], default="fld")
    parser.add_argument("--min-wall-length", type=float, default=0.50)
    parser.add_argument("--no-manhattan", action="store_true")
    parser.add_argument("--no-merge", action="store_true")
    args = parser.parse_args()

    if not args.scan.exists():
        print(f"error: scan file does not exist: {args.scan}", file=sys.stderr)
        return 1

    # Late import so --help works without paying the cv2 + open3d import cost.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.models.vectorize_job import DetectorName, VectorizeParams
    from app.vectorize.pipeline import run_vectorize

    run_id = uuid.uuid4().hex[:8]
    artifact_dir = args.out / run_id
    result_dir = artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    params = VectorizeParams(
        elevation_m=args.elevation,
        slab_thickness_m=args.slab,
        resolution_m_per_px=args.resolution,
        detector=DetectorName(args.detector),
        min_wall_length_m=args.min_wall_length,
        manhattan_snap=not args.no_manhattan,
        merge_collinear=not args.no_merge,
    )

    print(f"smoke run {run_id}")
    print(f"  scan:   {args.scan}")
    print(f"  out:    {artifact_dir}")
    print(f"  params: {params.model_dump(mode='json')}")

    # Re-route SSE events to stdout so we get progress in the terminal.
    from app import sse as _sse
    def _print_publish(job_id, stage, message, progress):
        print(f"  [{progress * 100:5.1f}%] {stage:>10s} | {message}")
    _orig = _sse.publish
    _sse.publish = _print_publish  # type: ignore[assignment]

    try:
        t0 = time.time()
        run_vectorize(
            job_id=run_id,
            scan_path=args.scan,
            artifact_dir=artifact_dir,
            result_dir=result_dir,
            params=params,
        )
        print(f"done in {time.time() - t0:.1f} s")
        print(f"artifacts: {artifact_dir}")
    finally:
        _sse.publish = _orig  # type: ignore[assignment]

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
