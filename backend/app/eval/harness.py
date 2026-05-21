"""End-to-end eval harness: run the pipeline + score it.

Usage from Python
-----------------
.. code-block:: python

    from pathlib import Path
    from app.eval.harness import run_eval

    metrics = run_eval(
        scan_path=Path("demo/stevenson.laz"),
        gt_path=Path("demo/stevenson_gt.json"),
        out_dir=Path("eval_runs/2026-05-20"),
    )
    print(metrics.wall_f1)

Usage from CLI
--------------
.. code-block:: bash

    cd backend
    python -m app.eval \
        --scan demo/stevenson.laz \
        --gt   demo/stevenson_gt.json \
        --out  eval_runs/baseline

    # Then later, regression check against the baseline:
    python -m app.eval --compare eval_runs/baseline/metrics.json \
                       --against  eval_runs/2026-05-21/metrics.json
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from .ground_truth import GroundTruth, load_ground_truth
from .metrics import PipelineMetrics, compute_metrics, render_report


@dataclass
class EvalRun:
    """Bookkeeping for a single eval run."""
    scan_path: Path
    gt_path: Path
    result_dir: Path
    metrics_path: Path
    metrics: PipelineMetrics


def run_eval(
    scan_path: Path,
    gt_path: Path,
    out_dir: Path,
    params_overrides: Optional[dict] = None,
    match_iou: float = 0.30,
) -> EvalRun:
    """Run the vectorize pipeline against a scan and score it.

    Parameters
    ----------
    scan_path : Path
        Path to the scan file (.las, .laz, .ply, .e57).
    gt_path : Path
        Path to the ground-truth JSON (see ``ground_truth.GroundTruth``).
    out_dir : Path
        Output directory for the pipeline run + the metrics report.
    params_overrides : dict, optional
        Override VectorizeParams fields.  Useful for A/B tests of
        individual knobs without changing defaults.
    match_iou : float
        IoU threshold for segment matching.  See
        ``metrics.segment_precision_recall``.
    """
    from ..models.vectorize_job import VectorizeParams
    from ..vectorize.pipeline import run_vectorize
    import uuid as _uuid

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = out_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    gt = load_ground_truth(gt_path)

    params_dict = {}
    if params_overrides:
        params_dict.update(params_overrides)
    params = VectorizeParams(**params_dict)

    # The pipeline writes result.json itself; we just need to know if it
    # errored so we can surface that to the caller.
    captured: dict = {}

    def _on_error(msg: str) -> None:
        captured["error"] = msg

    job_id = f"eval-{_uuid.uuid4().hex[:8]}"
    t0 = time.monotonic()
    run_vectorize(
        job_id=job_id,
        scan_path=Path(scan_path),
        artifact_dir=artifact_dir,
        result_dir=out_dir,
        params=params,
        on_error=_on_error,
    )
    elapsed = time.monotonic() - t0

    if "error" in captured:
        raise RuntimeError(f"pipeline failed: {captured['error']}")

    pm = compute_metrics(out_dir, gt, match_iou=match_iou)
    # Use harness wall-clock if result.json didn't have one (older runs).
    if pm.runtime_s <= 0:
        pm.runtime_s = float(elapsed)

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(pm.to_dict(), indent=2))

    print(render_report(pm))

    return EvalRun(
        scan_path=Path(scan_path),
        gt_path=Path(gt_path),
        result_dir=out_dir,
        metrics_path=metrics_path,
        metrics=pm,
    )


def compare_metrics(
    baseline_path: Path,
    new_path: Path,
    regression_pct: float = 2.0,
) -> tuple[bool, list[str]]:
    """Compare two metrics.json files and flag regressions.

    Returns ``(passed, messages)``.  ``passed`` is True if no metric
    regressed by more than ``regression_pct`` (default 2 %).

    What counts as a regression:
      - wall_f1, wall_precision, wall_recall, coverage_pct, mean_segment_iou
        DROPPED by > regression_pct.
      - runtime_s INCREASED by > regression_pct.
      - rooms_detected count moved away from rooms_expected.
    """
    base = json.loads(Path(baseline_path).read_text())
    new = json.loads(Path(new_path).read_text())
    messages: list[str] = []
    passed = True

    def _check_drop(name: str, higher_is_better: bool = True):
        nonlocal passed
        b = float(base.get(name, 0.0))
        n = float(new.get(name, 0.0))
        if b == 0:
            return
        delta_pct = 100.0 * (n - b) / b
        if higher_is_better and delta_pct < -regression_pct:
            messages.append(
                f"REGRESS  {name}: {b:.3f} → {n:.3f}  ({delta_pct:+.1f}%)"
            )
            passed = False
        elif not higher_is_better and delta_pct > regression_pct:
            messages.append(
                f"REGRESS  {name}: {b:.3f} → {n:.3f}  ({delta_pct:+.1f}%)"
            )
            passed = False
        else:
            arrow = "↑" if delta_pct > 0 else "↓"
            messages.append(f"  ok     {name}: {b:.3f} → {n:.3f}  ({delta_pct:+.1f}% {arrow})")

    _check_drop("wall_f1")
    _check_drop("wall_precision")
    _check_drop("wall_recall")
    _check_drop("coverage_pct")
    _check_drop("mean_segment_iou")
    _check_drop("runtime_s", higher_is_better=False)

    # Rooms: any change is interesting, but tolerate ±1 vs baseline.
    base_rooms = int(base.get("rooms_detected", 0))
    new_rooms = int(new.get("rooms_detected", 0))
    if abs(new_rooms - base_rooms) > 1:
        passed = False
        messages.append(f"REGRESS  rooms_detected: {base_rooms} → {new_rooms}")
    else:
        messages.append(f"  ok     rooms_detected: {base_rooms} → {new_rooms}")

    return passed, messages
