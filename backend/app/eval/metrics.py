"""Quantitative metrics for the vectorize pipeline.

Implementations are dependency-light (numpy + shapely + json) so this
module is usable from CI without spinning up Open3D / CUDA.

Metric definitions
------------------

**Segment IoU** (per ground-truth segment)
    Buffer each segment by ``tolerance_m`` into a thin oriented rectangle.
    For a GT segment G and the closest detected segment D, IoU = area(G ∩ D)
    / area(G ∪ D).  Range 0–1; > 0.5 means "the detected line covers most
    of the GT line and doesn't extend much beyond it".

**Wall precision** (whole-run)
    # detected walls that match at least one GT wall (IoU ≥ ``match_iou``)
    / total detected walls.  Penalises noise detections.

**Wall recall** (whole-run)
    # GT walls that match at least one detected wall (IoU ≥ ``match_iou``)
    / total GT walls.  Penalises missed walls.

**Wall F1**
    Harmonic mean of precision + recall.  Single number for trend tracking.

**Coverage**
    Lifted directly from the pipeline's own coverage diagnostic
    (fraction of raster wall pixels within ``radius`` of a kept segment).
    Already in :class:`models.vectorize_job.VectorizeMetrics`; we just
    surface it in the eval report.

**Rooms / columns / openings counts**
    Compared to ground truth's expected counts; rendered as ratios.

**Runtime**
    Compared to ``runtime_budget_s``.  A run that exceeds the budget by
    > 20 % fails the eval; > 50 % is a hard regression flag.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from shapely.geometry import LineString
    from shapely.ops import unary_union
    _HAS_SHAPELY = True
except Exception:
    _HAS_SHAPELY = False


@dataclass
class PipelineMetrics:
    """Aggregate metrics for one pipeline run vs ground truth."""

    # Wall geometry — primary correctness metric.
    wall_precision: float = 0.0
    wall_recall: float = 0.0
    wall_f1: float = 0.0
    median_segment_iou: float = 0.0
    mean_segment_iou: float = 0.0

    # Counts — secondary checks.
    walls_detected: int = 0
    walls_expected: int = 0
    rooms_detected: int = 0
    rooms_expected: int = 0
    columns_detected: int = 0
    columns_expected: int = 0
    openings_detected: int = 0
    openings_expected: int = 0

    # Operational metrics.
    runtime_s: float = 0.0
    runtime_budget_s: float = 0.0
    coverage_pct: float = 0.0

    # Pass/fail summary (booleans for CI).
    pass_runtime: bool = True
    pass_walls: bool = True
    pass_rooms: bool = True

    # Per-segment IoU samples for distribution analysis.
    iou_samples: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Drop the long list when serialising — keep it for in-memory use only.
        d.pop("iou_samples", None)
        return d

    @property
    def runtime_overrun_pct(self) -> float:
        if self.runtime_budget_s <= 0:
            return 0.0
        return 100.0 * (self.runtime_s / self.runtime_budget_s - 1.0)


# ── Per-segment IoU (the workhorse) ──────────────────────────────────────────

def segment_iou(
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    tolerance_m: float,
) -> float:
    """Compute IoU of two segments using a thin oriented buffer.

    ``seg_a`` and ``seg_b`` are each ``(2, 2)`` arrays of ``[[x1,y1],[x2,y2]]``.
    ``tolerance_m`` is the half-width of the buffer (i.e. the buffer is
    rectangular with width 2*tolerance_m perpendicular to the segment).

    Returns 0.0 when shapely is unavailable; callers should treat that as
    "skip this comparison" rather than "no match".

    Implementation note: ``LineString.buffer(d, cap_style=2)`` produces a
    flat (rectangular) buffer, which is what we want for line-on-line IoU.
    The default round buffer would systematically over-rate co-linear but
    end-misaligned segments.
    """
    if not _HAS_SHAPELY:
        return 0.0
    line_a = LineString([tuple(seg_a[0]), tuple(seg_a[1])])
    line_b = LineString([tuple(seg_b[0]), tuple(seg_b[1])])
    if line_a.length == 0 or line_b.length == 0:
        return 0.0
    poly_a = line_a.buffer(tolerance_m, cap_style=2)
    poly_b = line_b.buffer(tolerance_m, cap_style=2)
    inter = poly_a.intersection(poly_b).area
    union = poly_a.union(poly_b).area
    if union <= 0:
        return 0.0
    return float(inter / union)


def segment_precision_recall(
    detected: np.ndarray,
    ground_truth: np.ndarray,
    tolerance_m: float,
    match_iou: float = 0.30,
) -> tuple[float, float, list[float]]:
    """Compute precision, recall, and per-GT-segment max-IoU.

    A "match" is a detected segment whose IoU vs the GT segment is at
    least ``match_iou``.  Default 0.30 — generous enough that a detected
    segment that's slightly shorter than the GT (typical wall-end
    overshoot in the rasterized truth) still counts as a hit.  Tighten
    to 0.5 for ML-quality evaluation.

    Returns (precision, recall, ious).  ``ious`` is a list of best-match
    IoUs per GT segment, useful for histogram plots.
    """
    if not _HAS_SHAPELY:
        return 0.0, 0.0, []
    n_det = len(detected)
    n_gt = len(ground_truth)
    if n_det == 0 and n_gt == 0:
        return 1.0, 1.0, []
    if n_det == 0:
        return 0.0, 0.0, [0.0] * n_gt
    if n_gt == 0:
        return 0.0, 1.0, []

    # All-pairs IoU.  Trivial O(N*M) — fine for typical N, M < 1000.
    iou_mat = np.zeros((n_det, n_gt), dtype=np.float64)
    for i in range(n_det):
        for j in range(n_gt):
            iou_mat[i, j] = segment_iou(detected[i], ground_truth[j], tolerance_m)

    det_best = iou_mat.max(axis=1) if n_gt > 0 else np.zeros(n_det)
    gt_best = iou_mat.max(axis=0) if n_det > 0 else np.zeros(n_gt)

    det_hit = (det_best >= match_iou).sum()
    gt_hit = (gt_best >= match_iou).sum()

    precision = float(det_hit) / float(n_det)
    recall = float(gt_hit) / float(n_gt)
    return precision, recall, gt_best.tolist()


# ── End-to-end metric computation ────────────────────────────────────────────

def compute_metrics(
    result_dir: Path,
    gt,                                     # GroundTruth — forward ref for circular import safety
    match_iou: float = 0.30,
) -> PipelineMetrics:
    """Read a pipeline result_dir and compare to ground truth.

    Parameters
    ----------
    result_dir : Path
        The directory containing ``result.json`` + ``segments.json`` from
        a pipeline run.  We avoid re-running the pipeline so this can be
        used in CI against pre-captured outputs.
    gt : GroundTruth
        The expected geometry / counts.
    match_iou : float
        IoU threshold for "this segment matches".  See
        :func:`segment_precision_recall`.

    Returns
    -------
    PipelineMetrics
    """
    import json as _json

    result_path = Path(result_dir) / "result.json"
    segments_path = Path(result_dir) / "segments.json"
    if not result_path.exists():
        raise FileNotFoundError(f"missing result.json in {result_dir}")
    if not segments_path.exists():
        raise FileNotFoundError(f"missing segments.json in {result_dir}")

    result_data = _json.loads(result_path.read_text())
    segments_data = _json.loads(segments_path.read_text())

    # Pull the WALLS layer specifically (the centerlines, post-pairing) —
    # we score against those, not the WALLS_FACES doubles.
    detected_walls_list = [
        s for s in segments_data.get("segments", []) if s["layer"] == "walls"
    ]
    detected_walls = np.array([
        [[s["x1"], s["y1"]], [s["x2"], s["y2"]]] for s in detected_walls_list
    ], dtype=np.float64) if detected_walls_list else np.zeros((0, 2, 2), dtype=np.float64)

    precision, recall, ious = segment_precision_recall(
        detected_walls, gt.walls, gt.tolerance_m, match_iou=match_iou,
    )
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    metrics_dict = result_data.get("metrics", {}) or {}
    runtime = float(metrics_dict.get("processing_time_s")
                    or result_data.get("processing_time_s", 0.0))
    coverage = float(metrics_dict.get("coverage_pct", 0.0))
    rooms_detected = int(metrics_dict.get("rooms_detected") or 0)
    columns_detected = int(metrics_dict.get("columns_detected") or 0)
    openings_detected = int(metrics_dict.get("openings_detected") or 0)

    pm = PipelineMetrics(
        wall_precision=precision,
        wall_recall=recall,
        wall_f1=f1,
        median_segment_iou=float(np.median(ious)) if ious else 0.0,
        mean_segment_iou=float(np.mean(ious)) if ious else 0.0,
        walls_detected=int(len(detected_walls)),
        walls_expected=int(gt.n_walls),
        rooms_detected=rooms_detected,
        rooms_expected=int(gt.rooms_expected),
        columns_detected=columns_detected,
        columns_expected=int(gt.columns_expected),
        openings_detected=openings_detected,
        openings_expected=int(gt.openings_expected),
        runtime_s=runtime,
        runtime_budget_s=float(gt.runtime_budget_s),
        coverage_pct=coverage,
        iou_samples=list(ious),
    )

    # Pass/fail booleans for CI.
    pm.pass_runtime = (runtime <= gt.runtime_budget_s * 1.20)  # 20% slack
    pm.pass_walls = (f1 >= 0.70)                               # > 70% F1 required
    pm.pass_rooms = (
        gt.rooms_expected == 0 or
        abs(rooms_detected - gt.rooms_expected) <= max(1, int(0.20 * gt.rooms_expected))
    )
    return pm


def render_report(pm: PipelineMetrics) -> str:
    """Pretty-print a metrics report for stdout / CI logs."""
    lines = [
        "─── Vectorize pipeline eval ─────────────────────────────────────",
        f"  Walls    precision: {pm.wall_precision * 100:5.1f}%  "
        f"recall: {pm.wall_recall * 100:5.1f}%  "
        f"F1: {pm.wall_f1 * 100:5.1f}%    "
        f"({pm.walls_detected} detected / {pm.walls_expected} expected)",
        f"  IoU      median:    {pm.median_segment_iou:.3f}   "
        f"mean: {pm.mean_segment_iou:.3f}",
        f"  Coverage walls:     {pm.coverage_pct:5.1f}%",
        f"  Rooms    detected:  {pm.rooms_detected}  /  {pm.rooms_expected} expected",
        f"  Columns  detected:  {pm.columns_detected}  /  {pm.columns_expected} expected",
        f"  Openings detected:  {pm.openings_detected}  /  {pm.openings_expected} expected",
        f"  Runtime:            {pm.runtime_s:6.1f} s   (budget "
        f"{pm.runtime_budget_s:.0f} s, overrun {pm.runtime_overrun_pct:+5.1f}%)",
        "─── Pass/fail ──────────────────────────────────────────────────",
        f"  runtime: {'PASS' if pm.pass_runtime else 'FAIL'}    "
        f"walls: {'PASS' if pm.pass_walls else 'FAIL'}    "
        f"rooms: {'PASS' if pm.pass_rooms else 'FAIL'}",
    ]
    return "\n".join(lines)
