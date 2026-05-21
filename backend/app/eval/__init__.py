"""Vectorize-pipeline evaluation harness.

Why this module exists
----------------------
The classical pipeline (v0.6, post Phase A–E) is good enough to ship.  The
ML drop-in track (v0.7+) is going to start swapping individual stages
(wall extractor, opening detector, room reconstructor) for learned models.
Without quantitative regression detection we will silently degrade quality
while chasing improvements somewhere else.

This module provides:

  - :class:`metrics.PipelineMetrics` — IoU + precision/recall on walls,
    coverage, rooms detected, runtime — computed from a pipeline run's
    output against a ground-truth JSON file.
  - :class:`ground_truth.GroundTruth` — schema for a tiny, human-editable
    JSON file describing what *should* have been detected (walls, rooms,
    columns, openings).
  - :class:`harness.EvalRun` — end-to-end "run pipeline + compute metrics
    + write report" loop.
  - ``python -m app.eval`` — CLI for capturing baselines + checking new
    runs against them.

This module does NOT depend on the rest of the vectorize pipeline at
import time so it can be reused by CI (no Open3D/CUDA needed for metric
computation against a pre-existing result.json).
"""
from .metrics import (
    PipelineMetrics,
    compute_metrics,
    segment_iou,
    segment_precision_recall,
)
from .ground_truth import GroundTruth, load_ground_truth, save_ground_truth
from .harness import EvalRun, compare_metrics, run_eval

__all__ = [
    "PipelineMetrics",
    "compute_metrics",
    "segment_iou",
    "segment_precision_recall",
    "GroundTruth",
    "load_ground_truth",
    "save_ground_truth",
    "EvalRun",
    "run_eval",
    "compare_metrics",
]
