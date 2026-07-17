"""Classical line detectors operating on a 2D raster image.

This is the Phase 0/1 detection layer.  Two detectors are wrapped behind a
uniform interface; both return an ``(N, 4)`` array of segments in image-pixel
coordinates with columns ``[x1, y1, x2, y2]``.

- :func:`detect_hough` — probabilistic Hough transform.  Mature, well-tuned,
  but produces many short fragments along noisy edges.
- :func:`detect_fld`  — Fast Line Detector (``cv2.ximgproc.createFastLineDetector``).
  The modern LSD replacement.  Cleaner long segments on architectural edges.
  Requires opencv-contrib (we depend on ``opencv-contrib-python-headless``).

Both detectors work on a preprocessed raster.  Pre-run the slice through
:func:`backend.app.vectorize.preprocess.preprocess` first.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class HoughParams:
    canny_threshold1: int = 40
    canny_threshold2: int = 120
    hough_rho_px: float = 1.0
    hough_theta_deg: float = 0.5      # angular resolution
    hough_vote_threshold: int = 30    # min votes per accepted line
    min_line_length_px: int = 20      # in pixels — converted from metres upstream
    max_line_gap_px: int = 8


@dataclass
class FldParams:
    length_threshold: int = 20        # pixels — segments shorter than this are dropped
    distance_threshold: float = 1.41  # pixels — points within this of the segment count as inliers
    canny_th1: float = 40.0
    canny_th2: float = 120.0
    canny_aperture: int = 3
    do_merge: bool = True             # FLD's own collinear merging


def detect_hough(
    image: np.ndarray,
    params: HoughParams | None = None,
) -> np.ndarray:
    """Detect line segments via Canny + probabilistic Hough.

    Returns an ``(N, 4)`` int32 array of ``[x1, y1, x2, y2]`` in pixel
    coordinates (same orientation as ``image``).  Returns shape ``(0, 4)``
    when no lines are found.
    """
    if params is None:
        params = HoughParams()

    edges = cv2.Canny(
        image,
        threshold1=params.canny_threshold1,
        threshold2=params.canny_threshold2,
        L2gradient=True,
    )

    raw = cv2.HoughLinesP(
        edges,
        rho=params.hough_rho_px,
        theta=np.deg2rad(params.hough_theta_deg),
        threshold=params.hough_vote_threshold,
        minLineLength=params.min_line_length_px,
        maxLineGap=params.max_line_gap_px,
    )

    if raw is None or len(raw) == 0:
        return np.zeros((0, 4), dtype=np.int32)
    return raw.reshape(-1, 4).astype(np.int32)


def detect_fld(
    image: np.ndarray,
    params: FldParams | None = None,
) -> np.ndarray:
    """Detect line segments via the Fast Line Detector (LSD replacement).

    Returns an ``(N, 4)`` float32 array of ``[x1, y1, x2, y2]`` in pixel
    coordinates.  Returns shape ``(0, 4)`` when no lines are found or when
    the ximgproc module isn't available (e.g. wrong opencv build).
    """
    if params is None:
        params = FldParams()

    if not hasattr(cv2, "ximgproc"):
        # Caller installed `opencv-python-headless` instead of
        # `opencv-contrib-python-headless`.  Fail soft — caller can fall
        # back to Hough.
        return np.zeros((0, 4), dtype=np.float32)

    fld = cv2.ximgproc.createFastLineDetector(  # type: ignore[attr-defined]
        length_threshold=params.length_threshold,
        distance_threshold=params.distance_threshold,
        canny_th1=params.canny_th1,
        canny_th2=params.canny_th2,
        canny_aperture_size=params.canny_aperture,
        do_merge=params.do_merge,
    )
    raw = fld.detect(image)
    if raw is None or len(raw) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return raw.reshape(-1, 4).astype(np.float32)


def merge_detectors(*detector_outputs: np.ndarray) -> np.ndarray:
    """Concatenate segments from multiple detectors into one ``(N, 4)`` array.

    No deduplication is performed here — that is :mod:`.regularize`'s job
    (``merge_collinear``).  This is purely a convenience for the ``both``
    detector mode.
    """
    arrays = [a for a in detector_outputs if a is not None and len(a) > 0]
    if not arrays:
        return np.zeros((0, 4), dtype=np.float32)
    return np.vstack([a.astype(np.float32) for a in arrays])


# ── Pixel-segment → world-segment conversion ──────────────────────────────────

def segments_pixels_to_world(
    segments_px: np.ndarray,
    affine,                              # RasterAffine — kept untyped to avoid circular import
) -> np.ndarray:
    """Convert ``(N, 4)`` pixel segments to ``(N, 2, 2)`` world segments.

    The output shape ``(N, 2, 2)`` is ``[[x1, y1], [x2, y2]]`` per segment —
    the same shape used elsewhere in the codebase (see ``scanplan.py``).

    Uses the pixel-CENTRE convention (+0.5) to match
    ``RasterAffine.pixel_to_world`` — omitting it introduced a systematic
    half-pixel (5 mm at 1 cm/px) bias versus the contour wall extractor.
    """
    if len(segments_px) == 0:
        return np.zeros((0, 2, 2), dtype=np.float64)

    seg = segments_px.astype(np.float64)
    res = affine.resolution_m_per_px
    ox, oy = affine.origin_x, affine.origin_y

    x1w = ox + (seg[:, 0] + 0.5) * res
    y1w = oy + (seg[:, 1] + 0.5) * res
    x2w = ox + (seg[:, 2] + 0.5) * res
    y2w = oy + (seg[:, 3] + 0.5) * res

    out = np.zeros((len(seg), 2, 2), dtype=np.float64)
    out[:, 0, 0] = x1w
    out[:, 0, 1] = y1w
    out[:, 1, 0] = x2w
    out[:, 1, 1] = y2w
    return out


# ── Overlay rendering for visual review ───────────────────────────────────────

def render_overlay(
    image: np.ndarray,
    segments_px: np.ndarray,
    color_bgr: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 1,
) -> np.ndarray:
    """Draw detected segments on top of the raster for visual inspection.

    Returns a BGR image (the raster is converted from grayscale).  Caller is
    responsible for any Y-flip needed before display.
    """
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in segments_px.astype(np.int32):
        cv2.line(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color_bgr, thickness)
    return overlay
