"""Raster image conditioning for vectorization.

The slice produced by :mod:`.slicer` is a raw binary occupancy map — one pixel
per point that landed in the slab.  Real scans are noisy: walls are dotted
lines (not solid), scan-sweep artefacts add stray pixels, and occlusion gaps
break otherwise straight walls.

This module bridges that gap before detection runs.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreprocessParams:
    close_kernel_px: int = 3       # bridge ≤ 3-px gaps along walls
    close_iterations: int = 2
    open_kernel_px: int = 0        # disabled by default — removes thin walls if set too high
    open_iterations: int = 0
    blur_kernel_px: int = 3        # Gaussian blur before edge detection; 0 disables
    fill_holes: bool = False       # whether to fill enclosed regions (room interiors)


def preprocess(image: np.ndarray, params: PreprocessParams | None = None) -> np.ndarray:
    """Clean up a raw occupancy raster for line detection.

    Operates in three steps, all optional via params:
      1. Morphological CLOSE — bridges small gaps along walls (occlusion).
      2. Morphological OPEN  — removes isolated speckle (scanner noise).
      3. Gaussian blur       — softens jagged edges before Canny/Hough.

    Returns a uint8 grayscale image in the same orientation as the input.
    """
    if params is None:
        params = PreprocessParams()

    out = image.copy()

    if params.close_kernel_px > 0 and params.close_iterations > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (params.close_kernel_px, params.close_kernel_px),
        )
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k, iterations=params.close_iterations)

    if params.open_kernel_px > 0 and params.open_iterations > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (params.open_kernel_px, params.open_kernel_px),
        )
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k, iterations=params.open_iterations)

    if params.fill_holes:
        # SciPy version is more robust but adds a dep — implement via flood-fill instead.
        # Pad by 1, flood-fill from the corner, invert the result to fill enclosed holes.
        h, w = out.shape
        flood = out.copy()
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(flood, mask, (0, 0), 255)
        inv = cv2.bitwise_not(flood)
        out = cv2.bitwise_or(out, inv)

    if params.blur_kernel_px > 0:
        k = params.blur_kernel_px
        if k % 2 == 0:
            k += 1  # cv2 requires odd kernel
        out = cv2.GaussianBlur(out, (k, k), sigmaX=0)

    return out
