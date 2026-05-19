"""Point cloud → 2D raster image at a chosen elevation.

The vectorization pipeline operates on 2D raster slices rather than the raw
3D point cloud — this is the architectural choice the Stevenson roadmap calls
out as the key risk-reducer.  This module turns ``(point_cloud, elevation)``
into ``(raster_image, world↔pixel affine)`` and persists both for downstream
detection stages.

Coordinate convention
---------------------
Internal: pixel ``(col=x_idx, row=y_idx)`` maps to world ``(X, Y)`` as

    world_x = origin_x + col * resolution
    world_y = origin_y + row * resolution

That is, ``row=0`` is the lowest Y in the world (south).  This matches the
inline convention used in ``app/pipeline/scanplan.py`` so utility code can be
reused across both pipelines.

On disk, the PNG is written with row=0 flipped to the top (north-up) so that
the image looks like a normal plan view in any image viewer.  The
``raster_to_world`` / ``world_to_raster`` helpers operate on the *internal*
(unflipped) convention.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import open3d as o3d


# Hard upper bound on a single raster dimension — prevents OOM on a misconfigured
# resolution_m_per_px against a multi-hectare scan.  At 1 cm/px this allows up
# to a 160 m × 160 m building.  Drop resolution to fit larger sites.
MAX_RASTER_DIM_PX = 16_384


@dataclass
class RasterAffine:
    """World ↔ pixel mapping for a raster slice.

    Pixel ``(col, row)`` centre lies at world ``(origin_x + col * res,
    origin_y + row * res)``.
    """
    origin_x: float
    origin_y: float
    resolution_m_per_px: float
    width_px: int
    height_px: int

    def world_to_pixel(self, pts_xy: np.ndarray) -> np.ndarray:
        """Map an ``(N, 2)`` array of world XY to integer pixel ``(col, row)``."""
        cols = np.floor((pts_xy[:, 0] - self.origin_x) / self.resolution_m_per_px).astype(np.int32)
        rows = np.floor((pts_xy[:, 1] - self.origin_y) / self.resolution_m_per_px).astype(np.int32)
        return np.stack([cols, rows], axis=1)

    def pixel_to_world(self, pix: np.ndarray) -> np.ndarray:
        """Map an ``(N, 2)`` array of ``(col, row)`` to world XY (pixel centre)."""
        xs = self.origin_x + (pix[:, 0].astype(np.float64) + 0.5) * self.resolution_m_per_px
        ys = self.origin_y + (pix[:, 1].astype(np.float64) + 0.5) * self.resolution_m_per_px
        return np.stack([xs, ys], axis=1)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "RasterAffine":
        return cls(**data)


@dataclass
class SliceResult:
    """Output of :func:`slice_to_raster`."""
    image: np.ndarray            # (H, W) uint8, 0 or 255 — internal convention (row=0 = low Y)
    affine: RasterAffine
    elevation_m: float
    slab_thickness_m: float
    axis_idx: int                # 2 = Z-up, 1 = Y-up
    n_points_in_slab: int        # raw count before binning


def slice_to_raster(
    pcd: o3d.geometry.PointCloud,
    elevation_m: float,
    slab_thickness_m: float = 0.20,
    resolution_m_per_px: float = 0.01,
    axis_idx: int = 2,
    bbox_padding_m: float = 0.50,
) -> SliceResult:
    """Project the points within ``[elevation - slab/2, elevation + slab/2]`` to a 2D raster.

    Parameters
    ----------
    pcd : open3d.geometry.PointCloud
        Already-loaded point cloud.  Caller is responsible for the coordinate frame.
    elevation_m : float
        Absolute elevation along the vertical axis (in the point cloud's frame)
        at which to take the slice.
    slab_thickness_m : float
        Vertical thickness of the slab.  Points within
        ``[elevation - slab/2, elevation + slab/2]`` are kept.
    resolution_m_per_px : float
        World metres per raster pixel.
    axis_idx : int
        Index of the vertical axis.  2 = Z-up (standard LiDAR), 1 = Y-up.
    bbox_padding_m : float
        Extra margin added around the XY extent of the slab points.  Prevents
        wall lines being clipped at the raster boundary.

    Returns
    -------
    SliceResult
        Contains the binary image (uint8, values 0 or 255), the world↔pixel
        affine, and bookkeeping metadata.

    Raises
    ------
    ValueError
        If the slab is empty, the requested raster would exceed
        ``MAX_RASTER_DIM_PX``, or the axis index is invalid.
    """
    if axis_idx not in (1, 2):
        raise ValueError(f"axis_idx must be 1 (Y-up) or 2 (Z-up); got {axis_idx}")
    if resolution_m_per_px <= 0:
        raise ValueError(f"resolution_m_per_px must be positive; got {resolution_m_per_px}")

    pts = np.asarray(pcd.points, dtype=np.float64)
    if len(pts) == 0:
        raise ValueError("Point cloud is empty.")

    # Project to the 2D plane orthogonal to the vertical axis.
    # Z-up → use (X, Y); Y-up → use (X, Z).
    plane_cols = (0, 1) if axis_idx == 2 else (0, 2)

    vertical = pts[:, axis_idx]
    half = slab_thickness_m / 2.0
    mask = (vertical >= elevation_m - half) & (vertical <= elevation_m + half)
    n_in_slab = int(mask.sum())
    if n_in_slab == 0:
        raise ValueError(
            f"No points in slab at elevation {elevation_m:.2f} m "
            f"(±{half:.2f} m).  Vertical range of cloud: "
            f"[{vertical.min():.2f}, {vertical.max():.2f}] m."
        )

    pts_2d = pts[np.ix_(mask, plane_cols)]

    # XY extent + padding becomes the raster bounding box.
    x_min = float(pts_2d[:, 0].min()) - bbox_padding_m
    y_min = float(pts_2d[:, 1].min()) - bbox_padding_m
    x_max = float(pts_2d[:, 0].max()) + bbox_padding_m
    y_max = float(pts_2d[:, 1].max()) + bbox_padding_m

    width_px = int(np.ceil((x_max - x_min) / resolution_m_per_px))
    height_px = int(np.ceil((y_max - y_min) / resolution_m_per_px))

    if width_px > MAX_RASTER_DIM_PX or height_px > MAX_RASTER_DIM_PX:
        raise ValueError(
            f"Raster size {width_px}×{height_px} exceeds the hard cap of "
            f"{MAX_RASTER_DIM_PX} per side.  Increase resolution_m_per_px "
            f"(currently {resolution_m_per_px} m/px) or slice a smaller region."
        )
    if width_px < 4 or height_px < 4:
        raise ValueError(
            f"Raster size {width_px}×{height_px} is too small to contain useful "
            f"detail.  Decrease resolution_m_per_px or check the slab elevation."
        )

    cols = np.clip(
        np.floor((pts_2d[:, 0] - x_min) / resolution_m_per_px).astype(np.int32),
        0, width_px - 1,
    )
    rows = np.clip(
        np.floor((pts_2d[:, 1] - y_min) / resolution_m_per_px).astype(np.int32),
        0, height_px - 1,
    )

    image = np.zeros((height_px, width_px), dtype=np.uint8)
    image[rows, cols] = 255

    affine = RasterAffine(
        origin_x=x_min,
        origin_y=y_min,
        resolution_m_per_px=resolution_m_per_px,
        width_px=width_px,
        height_px=height_px,
    )

    return SliceResult(
        image=image,
        affine=affine,
        elevation_m=elevation_m,
        slab_thickness_m=slab_thickness_m,
        axis_idx=axis_idx,
        n_points_in_slab=n_in_slab,
    )


# ── Persistence helpers ──────────────────────────────────────────────────────

def save_slice(result: SliceResult, out_dir: Path, basename: str = "slice") -> dict:
    """Persist a :class:`SliceResult` to disk.

    Writes:
      - ``{basename}.png`` — north-up PNG (Y-flipped from internal convention)
      - ``{basename}.json`` — affine + metadata sidecar

    Returns a small dict suitable for inclusion in a job result payload.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{basename}.png"
    json_path = out_dir / f"{basename}.json"

    # Flip vertically only for display.  The on-disk PNG is therefore a "plan
    # view" — looks correct when viewed in any image viewer.  Code that loads
    # the PNG back as an internal raster MUST re-flip via cv2.flip(img, 0).
    image_display = cv2.flip(result.image, 0)
    cv2.imwrite(str(png_path), image_display)

    sidecar = {
        "affine": result.affine.to_json(),
        "elevation_m": result.elevation_m,
        "slab_thickness_m": result.slab_thickness_m,
        "axis_idx": result.axis_idx,
        "n_points_in_slab": result.n_points_in_slab,
        "y_axis_flipped_for_display": True,
    }
    json_path.write_text(json.dumps(sidecar, indent=2))
    return sidecar


def load_slice_png(png_path: Path) -> np.ndarray:
    """Load a slice PNG written by :func:`save_slice` back into internal convention.

    Reverses the Y-flip applied during save so detection runs on the same
    orientation the slicer originally produced.
    """
    img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read raster PNG at {png_path}")
    return cv2.flip(img, 0)
