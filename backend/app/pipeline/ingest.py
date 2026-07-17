"""Point cloud ingestion: LAS, LAZ, PLY, E57 → Open3D PointCloud.

Also produces a decimated PLY for the browser viewer.

Phase 4 fixes applied:
  LAZ-1/2  Percentage-based classification mask — handles interior terrestrial scans
           where ASPRS class 6 is absent (Leica RTC360, FARO Focus, NavVis VLX).
  LAZ-3    Intensity percentile normalization (p2–p98) — eliminates outlier collapse.
  LAZ-4    RGB bit-depth auto-detection — handles 8-bit values stored in uint16 fields.
  LAZ-5    Chunked streaming via laspy.open() for files > 500 MB — limits peak RAM.
  LAZ-6    Analytical voxel estimate — replaces 20-call binary search with 1 call.
  LAZ-7    E57 multi-scan support — iterates all scan positions, not just index 0.
  LAZ-8    LAS header logging — logs version, point count, scale, and bounding box.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import open3d as o3d

# Progress callback for long loads: ``on_progress(points_read, total_points)``.
# Invoked after every chunk in the chunked LAS path so callers can surface
# incremental progress during multi-minute loads of multi-GB scans.
ProgressCallback = Callable[[int, int], None]

# Assembly-phase callback: fired once per named phase AFTER the last chunk,
# so callers can keep the progress bar moving through the (previously silent)
# window between "last chunk read" and "cloud ready".  Phases, in order:
#   "classify"    — building + applying the file-global classification mask
#   "assemble"    — copying the surviving points into Open3D
#   "colors"      — color / intensity normalization (only when loaded)
#   "unit_detect" — feet/metres detection over the assembled cloud
# Callers must tolerate unknown phase names (forward compatibility).
PhaseCallback = Callable[[str], None]

# ASPRS noise classes always excluded when classification data is available.
_NOISE_CLASSES = frozenset({7, 18})  # Low noise (7), High noise (18)

# Files larger than this are read in chunks to cap peak RAM usage.
_CHUNK_THRESHOLD_BYTES = 500 * 1024 * 1024  # 500 MB

# US survey / international foot → metre.
_FT_TO_M = 0.3048

# Plausible interior floor-to-ceiling heights, in metres.  Covers residential
# (2.2 m) through warehouse/office-lobby (5.5 m).  A feet-based scan read as
# metres shows 7–15 "metre" ceilings — far outside this window — which is the
# unambiguous detection signal (10.76× area error if missed).
_CEILING_MIN_M = 2.0
_CEILING_MAX_M = 5.5

# Plausible wall thickness in metres (drywall partition → thick masonry).
_WALL_THICKNESS_MIN_M = 0.04
_WALL_THICKNESS_MAX_M = 0.70

# Unit detection only needs a height histogram + a mid-height wall slice.
# Running those over a 140 M-point cloud is minutes of pure numpy with no
# progress events — the UI looks hung at "Detecting scan units…".  2 M
# random samples is plenty for a stable floor/ceiling peak gap.
_UNIT_DETECT_MAX_POINTS = 2_000_000


class UnitDetectionError(ValueError):
    """Raised when the scan's linear unit cannot be determined and the
    geometry is implausible under both metres and feet."""


@dataclass
class UnitDetection:
    """Result of the ingest-time unit heuristic.

    ``unit`` is the detected unit of the RAW file ("m" or "ft");
    ``scale_to_m`` is the factor applied to convert to metres (1.0 for
    metres).  ``method`` records which heuristic decided:

    - ``ceiling_gap``: floor→ceiling peak gap fell in the plausible window
      for exactly one unit.
    - ``wall_thickness``: ceiling gap was inconclusive but the median wall
      thickness was plausible for exactly one unit.
    - ``assumed_metres``: no usable vertical structure (e.g. outdoor or
      single-plane scan) — left unchanged, flagged for the caller.
    """
    unit: str                          # "m" | "ft"
    scale_to_m: float
    method: str                        # "ceiling_gap" | "wall_thickness" | "assumed_metres"
    floor_to_ceiling_raw: float | None  # in raw file units, None if not found
    wall_thickness_raw: float | None    # in raw file units, None if not measured


def load_point_cloud(
    path: Path,
    normalize_units: bool = True,
    on_progress: Optional[ProgressCallback] = None,
    load_colors: bool = True,
    on_phase: Optional[PhaseCallback] = None,
) -> o3d.geometry.PointCloud:
    """Load a point cloud from LAS/LAZ, PLY, or E57.

    When ``normalize_units`` is True (default), the linear unit of the file
    is detected from interior geometry (floor-to-ceiling height, wall
    thickness) and feet-based scans are converted to metres in place.  A scan
    whose geometry is implausible under both interpretations raises
    :class:`UnitDetectionError` — a silent wrong-unit assumption corrupts
    every downstream area by 10.76×, so we fail loudly instead.

    ``on_progress(points_read, total_points)`` is invoked periodically during
    chunked LAS/LAZ streaming (files > 500 MB) so a multi-minute load isn't
    silent.  Other formats load in one shot and don't report progress.

    ``load_colors=False`` skips LAS/LAZ color/intensity loading entirely —
    the vectorize pipeline only consumes geometry, and accumulating colors
    for a 680 MB scan wastes gigabytes of RAM.  (PLY/E57 loads are one-shot
    and keep whatever the file carries.)

    ``on_phase(name)`` is fired once per post-read assembly phase
    ("classify" / "assemble" / "colors" / "unit_detect") so callers can keep
    the progress bar honest during the memory-hungry window after the last
    chunk callback.
    """
    suffix = path.suffix.lower()
    if suffix in (".las", ".laz"):
        pcd = _load_las(
            path, on_progress=on_progress, load_colors=load_colors,
            on_phase=on_phase,
        )
    elif suffix == ".ply":
        pcd = o3d.io.read_point_cloud(str(path))
        if len(pcd.points) == 0:
            raise ValueError(f"Empty point cloud: {path}")
    elif suffix == ".e57":
        pcd = _load_e57(path)
    else:
        raise ValueError(
            f"Unsupported scan format: {suffix}. Expected .las, .laz, .ply, .e57"
        )

    if normalize_units:
        if on_phase is not None:
            on_phase("unit_detect")
        pcd, detection = normalize_units_to_metres(pcd)
        if detection.scale_to_m != 1.0:
            print(
                f"  Units: detected {detection.unit} (method={detection.method}, "
                f"floor-to-ceiling={detection.floor_to_ceiling_raw}) — converted to metres"
            )
    return pcd


# ---------------------------------------------------------------------------
# Unit detection (feet vs metres)
# ---------------------------------------------------------------------------

def _find_floor_ceiling_gap(pts: np.ndarray, axis_idx: int = 2) -> float | None:
    """Estimate the floor-to-ceiling distance from vertical density peaks.

    Interior scans put dense horizontal slabs at the floor and ceiling.  We
    histogram the vertical coordinate and take the gap between the two most
    populated well-separated peaks.  Returns None when no such structure
    exists (outdoor scan, single plane, too few points).
    """
    if len(pts) < 1_000:
        return None
    z = pts[:, axis_idx]
    z_min, z_max = float(z.min()), float(z.max())
    extent = z_max - z_min
    if extent <= 0:
        return None

    n_bins = 256
    counts, bin_edges = np.histogram(z, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Local maxima above a noise floor.  Boundary bins use a one-sided test —
    # the floor slab is often exactly at z_min (bin 0).
    noise_floor = max(float(np.median(counts)) * 2.0, len(pts) / n_bins * 0.5)
    def _is_peak(i: int) -> bool:
        left_ok = i == 0 or counts[i] >= counts[i - 1]
        right_ok = i == n_bins - 1 or counts[i] >= counts[i + 1]
        return left_ok and right_ok and counts[i] > noise_floor
    peak_idx = [i for i in range(n_bins) if _is_peak(i)]
    if len(peak_idx) < 2:
        return None

    # Strongest peak = one slab (usually the floor).  Partner = the strongest
    # peak at least 25% of the extent away (the other slab).  This tolerates
    # furniture bands and mid-height clutter peaks in between.
    peak_idx.sort(key=lambda i: -counts[i])
    a = peak_idx[0]
    min_sep = 0.25 * extent
    partner = None
    for i in peak_idx[1:]:
        if abs(bin_centers[i] - bin_centers[a]) >= min_sep:
            partner = i
            break
    if partner is None:
        return None
    return float(abs(bin_centers[partner] - bin_centers[a]))


def _estimate_wall_thickness(
    pts: np.ndarray,
    axis_idx: int,
    z_low: float,
    z_high: float,
    raster_res: float = 0.02,
) -> float | None:
    """Rough median wall thickness (raw units) from a mid-height slice.

    Rasterizes the band ``[z_low, z_high]`` to a coarse occupancy grid and
    uses a distance transform: for wall pixels, twice the median distance to
    free space approximates the wall thickness.  Cheap (single slice, coarse
    grid) — good enough to distinguish 0.1 m walls from 0.1 ft (0.03 m) ones.
    """
    try:
        import cv2
    except ImportError:
        return None

    band_mask = (pts[:, axis_idx] >= z_low) & (pts[:, axis_idx] <= z_high)
    band = pts[band_mask]
    if len(band) < 500:
        return None
    plan_axes = [i for i in range(3) if i != axis_idx]
    xy = band[:, plan_axes]
    mins = xy.min(axis=0)
    span = xy.max(axis=0) - mins
    w = int(np.ceil(span[0] / raster_res)) + 1
    h = int(np.ceil(span[1] / raster_res)) + 1
    if w * h > 8_000_000 or w < 4 or h < 4:
        return None

    occ = np.zeros((h, w), dtype=np.uint8)
    cols = np.clip(((xy[:, 0] - mins[0]) / raster_res).astype(np.int32), 0, w - 1)
    rows = np.clip(((xy[:, 1] - mins[1]) / raster_res).astype(np.int32), 0, h - 1)
    occ[rows, cols] = 255

    # Distance (in px) from each wall pixel to the nearest free pixel; a wall
    # of thickness T has interior distances up to T/2.  The 90th percentile
    # over wall pixels approximates half the typical thickness.
    dist = cv2.distanceTransform(occ, distanceType=cv2.DIST_L2, maskSize=3)
    wall_dist = dist[occ > 0]
    if len(wall_dist) == 0:
        return None
    half_px = float(np.percentile(wall_dist, 90))
    return 2.0 * half_px * raster_res


def detect_units(pts: np.ndarray, axis_idx: int = 2) -> UnitDetection:
    """Detect whether a point cloud is in metres or feet.

    Primary heuristic: floor-to-ceiling peak gap.  Secondary: median wall
    thickness from a mid-height slice.  Raises :class:`UnitDetectionError`
    when vertical structure IS present but implausible under both units.

    Huge clouds are randomly subsampled to :data:`_UNIT_DETECT_MAX_POINTS`
    first — the histogram is stable well below that, and scanning every
    point of a 100 M+ cloud freezes the job with no progress updates.
    """
    if len(pts) > _UNIT_DETECT_MAX_POINTS:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(pts), size=_UNIT_DETECT_MAX_POINTS, replace=False)
        pts = pts[idx]

    gap = _find_floor_ceiling_gap(pts, axis_idx)

    if gap is not None:
        plausible_m = _CEILING_MIN_M <= gap <= _CEILING_MAX_M
        plausible_ft = _CEILING_MIN_M <= gap * _FT_TO_M <= _CEILING_MAX_M
        if plausible_m and not plausible_ft:
            return UnitDetection("m", 1.0, "ceiling_gap", gap, None)
        if plausible_ft and not plausible_m:
            return UnitDetection("ft", _FT_TO_M, "ceiling_gap", gap, None)

        # Gap found but fits neither unit (e.g. 5.8 raw units) — try wall
        # thickness as a tie-break before failing.
        z = pts[:, axis_idx]
        mid = float(np.median(z))
        band = 0.15 * gap
        thickness = _estimate_wall_thickness(pts, axis_idx, mid - band, mid + band)
        if thickness is not None and thickness > 0:
            t_m_ok = _WALL_THICKNESS_MIN_M <= thickness <= _WALL_THICKNESS_MAX_M
            t_ft_ok = (
                _WALL_THICKNESS_MIN_M <= thickness * _FT_TO_M <= _WALL_THICKNESS_MAX_M
            )
            if t_m_ok and not t_ft_ok:
                return UnitDetection("m", 1.0, "wall_thickness", gap, thickness)
            if t_ft_ok and not t_m_ok:
                return UnitDetection("ft", _FT_TO_M, "wall_thickness", gap, thickness)

        raise UnitDetectionError(
            f"Cannot determine scan units: floor-to-ceiling distance is "
            f"{gap:.2f} raw units — implausible as metres "
            f"({_CEILING_MIN_M}–{_CEILING_MAX_M} m) and as feet "
            f"({gap * _FT_TO_M:.2f} m after conversion). "
            f"Re-export the scan in metres or feet, or verify the vertical axis."
        )

    # No floor/ceiling structure found (outdoor scan, single plane, sparse
    # cloud).  Assume metres but tell the caller detection didn't run.
    return UnitDetection("m", 1.0, "assumed_metres", None, None)


def normalize_units_to_metres(
    pcd: o3d.geometry.PointCloud,
    axis_idx: int = 2,
) -> tuple[o3d.geometry.PointCloud, UnitDetection]:
    """Detect the unit of ``pcd`` and scale it to metres in place.

    Returns ``(pcd, detection)``.  Raises :class:`UnitDetectionError` when
    the unit cannot be determined (see :func:`detect_units`).
    """
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return pcd, UnitDetection("m", 1.0, "assumed_metres", None, None)
    detection = detect_units(pts, axis_idx)
    if detection.scale_to_m != 1.0:
        pcd.points = o3d.utility.Vector3dVector(pts * detection.scale_to_m)
    return pcd, detection


def write_decimated_ply(
    pcd: o3d.geometry.PointCloud,
    output_path: Path,
    target_points: int = 200_000,
) -> Path:
    """Downsample to at most target_points, write float32 binary PLY for browser viewer.

    Open3D always writes property double internally; Three.js PLYLoader works better
    with float32, so we write the binary PLY manually using numpy/struct.
    """
    n = len(pcd.points)
    if n > target_points:
        voxel_size = _estimate_voxel_for_target(pcd, target_points)
        pcd = pcd.voxel_down_sample(voxel_size)

    pts = np.asarray(pcd.points, dtype=np.float32)
    has_colors = pcd.has_colors()
    if has_colors:
        cols = np.clip(np.asarray(pcd.colors, dtype=np.float32), 0.0, 1.0)

    n_pts = len(pts)
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        "comment Created by write_decimated_ply",
        f"element vertex {n_pts}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_colors:
        header_lines += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
    header_lines.append("end_header")
    header = "\n".join(header_lines) + "\n"

    with open(output_path, "wb") as f:
        f.write(header.encode("ascii"))
        if has_colors:
            rgb_u8 = (cols * 255).astype(np.uint8)
            # Interleave xyz (float32 × 3) + rgb (uint8 × 3) per vertex.
            vertex_block = np.zeros(
                n_pts,
                dtype=[
                    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                    ("r", "u1"),  ("g", "u1"),  ("b", "u1"),
                ],
            )
            vertex_block["x"] = pts[:, 0]
            vertex_block["y"] = pts[:, 1]
            vertex_block["z"] = pts[:, 2]
            vertex_block["r"] = rgb_u8[:, 0]
            vertex_block["g"] = rgb_u8[:, 1]
            vertex_block["b"] = rgb_u8[:, 2]
            f.write(vertex_block.tobytes())
        else:
            f.write(pts.astype("<f4").tobytes())

    return output_path


# ---------------------------------------------------------------------------
# LAZ / LAS helpers
# ---------------------------------------------------------------------------

def _log_las_header(las: object) -> None:
    """LAZ-8: emit one line of header info for coordinate-frame debugging.

    Tells you immediately if a scan is in UTM (X≈500 000), geographic
    (X≈-117.5), or a local origin (X≈0) — critical for diagnosing why rooms
    appear at the wrong place in the viewer.
    """
    try:
        h = las.header
        print(
            f"  LAS {h.version}: {h.point_count:,} pts | "
            f"scale={h.x_scale:.8g} | "
            f"X:{h.mins[0]:.2f}→{h.maxs[0]:.2f} "
            f"Y:{h.mins[1]:.2f}→{h.maxs[1]:.2f} "
            f"Z:{h.mins[2]:.2f}→{h.maxs[2]:.2f}"
        )
    except Exception:
        pass


def _build_classification_mask(cls: np.ndarray, n_total: int) -> np.ndarray:
    """LAZ-1/2: percentage-based classification mask that handles interior scans.

    ASPRS class 6 ("Building") is the correct filter for exterior aerial/mobile
    surveys.  Interior terrestrial scanners (Leica RTC360, FARO Focus, NavVis
    VLX, Matterport Pro3) classify almost all points as class 0 (Never
    Classified) or class 1 (Unclassified).  The old absolute count threshold
    of 500 caused silent data loss when a handful of spurious class-6 artefacts
    existed while millions of real interior points were class 0.

    Thresholds:
      ≥10 % class 6 → exterior survey, keep class 6 only
       2–10% class 6 → partially classified, keep class 6 + unclassified
      < 2%  class 6 → interior scan, keep all non-noise points
    """
    noise_mask = ~np.isin(cls, list(_NOISE_CLASSES))
    pct_class6 = int((cls == 6).sum()) / max(n_total, 1)

    if pct_class6 >= 0.10:
        # Clearly an exterior/aerial survey with building classification.
        return (cls == 6) & noise_mask
    elif pct_class6 >= 0.02:
        # Some classification exists — keep building + unclassified points.
        return ((cls == 6) | (cls == 1) | (cls == 0)) & noise_mask
    else:
        # Interior scan with no meaningful class 6 — keep all non-noise points.
        return noise_mask


def _normalize_rgb(
    r_raw: np.ndarray,
    g_raw: np.ndarray,
    b_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LAZ-4: auto-detect 8-bit vs 16-bit RGB to prevent near-black colors.

    LAS spec says RGB is uint16 (0–65535), but Leica RTC360, NavVis VLX, and
    many other scanners store 8-bit values (0–255) in the uint16 container.
    Dividing 8-bit data by 65535 produces colors ≈ 0.004 — essentially black.
    """
    max_val = max(float(r_raw.max()), float(g_raw.max()), float(b_raw.max()))
    if max_val <= 1.0:
        # Already normalized (some E57 exports write floats in [0, 1]).
        return r_raw.astype(np.float64), g_raw.astype(np.float64), b_raw.astype(np.float64)
    scale = 255.0 if max_val <= 255.0 else 65535.0
    return r_raw / scale, g_raw / scale, b_raw / scale


def _normalize_intensity(intens: np.ndarray) -> np.ndarray:
    """LAZ-3: robust percentile normalization to prevent outlier-collapsed colors.

    Dividing by max() collapses the whole dynamic range when one retroreflector
    hit or scanner startup pulse produces an intensity of 65535.  Clipping to
    the 2nd–98th percentile eliminates those outliers before normalization.
    """
    if intens.max() <= intens.min():
        return np.zeros_like(intens, dtype=np.float64)
    p2 = np.percentile(intens, 2)
    p98 = np.percentile(intens, 98)
    return np.clip((intens - p2) / max(float(p98 - p2), 1.0), 0.0, 1.0)


def _apply_intensity_colors(pcd: o3d.geometry.PointCloud, intens: np.ndarray) -> None:
    """Map intensity to a blue-to-cyan-to-white gradient and assign to PCD."""
    intens_norm = _normalize_intensity(intens)
    r_ch = intens_norm * 0.40
    g_ch = intens_norm * 0.85
    b_ch = np.clip(0.55 + intens_norm * 0.45, 0.0, 1.0)
    pcd.colors = o3d.utility.Vector3dVector(np.vstack([r_ch, g_ch, b_ch]).T)


# ---------------------------------------------------------------------------
# LAS / LAZ loaders
# ---------------------------------------------------------------------------

def _load_las(
    path: Path,
    on_progress: Optional[ProgressCallback] = None,
    load_colors: bool = True,
    on_phase: Optional[PhaseCallback] = None,
) -> o3d.geometry.PointCloud:
    """Load a LAS/LAZ file.  Routes to chunked streaming for files > 500 MB."""
    import laspy

    file_size = os.path.getsize(path)
    if file_size > _CHUNK_THRESHOLD_BYTES:
        print(f"  Large scan ({file_size / 1e6:.0f} MB) — using chunked streaming")
        return _load_las_chunked(
            path, on_progress=on_progress, load_colors=load_colors,
            on_phase=on_phase,
        )

    las = laspy.read(str(path))

    # LAZ-8: log header for coordinate-frame debugging.
    _log_las_header(las)

    n_total = len(las.x)
    mask = np.ones(n_total, dtype=bool)
    try:
        cls = np.asarray(las.classification, dtype=np.int32)
        mask = _build_classification_mask(cls, n_total)
    except Exception:
        pass

    pts = np.vstack([
        np.asarray(las.x, dtype=np.float64)[mask],
        np.asarray(las.y, dtype=np.float64)[mask],
        np.asarray(las.z, dtype=np.float64)[mask],
    ]).T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # Colors: prefer true RGB; fall back to intensity; fall back to no colors.
    # Skipped entirely for geometry-only callers (vectorize never uses them).
    color_set = False
    if load_colors and hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
        try:
            r_raw = np.asarray(las.red,   dtype=np.float64)[mask]
            g_raw = np.asarray(las.green, dtype=np.float64)[mask]
            b_raw = np.asarray(las.blue,  dtype=np.float64)[mask]
            r, g, b = _normalize_rgb(r_raw, g_raw, b_raw)
            pcd.colors = o3d.utility.Vector3dVector(np.vstack([r, g, b]).T)
            color_set = True
        except Exception:
            pass

    if load_colors and not color_set:
        try:
            intens = np.asarray(las.intensity, dtype=np.float64)[mask]
            _apply_intensity_colors(pcd, intens)
        except Exception:
            pass

    return pcd


def _grow_buffer(arr: np.ndarray, new_capacity: int) -> np.ndarray:
    """Grow a preallocated fill buffer to ``new_capacity`` rows, copying the
    existing contents.  Only used when the LAS header undercounts."""
    shape = (new_capacity,) if arr.ndim == 1 else (new_capacity, arr.shape[1])
    out = np.empty(shape, dtype=arr.dtype)
    out[: len(arr)] = arr
    return out


def _load_las_chunked(
    path: Path,
    chunk_points: int = 1_000_000,
    on_progress: Optional[ProgressCallback] = None,
    load_colors: bool = True,
    on_phase: Optional[PhaseCallback] = None,
) -> o3d.geometry.PointCloud:
    """LAZ-5: stream-read large LAS/LAZ files in 1M-point chunks.

    laspy.read() loads the complete file before returning.  For a 2 GB LAZ file
    this means 2 GB sits in RAM alongside the Open3D objects being built from
    it.  Chunked streaming decompresses one ~1 M-point chunk at a time.

    Memory discipline (Phase 4.5 — a 680 MB LAS previously peaked at ~22 GB
    RSS and thrashed a 32 GB machine):

    - ``xyz`` is PREALLOCATED from ``reader.header.point_count`` and filled
      per chunk.  The old list-of-chunks + ``np.vstack`` pattern briefly held
      TWO full copies of the coordinates (24 B/point each).  Headers can lie:
      a short read is trimmed at the end; an undercount grows the buffer.
    - Colors are kept as raw ``uint16`` (6 B/point vs 24 B/point float64)
      until AFTER the classification mask is applied, and only when
      ``load_colors`` is True.  Geometry-only callers (the vectorize
      pipeline) skip color accumulation entirely.

    The classification mask is FILE-GLOBAL: chunks accumulate raw points plus
    their classification codes, and :func:`_build_classification_mask` is
    applied once over the whole file at the end.  Deciding the regime per
    chunk was a correctness bug — an aerial survey whose class-6 percentage
    varies across chunks would keep "class 6 only" in one chunk and "all
    non-noise" in the next, silently mixing filtering policies within one
    scan (and disagreeing with the unchunked loader).

    ``on_progress(points_read, total_points)`` fires after every chunk —
    a 689 MB scan takes minutes to decompress, and without these callbacks
    the job looks hung.  ``total_points`` comes from the LAS header; if the
    header count is missing/zero, the running count is passed as both args.

    ``on_phase(name)`` fires once per post-read assembly phase ("classify",
    "assemble", "colors") — the window after the last chunk is the most
    memory-hungry part of the load and previously emitted nothing, making
    honest assembly work indistinguishable from a hang.
    """
    import laspy

    def _phase(name: str) -> None:
        if on_phase is not None:
            on_phase(name)

    has_rgb: bool | None = None if load_colors else False
    has_intensity: bool | None = None if load_colors else False

    write_pos = 0
    with laspy.open(str(path)) as reader:
        try:
            total_points = int(reader.header.point_count)
        except Exception:
            total_points = 0

        # Preallocate fill buffers from the header count (no list + vstack).
        capacity = max(total_points, chunk_points)
        xyz = np.empty((capacity, 3), dtype=np.float64)
        cls_buf: np.ndarray | None = np.empty(capacity, dtype=np.uint8)
        colors_buf: np.ndarray | None = None   # uint16 (N, 3), lazy-allocated
        intens_buf: np.ndarray | None = None   # uint16 (N,),  lazy-allocated

        for chunk in reader.chunk_iterator(chunk_points):
            n = len(chunk.x)
            end = write_pos + n
            if end > capacity:
                # Header undercounted — grow every live buffer (rare).
                capacity = max(end, capacity + chunk_points)
                xyz = _grow_buffer(xyz, capacity)
                if cls_buf is not None:
                    cls_buf = _grow_buffer(cls_buf, capacity)
                if colors_buf is not None:
                    colors_buf = _grow_buffer(colors_buf, capacity)
                if intens_buf is not None:
                    intens_buf = _grow_buffer(intens_buf, capacity)

            xyz[write_pos:end, 0] = chunk.x
            xyz[write_pos:end, 1] = chunk.y
            xyz[write_pos:end, 2] = chunk.z

            if cls_buf is not None:
                try:
                    cls_buf[write_pos:end] = np.asarray(
                        chunk.classification, dtype=np.uint8,
                    )
                except Exception:
                    cls_buf = None

            # Auto-detect color/intensity fields from the first chunk.
            if has_rgb is None:
                has_rgb = (
                    hasattr(chunk, "red")
                    and hasattr(chunk, "green")
                    and hasattr(chunk, "blue")
                )
            if has_intensity is None:
                has_intensity = hasattr(chunk, "intensity")

            if has_rgb:
                try:
                    if colors_buf is None:
                        colors_buf = np.empty((capacity, 3), dtype=np.uint16)
                    colors_buf[write_pos:end, 0] = np.asarray(chunk.red,   dtype=np.uint16)
                    colors_buf[write_pos:end, 1] = np.asarray(chunk.green, dtype=np.uint16)
                    colors_buf[write_pos:end, 2] = np.asarray(chunk.blue,  dtype=np.uint16)
                except Exception:
                    has_rgb = False
                    colors_buf = None  # discard partial accumulation

            if not has_rgb and has_intensity:
                try:
                    if intens_buf is None:
                        intens_buf = np.empty(capacity, dtype=np.uint16)
                    intens_buf[write_pos:end] = np.asarray(
                        chunk.intensity, dtype=np.uint16,
                    )
                except Exception:
                    has_intensity = False
                    intens_buf = None

            write_pos = end
            if on_progress is not None:
                on_progress(write_pos, total_points or write_pos)

    if write_pos == 0:
        raise ValueError(f"No points read from {path}")

    # Trim short reads.  These are views — no copy; the (rare) capacity slack
    # is released once the masked copies below replace the last references.
    xyz = xyz[:write_pos]

    # File-global classification mask — same decision the unchunked loader
    # would make on the complete file.  uint8 comparisons are identical to
    # the unchunked loader's int32 ones; skipping the cast avoids a 4× copy.
    _phase("classify")
    if cls_buf is not None:
        mask = _build_classification_mask(cls_buf[:write_pos], write_pos)
        cls_buf = None
        keep_all = bool(mask.all())
    else:
        mask = None
        keep_all = True

    _phase("assemble")
    pcd = o3d.geometry.PointCloud()
    # When the mask keeps everything, skip the fancy-index copy — Open3D
    # copies into its own storage anyway.
    pcd.points = o3d.utility.Vector3dVector(xyz if keep_all else xyz[mask])
    del xyz

    if colors_buf is not None:
        _phase("colors")
        try:
            raw = colors_buf[:write_pos]
            if not keep_all:
                raw = raw[mask]
            colors_buf = None
            # uint16 / float divides to float64 — normalization happens on
            # the (smaller) post-mask array only.
            r, g, b = _normalize_rgb(raw[:, 0], raw[:, 1], raw[:, 2])
            pcd.colors = o3d.utility.Vector3dVector(np.vstack([r, g, b]).T)
        except Exception:
            pass
    elif intens_buf is not None:
        _phase("colors")
        try:
            intens = intens_buf[:write_pos]
            if not keep_all:
                intens = intens[mask]
            intens_buf = None
            _apply_intensity_colors(pcd, intens.astype(np.float64))
        except Exception:
            pass

    return pcd


# ---------------------------------------------------------------------------
# E57 loader
# ---------------------------------------------------------------------------

def _load_e57(path: Path) -> o3d.geometry.PointCloud:
    """LAZ-7: read ALL scan positions from an E57, not just index 0.

    Leica Cyclone Register 360 and similar software export every registered
    scan position into a single E57 container.  The previous implementation
    called read_scan(0) — discarding ~90 % of the data for a 10-position file.
    """
    import pye57

    e57 = pye57.E57(str(path))
    print(f"  E57: {e57.scan_count} scan(s) in {path.name}")

    all_pts: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []  # raw values, scaled after accumulation
    all_intens: list[np.ndarray] = []
    has_color = False
    has_intens = False

    for i in range(e57.scan_count):
        try:
            data = e57.read_scan(
                i, intensity=True, colors=True, ignore_missing_fields=True
            )
            pts = np.vstack([
                data["cartesianX"],
                data["cartesianY"],
                data["cartesianZ"],
            ]).T.astype(np.float64)
            all_pts.append(pts)

            # Colors take priority over intensity.
            if (
                "colorRed" in data
                and "colorGreen" in data
                and "colorBlue" in data
            ):
                r_raw = np.asarray(data["colorRed"],   dtype=np.float64)
                g_raw = np.asarray(data["colorGreen"], dtype=np.float64)
                b_raw = np.asarray(data["colorBlue"],  dtype=np.float64)
                all_colors.append(np.vstack([r_raw, g_raw, b_raw]).T)
                has_color = True
            elif "intensity" in data:
                all_intens.append(np.asarray(data["intensity"], dtype=np.float64))
                has_intens = True

        except Exception:
            continue  # skip malformed individual scan positions

    if not all_pts:
        raise ValueError(f"No readable scans in E57 file: {path}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(all_pts))

    if has_color and all_colors:
        try:
            all_raw = np.vstack(all_colors)
            r, g, b = _normalize_rgb(all_raw[:, 0], all_raw[:, 1], all_raw[:, 2])
            pcd.colors = o3d.utility.Vector3dVector(np.vstack([r, g, b]).T)
        except Exception:
            pass
    elif has_intens and all_intens:
        try:
            _apply_intensity_colors(pcd, np.concatenate(all_intens))
        except Exception:
            pass

    return pcd


# ---------------------------------------------------------------------------
# Voxel estimation
# ---------------------------------------------------------------------------

def _estimate_voxel_for_target(pcd: o3d.geometry.PointCloud, target: int) -> float:
    """LAZ-6: analytical voxel estimate — replaces 20-call binary search.

    The old binary search called voxel_down_sample() 20 times.  For a 7 M-point
    cloud each call costs ~0.5 s → 10 s of pure overhead at the export step.

    An analytical estimate derived from point-cloud density is accurate within
    ~20 % and requires at most one verification downsample call.

    Derivation:
      density (pts/m³) = N / volume
      We want ~target voxels → each voxel covers volume/target m³
      voxel_size = (volume / target)^(1/3)
    """
    n = len(pcd.points)
    if n <= target:
        return 0.001  # no downsampling needed

    pts = np.asarray(pcd.points)
    bbox = pts.max(axis=0) - pts.min(axis=0)
    volume = float(np.prod(np.maximum(bbox, 1e-6)))
    voxel_estimate = (volume / max(target, 1)) ** (1.0 / 3.0)
    voxel_estimate = max(voxel_estimate, 0.001)

    # One verification call — adjust if the estimate left too many points.
    ds = pcd.voxel_down_sample(voxel_estimate)
    ratio = len(ds.points) / max(target, 1)
    if ratio > 1.3:
        # Estimate was slightly too small; scale up proportionally.
        voxel_estimate *= ratio ** (1.0 / 3.0)

    return voxel_estimate
