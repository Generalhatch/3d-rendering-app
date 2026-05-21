"""Synthetic point cloud + ground truth for eval-harness regression tests.

Generates a tiny, deterministic 2-room building (5 m × 3 m, split by an
internal wall at x = 2.5) as both:

  - A point cloud (PLY file) with floor, ceiling, and wall surface points
    at 1 cm spacing — small enough to run in < 5 s on any machine.
  - A matching ground-truth JSON ready for the eval harness.

This is the CI fixture — it lets us run the full eval loop without a real
LiDAR scan, so regression detection runs on every PR without depending on
large binary assets.

Use:
    cd backend
    python -m app.eval.synthetic_fixture --out /tmp/eval_fixture
    python -m app.eval --scan /tmp/eval_fixture/synth.ply \\
                       --gt   /tmp/eval_fixture/synth_gt.json \\
                       --out  /tmp/eval_fixture/run
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

from .ground_truth import GroundTruth, save_ground_truth


def make_synthetic_building(
    out_dir: Path,
    floor_size_m: tuple[float, float] = (5.0, 3.0),
    ceiling_height_m: float = 2.8,
    wall_thickness_m: float = 0.15,
    point_spacing_m: float = 0.01,
) -> tuple[Path, Path]:
    """Create a tiny 2-room building point cloud and ground-truth JSON.

    Returns ``(scan_path, gt_path)``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lx, ly = floor_size_m
    cz = ceiling_height_m
    t = wall_thickness_m

    rng = np.random.default_rng(seed=42)

    def _wall_points(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
        """Generate points along a wall segment, full ceiling height.

        Adds ±2 cm perpendicular noise (realistic scanner roughness for a
        finished interior wall) so the wall reads as a multi-pixel band in
        the raster — matches what real LiDAR produces and survives the
        morphological OPEN preprocess step.
        """
        length = float(np.hypot(x1 - x0, y1 - y0))
        n_long = max(2, int(length / point_spacing_m))
        n_z = max(2, int(cz / point_spacing_m))
        ts = np.linspace(0.0, 1.0, n_long)
        zs = np.linspace(0.05, cz, n_z)
        xs = x0 + (x1 - x0) * ts
        ys = y0 + (y1 - y0) * ts
        xy = np.stack([xs, ys], axis=1)
        repeated = np.repeat(xy, n_z, axis=0)
        z_col = np.tile(zs, n_long)
        pts = np.column_stack([repeated, z_col])
        # ±2 cm Gaussian jitter perpendicular to the wall (1 std = 1 cm).
        # Real terrestrial LiDAR shows ~3-5 mm range noise on matte
        # surfaces; we double it to give the wall mask a robust 4-6 px
        # thickness in the raster, on the order of a real scan's
        # behaviour after the ceiling/density gating.
        jitter = rng.standard_normal((len(pts), 1)) * 0.010
        dx = (x1 - x0) / max(length, 1e-9)
        dy = (y1 - y0) / max(length, 1e-9)
        nx = -dy
        ny = dx
        pts[:, 0] += jitter[:, 0] * nx
        pts[:, 1] += jitter[:, 0] * ny
        return pts

    # Walls (with both faces, 15 cm apart).  Walk the outer rectangle and
    # the internal divider at x = lx / 2.
    points_chunks: list[np.ndarray] = []
    half_t = t / 2.0
    # Bottom wall (y = 0) — outer face at -half_t, inner face at +half_t.
    points_chunks.append(_wall_points(0, -half_t, lx, -half_t))
    points_chunks.append(_wall_points(0, half_t, lx, half_t))
    # Top wall (y = ly).
    points_chunks.append(_wall_points(0, ly - half_t, lx, ly - half_t))
    points_chunks.append(_wall_points(0, ly + half_t, lx, ly + half_t))
    # Left wall.
    points_chunks.append(_wall_points(-half_t, 0, -half_t, ly))
    points_chunks.append(_wall_points(half_t, 0, half_t, ly))
    # Right wall.
    points_chunks.append(_wall_points(lx - half_t, 0, lx - half_t, ly))
    points_chunks.append(_wall_points(lx + half_t, 0, lx + half_t, ly))
    # Internal divider at x = lx / 2.
    mid_x = lx / 2.0
    points_chunks.append(_wall_points(mid_x - half_t, half_t, mid_x - half_t, ly - half_t))
    points_chunks.append(_wall_points(mid_x + half_t, half_t, mid_x + half_t, ly - half_t))

    # Floor.
    n_x = max(2, int(lx / point_spacing_m))
    n_y = max(2, int(ly / point_spacing_m))
    xs = np.linspace(0, lx, n_x)
    ys = np.linspace(0, ly, n_y)
    xx, yy = np.meshgrid(xs, ys)
    floor = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    points_chunks.append(floor)

    # Ceiling.
    ceiling = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, cz)])
    points_chunks.append(ceiling)

    pts = np.concatenate(points_chunks, axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    scan_path = out_dir / "synth.ply"
    o3d.io.write_point_cloud(str(scan_path), pcd, write_ascii=False)

    # Ground truth: the wall centerlines (between the two faces of each wall).
    gt_walls = np.array([
        [[0, 0], [lx, 0]],          # bottom
        [[lx, 0], [lx, ly]],        # right
        [[lx, ly], [0, ly]],        # top
        [[0, ly], [0, 0]],          # left
        [[mid_x, 0], [mid_x, ly]],  # internal divider
    ], dtype=np.float64)

    gt = GroundTruth(
        scan=str(scan_path),
        walls=gt_walls,
        rooms_expected=2,
        columns_expected=0,
        openings_expected=0,
        tolerance_m=0.20,
        runtime_budget_s=30.0,
    )
    gt_path = out_dir / "synth_gt.json"
    save_ground_truth(gt, gt_path)

    return scan_path, gt_path


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the synthetic 2-room fixture for eval CI.",
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    scan_path, gt_path = make_synthetic_building(args.out)
    print(f"Wrote {scan_path}")
    print(f"Wrote {gt_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli())
