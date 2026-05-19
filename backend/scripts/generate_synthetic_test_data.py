"""Generate synthetic test data for pipeline unit tests.

Usage:
    python scripts/generate_synthetic_test_data.py

Outputs (into backend/tests/fixtures/):
  - synthetic_scan.ply    — point cloud sampled on axis-aligned room walls,
                            rotated by a known angle (45°) and translated
  - synthetic_plan.dxf    — rectangular room plan with a hallway

The pipeline should recover the inverse of the applied transform.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import open3d as o3d

FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)


# ── Synthetic plan: an L-shaped floor ─────────────────────────────────────────

PLAN_WALLS = [
    # Outer perimeter
    ((0, 0),  (20, 0)),
    ((20, 0), (20, 15)),
    ((20, 15),(10, 15)),
    ((10, 15),(10, 10)),
    ((10, 10),( 0, 10)),
    (( 0, 10),( 0,  0)),
    # Internal wall (hallway)
    (( 0,  5),(10,  5)),
]


def generate_plan_dxf(path: Path) -> None:
    import ezdxf
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    for (x1, y1), (x2, y2) in PLAN_WALLS:
        msp.add_line((x1, y1, 0), (x2, y2, 0))
    # Room labels
    labels = [
        ("Office-A", (5, 7.5)),
        ("Office-B", (5, 2.5)),
        ("Conf Room", (15, 7.5)),
        ("Hallway", (5, 11)),
    ]
    for text, (x, y) in labels:
        msp.add_text(text, dxfattribs={"insert": (x, y, 0), "height": 0.5})
    doc.saveas(str(path))
    print(f"  Wrote plan: {path}")


# ── Synthetic scan: sample points on the plan walls, add noise, apply transform ─

def generate_scan_ply(
    path: Path,
    rotation_deg: float = 45.0,
    translation: tuple[float, float] = (5.0, -3.0),
    noise_std: float = 0.01,
    points_per_meter: int = 200,
) -> np.ndarray:
    """Generate a PLY scan by sampling wall points, adding noise, and transforming.

    Returns the applied 4x4 transform (so tests can verify the pipeline recovers it).
    """
    pts: list[np.ndarray] = []

    for (x1, y1), (x2, y2) in PLAN_WALLS:
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        n = max(2, int(length * points_per_meter))
        # Sample along wall at multiple heights (z = 0 to 3m for full wall)
        for z in np.linspace(0.1, 2.9, 20):
            ts = np.linspace(0, 1, n)
            xs = x1 + (x2 - x1) * ts + np.random.normal(0, noise_std, n)
            ys = y1 + (y2 - y1) * ts + np.random.normal(0, noise_std, n)
            zs = np.full(n, z) + np.random.normal(0, noise_std / 2, n)
            pts.append(np.vstack([xs, ys, zs]).T)

    # Add a floor plane (z=0)
    floor_xs = np.random.uniform(0, 20, 5000)
    floor_ys = np.random.uniform(0, 15, 5000)
    floor_zs = np.random.normal(0, 0.005, 5000)
    pts.append(np.vstack([floor_xs, floor_ys, floor_zs]).T)

    # Add some furniture clutter (boxes at desk height ~0.7m)
    for cx, cy in [(5, 2), (15, 5), (3, 7)]:
        furniture = np.random.uniform(-0.5, 0.5, (300, 3))
        furniture[:, 2] = furniture[:, 2] * 0.3 + 0.8  # z = 0.7–0.9m
        furniture[:, 0] += cx
        furniture[:, 1] += cy
        pts.append(furniture)

    all_pts = np.vstack(pts)

    # Apply known transform
    angle_rad = np.deg2rad(rotation_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    t = np.array([translation[0], translation[1], 0])

    transformed = all_pts @ R.T + t

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(transformed)
    o3d.io.write_point_cloud(str(path), pcd)
    print(f"  Wrote scan: {path} ({len(transformed):,} points)")

    # Return the 4x4 transform
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def main() -> None:
    print("Generating synthetic test data…")
    plan_path = FIXTURES_DIR / "synthetic_plan.dxf"
    scan_path = FIXTURES_DIR / "synthetic_scan.ply"
    transform_path = FIXTURES_DIR / "ground_truth_transform.npy"

    generate_plan_dxf(plan_path)
    T = generate_scan_ply(scan_path, rotation_deg=45.0, translation=(5.0, -3.0))
    np.save(str(transform_path), T)
    print(f"  Wrote ground truth transform: {transform_path}")
    print(f"  Applied transform (rotation=45°, translation=(5, -3))")
    print("Done.")


if __name__ == "__main__":
    main()
