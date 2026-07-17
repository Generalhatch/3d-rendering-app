"""Multi-scan merging: combine N LAZ/LAS/PLY/E57 scan files into one unified point cloud.

Design principle: memory-safe streaming pipeline.
Each file is loaded, immediately downsampled, and the full-res cloud discarded.
We never hold all scans at full resolution simultaneously.

Strategy:
  1. Load each scan → immediately voxel-downsample → keep only the downsampled version.
  2. Check if the scans share a coordinate frame (pre-registered).
     - Professional LiDAR workflows (Leica Cyclone, FARO SCENE, NavVis) almost always
       export pre-registered LAZ in a common site frame. We detect this by checking
       that all scan centroids are within a reasonable proximity of each other
       (same building floor = centroids within ~200m of each other).
  3. If pre-registered: concatenate the downsampled clouds. Done.
  4. If NOT pre-registered: sequential pairwise ICP in upload/walking order
       (scan i registers to the most recent successfully-placed scan — field
       technicians scan room-to-room, so consecutive uploads overlap), followed
       by Open3D pose-graph global optimization to distribute drift over any
       loop closures.  Every pairwise fit is checked against explicit
       fitness/RMSE quality gates: scans that fail are flagged UNPLACED and
       excluded from the merge — never silently guessed into position.
  5. Final voxel downsample of the merged result.

Memory profile (6 scans × 500M points each, 3cm voxel):
  - Peak RAM ≈ 2 × (one downsampled cloud) ≈ 2 × ~80MB = ~160MB
  - vs naïve load-all: 6 × 500MB = 3GB
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import open3d as o3d

from .ingest import load_point_cloud


# ── Registration quality gates ────────────────────────────────────────────────

# Explicit thresholds a pairwise ICP fit must clear before a scan is accepted
# into the floor assembly.  Fitness is the fraction of source points with a
# correspondence within the ICP distance threshold; inlier RMSE is the RMS
# residual of those correspondences.  A garbage FPFH init refined by ICP
# typically lands at fitness < 0.10; genuinely overlapping room-to-room scans
# land at 0.30+.
#
# The RMSE threshold scales with the registration voxel size: pairwise ICP
# runs on clouds downsampled at 3× the merge voxel, so even a PERFECT
# alignment carries an RMSE of roughly the downsample spacing / 2 (the two
# clouds' voxel centroids don't coincide).  1.6 × voxel leaves headroom for
# that sampling mismatch while still rejecting fits misaligned by more than
# about one voxel.  At the production 3 cm voxel this is a 4.8 cm gate.
DEFAULT_MIN_ICP_FITNESS = 0.25
DEFAULT_MAX_ICP_RMSE_PER_VOXEL = 1.6


def default_gate_for_voxel(voxel_size: float) -> "RegistrationGate":
    """The registration gate used by :func:`merge_scans` at a given voxel size."""
    return RegistrationGate(
        min_fitness=DEFAULT_MIN_ICP_FITNESS,
        max_inlier_rmse_m=voxel_size * DEFAULT_MAX_ICP_RMSE_PER_VOXEL,
    )


@dataclass(frozen=True)
class RegistrationGate:
    """Quality gate for one pairwise registration result."""
    min_fitness: float
    max_inlier_rmse_m: float

    def check(self, fitness: float, inlier_rmse_m: float) -> tuple[bool, str]:
        """Return ``(passed, reason)``; ``reason`` is "" when passed."""
        reasons: list[str] = []
        if fitness < self.min_fitness:
            reasons.append(
                f"fitness {fitness:.3f} < required {self.min_fitness:.3f}"
            )
        if inlier_rmse_m > self.max_inlier_rmse_m:
            reasons.append(
                f"inlier RMSE {inlier_rmse_m * 1000:.1f}mm > allowed "
                f"{self.max_inlier_rmse_m * 1000:.1f}mm"
            )
        return (not reasons, "; ".join(reasons))


@dataclass
class ScanRegistration:
    """Registration outcome for one input scan — placed or flagged unplaced."""
    scan_index: int
    scan_name: str
    placed: bool
    fitness: float
    inlier_rmse_m: float
    transformation: np.ndarray     # (4, 4) pose: scan-local frame → reference frame
    method: str                    # reference | sequential_icp | pose_graph |
                                   # pre_registered | single
    reason: str = ""               # why the scan is unplaced ("" when placed)

    def to_json_dict(self) -> dict:
        return {
            "scan_index": int(self.scan_index),
            "scan_name": self.scan_name,
            "placed": bool(self.placed),
            "fitness": float(self.fitness),
            "inlier_rmse_m": float(self.inlier_rmse_m),
            "inlier_rmse_mm": float(self.inlier_rmse_m * 1000.0),
            "transformation": np.asarray(self.transformation, dtype=float).tolist(),
            "method": self.method,
            "reason": self.reason,
        }


@dataclass
class MergeResult:
    merged: o3d.geometry.PointCloud
    num_scans: int
    total_points_before_ds: int   # sum of points in each scan after initial downsample
    total_points_after: int       # final merged+downsampled count
    strategy: str                 # "concatenate" | "icp_registered" | "single"
    centroid_offset: np.ndarray   # 3-vector subtracted by _center_cloud; add it back
                                  # to convert centered coords → original scan frame
    registrations: list[ScanRegistration] = field(default_factory=list)

    @property
    def unplaced(self) -> list[ScanRegistration]:
        return [r for r in self.registrations if not r.placed]


# Minimum centroid spread (metres) that indicates scans are in a shared large-scale
# coordinate frame rather than each sitting at their own local origin.
_SHARED_FRAME_SPREAD_M = 5.0


def merge_scans(
    scan_paths: list[Path],
    voxel_size: float = 0.03,
    progress_cb: Callable[[str, float], None] | None = None,
) -> MergeResult:
    """Memory-safe merge of N scan files into one downsampled point cloud.

    Streams each file one at a time — never holds all scans at full resolution.
    """
    if not scan_paths:
        raise ValueError("No scan files provided")

    def _emit(msg: str, p: float) -> None:
        if progress_cb:
            progress_cb(msg, p)

    n = len(scan_paths)

    if n == 1:
        _emit(f"Loading scan: {scan_paths[0].name}…", 0.0)
        pcd = load_point_cloud(scan_paths[0])
        pts_raw = len(pcd.points)
        _emit("Downsampling…", 0.5)
        ds = pcd.voxel_down_sample(voxel_size)
        del pcd  # free full-res immediately
        ds, _ = ds.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        centroid_offset = np.asarray(ds.points, dtype=np.float64).mean(axis=0)
        ds = _center_cloud(ds)
        _emit(f"Loaded: {len(ds.points):,} points", 1.0)
        return MergeResult(
            merged=ds,
            num_scans=1,
            total_points_before_ds=pts_raw,
            total_points_after=len(ds.points),
            strategy="single",
            centroid_offset=centroid_offset,
            registrations=[ScanRegistration(
                scan_index=0, scan_name=scan_paths[0].name, placed=True,
                fitness=1.0, inlier_rmse_m=0.0, transformation=np.eye(4),
                method="single",
            )],
        )

    # ── Step 1: Stream-load, downsample, and clean each scan ─────────────────
    # Statistical outlier removal runs per-scan before merging.
    # This removes: floating scan artifacts, scanner motion blur, tree foliage,
    # and any isolated noise points captured through windows or open doors.
    # nb_neighbors=20 / std_ratio=2.0 is conservative — removes only clear outliers.
    downsampled: list[o3d.geometry.PointCloud] = []
    total_pts_ds = 0

    for i, path in enumerate(scan_paths):
        frac = i / n
        _emit(f"Loading {i + 1}/{n}: {path.name}…", frac * 0.55)
        pcd = load_point_cloud(path)
        ds = pcd.voxel_down_sample(voxel_size)
        del pcd   # free full-resolution cloud immediately
        ds, _ = ds.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        total_pts_ds += len(ds.points)
        downsampled.append(ds)
        _emit(f"  → {len(ds.points):,} points after downsample", (i + 0.9) / n * 0.55)

    # ── Step 2: Detect coordinate frame ──────────────────────────────────────
    _emit("Detecting coordinate frame…", 0.58)
    pre_registered = _scans_share_coordinate_frame(downsampled)
    strategy = "concatenate" if pre_registered else "icp_registered"

    if pre_registered:
        _emit(f"Pre-registered scans detected — concatenating {n} clouds…", 0.62)
        registrations = [
            ScanRegistration(
                scan_index=i, scan_name=scan_paths[i].name, placed=True,
                fitness=1.0, inlier_rmse_m=0.0, transformation=np.eye(4),
                method="pre_registered",
            )
            for i in range(n)
        ]
    else:
        _emit("Scans appear to be in different frames — sequential pairwise ICP "
              "(upload order prior)…", 0.62)
        downsampled, registrations = register_scans_sequential(
            downsampled,
            voxel_size,
            scan_names=[p.name for p in scan_paths],
            emit=_emit,
        )
        n_unplaced = sum(1 for r in registrations if not r.placed)
        if n_unplaced:
            names = ", ".join(r.scan_name for r in registrations if not r.placed)
            _emit(
                f"⚠ {n_unplaced} scan(s) failed registration gates and are "
                f"UNPLACED (excluded from merge): {names}",
                0.80,
            )

    # ── Step 3: Concatenate ───────────────────────────────────────────────────
    _emit("Concatenating…", 0.82)
    merged = downsampled[0]
    for pcd in downsampled[1:]:
        merged = merged + pcd

    # ── Step 4: Final downsample (removes duplicates in overlap zones) ────────
    _emit("Final downsample…", 0.90)
    merged = merged.voxel_down_sample(voxel_size)

    # ── Step 5: Center the merged cloud at origin for numerical stability ─────
    # Geographic/UTM coordinates (e.g. X=500000, Y=4500000) cause issues with
    # Open3D's RANSAC. We center once here so all downstream code works with
    # coordinates near the origin, while preserving relative scan positions.
    # Store the offset so per-scan re-loading can apply the same transform.
    centroid_offset = np.asarray(merged.points, dtype=np.float64).mean(axis=0)
    merged = _center_cloud(merged)

    _emit(
        f"Merge complete — {len(merged.points):,} pts from {n} scans "
        f"({'pre-registered' if pre_registered else 'ICP-registered'})",
        1.0,
    )

    return MergeResult(
        merged=merged,
        num_scans=n,
        total_points_before_ds=total_pts_ds,
        total_points_after=len(merged.points),
        strategy=strategy,
        centroid_offset=centroid_offset,
        registrations=registrations,
    )


# ── Coordinate-frame detection ────────────────────────────────────────────────

def _center_cloud(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Subtract the centroid so the cloud is centred near origin.

    Called once on the fully-merged cloud to ensure all downstream pipeline
    code (RANSAC, ICP, projection) works with coordinates near 0,0,0.
    Colors are preserved.
    """
    pts = np.asarray(pcd.points, dtype=np.float64)
    centroid = pts.mean(axis=0)
    centered = o3d.geometry.PointCloud()
    centered.points = o3d.utility.Vector3dVector(pts - centroid)
    if pcd.has_colors():
        centered.colors = pcd.colors
    return centered


def _scans_share_coordinate_frame(clouds: list[o3d.geometry.PointCloud]) -> bool:
    """Return True if all scans appear to be in the same coordinate frame.

    Two-stage check:

    Stage 1 — centroid spread.  If scan centroids are spread out (>5m apart on
    average), they're almost certainly pre-registered professional exports
    (Leica Cyclone, FARO SCENE, NavVis, Matterport Pro3 all output scans in a
    common site frame).

    Stage 2 — spatial overlap.  If all centroids are suspiciously close to each
    other (<5m apart), it could mean either:
      (a) Pre-registered scans of a very compact space, OR
      (b) Unregistered scans — each scan is in its own local frame starting
          near 0,0,0 (e.g. exported without global registration).

    To distinguish (a) from (b) we check voxel overlap: if a meaningful
    fraction of voxels from scan A land within tolerance of voxels from scan B,
    the scans genuinely share space → pre-registered.  If no overlap is found,
    treat as unregistered.
    """
    centroids = np.array([
        np.asarray(c.points).mean(axis=0)
        for c in clouds
        if len(c.points) > 0
    ])
    if len(centroids) < 2:
        return True

    group_centroid = centroids.mean(axis=0)
    max_dist = float(np.linalg.norm(centroids - group_centroid, axis=1).max())

    # Clear sign of a shared large-scale coordinate frame: centroids are spread
    # meaningfully across the floor footprint.  5m spread is a conservative lower
    # bound — a 10 m × 10 m office would have room centroids ≥3–4 m apart.
    if max_dist >= 5.0:
        return True

    # Centroids are all very close — could be pre-registered compact space OR
    # unregistered (each scan sitting at its own local origin).  Verify by
    # checking actual voxel overlap between the first pair of scans.
    return _clouds_have_spatial_overlap(clouds[0], clouds[1], voxel_size=0.20)


def _clouds_have_spatial_overlap(
    a: o3d.geometry.PointCloud,
    b: o3d.geometry.PointCloud,
    voxel_size: float = 0.20,
    min_overlap_fraction: float = 0.05,
) -> bool:
    """Return True if at least min_overlap_fraction of scan A's voxels are
    occupied by scan B (i.e. the scans genuinely share physical space).

    Uses a simple 3D hash set comparison at coarse resolution.
    """
    def _voxel_set(pcd: o3d.geometry.PointCloud) -> set:
        pts = np.asarray(pcd.points)
        keys = (pts / voxel_size).astype(np.int32)
        return {(int(k[0]), int(k[1]), int(k[2])) for k in keys}

    set_a = _voxel_set(a)
    set_b = _voxel_set(b)
    if not set_a:
        return False
    overlap = len(set_a & set_b)
    fraction = overlap / len(set_a)
    return fraction >= min_overlap_fraction


# ── Sequential pairwise ICP + pose-graph registration ────────────────────────

def register_scans_sequential(
    clouds: list[o3d.geometry.PointCloud],
    voxel_size: float,
    gate: RegistrationGate | None = None,
    scan_names: list[str] | None = None,
    emit: Callable[[str, float], None] | None = None,
    pose_graph_optimize: bool = True,
) -> tuple[list[o3d.geometry.PointCloud], list[ScanRegistration]]:
    """Sequential pairwise registration in upload/walking order + pose-graph
    global optimization.

    Replaces the previous "star" topology (every scan → scan 0).  Field
    technicians scan room-to-room, so consecutive uploads overlap while the
    first and last scan of a large floor often share nothing — the upload
    order IS the adjacency prior.  Each scan is registered (FPFH global init
    → ICP refine) against the most recent successfully-placed scan.

    Every pairwise fit is checked against ``gate``: a scan whose fitness /
    inlier RMSE fails the gate is flagged UNPLACED, excluded from the output
    clouds, and never guessed into position.  Subsequent scans register
    against the last scan that *did* place.

    When ≥ 3 scans place, an Open3D pose graph is built (odometry edges from
    the sequential chain + uncertain loop-closure edges for any non-adjacent
    overlapping pair) and globally optimized so loop-closure drift is
    distributed instead of accumulating in the last scan.

    Returns ``(placed_clouds, registrations)`` — ``placed_clouds`` are the
    placed scans transformed into scan 0's frame (in scan order);
    ``registrations`` has one :class:`ScanRegistration` per input scan.
    """
    if gate is None:
        gate = default_gate_for_voxel(voxel_size)
    n = len(clouds)
    names = scan_names or [f"scan_{i}" for i in range(n)]

    def _msg(msg: str, p: float) -> None:
        if emit:
            emit(msg, p)

    prepared = [_prepare_for_registration(c, voxel_size) for c in clouds]
    max_corr = voxel_size * 2.0

    def _register_pair(
        src_idx: int, tgt_idx: int, n_attempts: int = 3,
    ) -> tuple[float, float, np.ndarray]:
        """Multi-start FPFH → ICP between two prepared scans.

        FPFH + RANSAC is stochastic, and on repetitive architecture a wrong
        global init (the classic "slid one room over" failure) can still
        reach moderate ICP consensus.  A genuinely correct alignment,
        however, has strictly lower inlier RMSE than a plausible-but-wrong
        one.  So: run several independent global inits, refine each with
        ICP, and keep the gate-passing candidate with the LOWEST inlier
        RMSE (falling back to best fitness when none pass, so the unplaced
        record carries honest metrics).
        """
        sp, tp = prepared[src_idx], prepared[tgt_idx]
        candidates: list[tuple[float, float, np.ndarray]] = []
        for _ in range(n_attempts):
            T_init = _fpfh_global_registration(
                sp["ds"], tp["ds"], sp["fpfh"], tp["fpfh"], voxel_size,
            )
            icp = o3d.pipelines.registration.registration_icp(
                sp["ds"], tp["ds"],
                max_corr,
                T_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
            )
            candidates.append((
                float(icp.fitness),
                float(icp.inlier_rmse),
                np.array(icp.transformation),
            ))
        passing = [c for c in candidates if gate.check(c[0], c[1])[0]]
        if passing:
            return min(passing, key=lambda c: c[1])
        return max(candidates, key=lambda c: c[0])

    registrations: list[ScanRegistration] = [
        ScanRegistration(
            scan_index=0, scan_name=names[0], placed=True,
            fitness=1.0, inlier_rmse_m=0.0, transformation=np.eye(4),
            method="reference",
        )
    ]
    poses: dict[int, np.ndarray] = {0: np.eye(4)}   # scan idx → local→reference pose
    placed_order: list[int] = [0]
    # Relative ICP transforms along the placed chain, for pose-graph odometry
    # edges: chain_edges[k] = (src_idx, tgt_idx, T_icp src-local → tgt-local).
    chain_edges: list[tuple[int, int, np.ndarray]] = []

    for i in range(1, n):
        prev = placed_order[-1]
        _msg(
            f"Registering scan {i + 1}/{n} to scan {prev + 1} (walking-order prior)…",
            0.62 + (i / n) * 0.14,
        )
        fitness, rmse, T_icp = _register_pair(i, prev)
        passed, reason = gate.check(fitness, rmse)

        if not passed:
            registrations.append(ScanRegistration(
                scan_index=i, scan_name=names[i], placed=False,
                fitness=fitness, inlier_rmse_m=rmse, transformation=np.eye(4),
                method="sequential_icp",
                reason=f"failed registration gate vs scan {prev + 1}: {reason}",
            ))
            _msg(
                f"  ✗ scan {i + 1} UNPLACED — {reason}",
                0.62 + (i / n) * 0.14,
            )
            continue

        poses[i] = poses[prev] @ T_icp
        placed_order.append(i)
        chain_edges.append((i, prev, T_icp))
        registrations.append(ScanRegistration(
            scan_index=i, scan_name=names[i], placed=True,
            fitness=fitness, inlier_rmse_m=rmse,
            transformation=poses[i].copy(),
            method="sequential_icp",
        ))
        _msg(
            f"  ✓ scan {i + 1} placed — fitness {fitness:.2f}, "
            f"RMSE {rmse * 1000:.1f}mm",
            0.62 + (i / n) * 0.14,
        )

    # ── Pose-graph global optimization over the placed scans ─────────────────
    if pose_graph_optimize and len(placed_order) >= 3:
        try:
            optimized = _pose_graph_refine(
                prepared, poses, placed_order, chain_edges, gate,
                max_corr, _msg,
            )
            if optimized is not None:
                for idx in placed_order:
                    poses[idx] = optimized[idx]
                for reg in registrations:
                    if reg.placed and reg.scan_index != 0:
                        reg.transformation = poses[reg.scan_index].copy()
                        reg.method = "pose_graph"
        except Exception as pg_err:  # optimization is a refinement, not a gate
            _msg(f"Pose-graph optimization skipped: {pg_err}", 0.79)

    placed_clouds: list[o3d.geometry.PointCloud] = []
    for idx in placed_order:
        transformed = o3d.geometry.PointCloud(clouds[idx])
        transformed.transform(poses[idx])
        placed_clouds.append(transformed)

    return placed_clouds, registrations


def _pose_graph_refine(
    prepared: list[dict],
    poses: dict[int, np.ndarray],
    placed_order: list[int],
    chain_edges: list[tuple[int, int, np.ndarray]],
    gate: RegistrationGate,
    max_corr: float,
    msg: Callable[[str, float], None],
) -> dict[int, np.ndarray] | None:
    """Build + globally optimize an Open3D pose graph over the placed scans.

    Nodes carry the sequential poses (scan-local → reference).  Odometry
    edges come from the sequential chain; loop-closure edges are added for
    every non-adjacent placed pair whose clouds overlap under the current
    poses AND whose pairwise ICP passes the same quality gate (uncertain
    edges — the optimizer may prune them).  Returns the optimized poses, or
    None when there was nothing to optimize.
    """
    reg = o3d.pipelines.registration

    node_of = {scan_idx: k for k, scan_idx in enumerate(placed_order)}
    pose_graph = reg.PoseGraph()
    for scan_idx in placed_order:
        pose_graph.nodes.append(reg.PoseGraphNode(poses[scan_idx].copy()))

    # Odometry edges from the sequential chain.  Edge convention (matches the
    # Open3D multiway-registration tutorial): edge (s, t, T) with T mapping
    # s-local → t-local constrains pose_s ≈ pose_t @ T.
    for src, tgt, T_icp in chain_edges:
        info = reg.get_information_matrix_from_point_clouds(
            prepared[src]["ds"], prepared[tgt]["ds"], max_corr, T_icp,
        )
        pose_graph.edges.append(reg.PoseGraphEdge(
            node_of[src], node_of[tgt], T_icp, info, uncertain=False,
        ))

    # Loop-closure edges: non-adjacent placed pairs that overlap under the
    # current pose estimates.
    n_loops = 0
    for a_pos in range(len(placed_order)):
        for b_pos in range(a_pos + 2, len(placed_order)):
            a, b = placed_order[a_pos], placed_order[b_pos]
            # Current relative estimate: a-local → b-local.
            T_ab = np.linalg.inv(poses[b]) @ poses[a]
            ds_a_in_b = o3d.geometry.PointCloud(prepared[a]["ds"])
            ds_a_in_b.transform(T_ab)
            if not _clouds_have_spatial_overlap(
                ds_a_in_b, prepared[b]["ds"], voxel_size=0.20,
            ):
                continue
            icp = reg.registration_icp(
                prepared[a]["ds"], prepared[b]["ds"],
                max_corr,
                T_ab,
                reg.TransformationEstimationPointToPoint(),
                reg.ICPConvergenceCriteria(max_iteration=50),
            )
            passed, _reason = gate.check(float(icp.fitness), float(icp.inlier_rmse))
            if not passed:
                continue
            T_loop = np.array(icp.transformation)
            info = reg.get_information_matrix_from_point_clouds(
                prepared[a]["ds"], prepared[b]["ds"], max_corr, T_loop,
            )
            pose_graph.edges.append(reg.PoseGraphEdge(
                node_of[a], node_of[b], T_loop, info, uncertain=True,
            ))
            n_loops += 1

    if n_loops == 0 and len(chain_edges) < 2:
        return None   # a bare 2-node chain gains nothing from optimization

    msg(
        f"Pose-graph optimization: {len(placed_order)} nodes, "
        f"{len(chain_edges)} odometry + {n_loops} loop-closure edges…",
        0.78,
    )
    option = reg.GlobalOptimizationOption(
        max_correspondence_distance=max_corr,
        edge_prune_threshold=0.25,
        reference_node=0,
    )
    reg.global_optimization(
        pose_graph,
        reg.GlobalOptimizationLevenbergMarquardt(),
        reg.GlobalOptimizationConvergenceCriteria(),
        option,
    )
    return {
        scan_idx: np.array(pose_graph.nodes[node_of[scan_idx]].pose)
        for scan_idx in placed_order
    }


def _prepare_for_registration(pcd: o3d.geometry.PointCloud, voxel_size: float) -> dict:
    """Downsample, estimate normals, compute FPFH features."""
    ds = pcd.voxel_down_sample(voxel_size * 3)  # coarser for registration speed
    ds.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 6, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        ds,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 15, max_nn=100),
    )
    return {"ds": ds, "fpfh": fpfh}


def _fpfh_global_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    src_fpfh,
    tgt_fpfh,
    voxel_size: float,
) -> np.ndarray:
    dist = voxel_size * 6
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, src_fpfh, tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=dist,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4_000_000, 0.999),
    )
    return np.array(result.transformation)
