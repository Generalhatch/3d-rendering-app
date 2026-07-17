"""Rebuild FloorGeometry + BOMA reports from operator-edited wall segments.

When auto topology finds zero rooms, the pipeline still emits walls/DXF but
skips ``floor_geometry.json`` and the measurement report.  Operators close
gaps in the editor, then call :func:`generate_measurement_from_segments`
(via ``POST /generate-measurement``) to re-run pairing → topology →
FloorGeometry → BOMA/REBNY/Gross + sheet — without re-ingesting the scan.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from ..models.vectorize_job import EditableSegment
from . import topology as topology_mod
from . import walls as walls_mod


class RoomsNotClosedError(ValueError):
    """Wall network still does not enclose any rooms."""

    def __init__(
        self,
        message: str,
        *,
        n_walls: int = 0,
        n_junctions: int = 0,
        n_edges: int = 0,
        dangling_endpoints: list[dict] | None = None,
    ) -> None:
        super().__init__(message)
        self.n_walls = n_walls
        self.n_junctions = n_junctions
        self.n_edges = n_edges
        self.dangling_endpoints = list(dangling_endpoints or [])


@dataclass
class _SegOpening:
    seg: np.ndarray
    width_m: float


@dataclass
class _SegColumn:
    corners: np.ndarray
    centre_m: tuple[float, float]
    size_m: tuple[float, float]
    is_round: bool = False


@dataclass
class GenerateMeasurementResult:
    rooms_detected: int
    has_floor_geometry: bool
    has_measurement_report: bool
    has_sheet: bool
    warnings: list[dict] = field(default_factory=list)
    n_walls: int = 0
    n_junctions: int = 0


def _segments_to_array(
    segments: Sequence[EditableSegment] | Sequence[dict[str, Any]],
    layer: str,
) -> np.ndarray:
    rows: list[list[list[float]]] = []
    for s in segments:
        if isinstance(s, dict):
            seg_layer = s.get("layer") or "walls"
            x1, y1, x2, y2 = s["x1"], s["y1"], s["x2"], s["y2"]
        else:
            seg_layer = s.layer or "walls"
            x1, y1, x2, y2 = s.x1, s.y1, s.x2, s.y2
        if seg_layer != layer:
            continue
        rows.append([[float(x1), float(y1)], [float(x2), float(y2)]])
    if not rows:
        return np.zeros((0, 2, 2), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def _load_segments_payload(result_dir: Path) -> list[dict[str, Any]]:
    path = result_dir / "segments.json"
    if not path.exists():
        raise FileNotFoundError(f"No segments.json in {result_dir}")
    payload = json.loads(path.read_text())
    return list(payload.get("segments") or [])


def _load_envelope_xy(result_dir: Path) -> Optional[np.ndarray]:
    path = result_dir / "envelope.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        pts = np.asarray(data.get("polygon_xy"), dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
            return None
        return pts
    except Exception:
        return None


def _openings_from_segments(segs: np.ndarray) -> list[_SegOpening]:
    out: list[_SegOpening] = []
    for seg in segs:
        width = float(np.linalg.norm(seg[1] - seg[0]))
        out.append(_SegOpening(seg=seg.copy(), width_m=width))
    return out


def _columns_from_segments(segs: np.ndarray) -> list[_SegColumn]:
    """Approximate column footprints from editor column segments.

    The editor stores columns as short line segments; for FloorGeometry we
    need a centre + polygon.  Use the segment midpoint and a small square
    sized to the segment length (clamped).
    """
    out: list[_SegColumn] = []
    for seg in segs:
        centre = (seg[0] + seg[1]) / 2.0
        length = float(np.linalg.norm(seg[1] - seg[0]))
        half = max(0.10, min(0.60, length / 2.0 if length > 1e-6 else 0.15))
        cx, cy = float(centre[0]), float(centre[1])
        corners = np.array([
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ], dtype=float)
        out.append(_SegColumn(
            corners=corners,
            centre_m=(cx, cy),
            size_m=(2.0 * half, 2.0 * half),
            is_round=False,
        ))
    return out


def generate_measurement_from_segments(
    result_dir: Path,
    job_id: str,
    *,
    segments: Sequence[EditableSegment] | Sequence[dict[str, Any]] | None = None,
    snapping_distance_m: float = 0.85,
    render_sheet: bool = True,
) -> GenerateMeasurementResult:
    """Pair walls, infer rooms, write floor geometry + BOMA + optional sheet.

    Parameters
    ----------
    result_dir
        Job results directory (``segments.json``, ``envelope.json``, …).
    job_id
        Used as ``floor_id`` and for DXF annotation context.
    segments
        Optional override; when ``None``, reads ``segments.json``.
    snapping_distance_m
        Topology axis-extension distance (same knob as the pipeline).
    render_sheet
        When True, also write ``sheet.svg`` / ``sheet.pdf`` and add DXF dims.

    Raises
    ------
    FileNotFoundError
        Missing ``segments.json``.
    RoomsNotClosedError
        Topology found zero rooms — operator must close more walls.
    """
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    if segments is None:
        seg_list: Sequence[Any] = _load_segments_payload(result_dir)
    else:
        seg_list = segments

    wall_segs = _segments_to_array(seg_list, "walls")
    opening_segs = _segments_to_array(seg_list, "openings")
    column_segs = _segments_to_array(seg_list, "columns")

    if len(wall_segs) < 3:
        raise RoomsNotClosedError(
            f"Need at least 3 wall segments to form a room "
            f"(found {len(wall_segs)}).  Draw or restore walls in the editor, "
            f"then try Generate BOMA again.",
            n_walls=len(wall_segs),
        )

    # Pairing can fail on sparse edited networks — fall back to unpaired
    # centerlines so topology still gets a chance.
    try:
        wall_pairing = walls_mod.pair_walls(wall_segs)
    except Exception:
        wall_pairing = walls_mod.WallPairingResult(
            walls=[],
            face_segments=np.zeros((0, 2, 2), dtype=np.float64),
            centerline_segments=wall_segs,
            median_thickness_m=0.10,
            n_paired=0,
            n_unpaired=len(wall_segs),
        )

    topology_params = topology_mod.TopologyParams(
        snap_tol_m=0.0,
        snapping_distance_m=float(snapping_distance_m),
        wall_thickness_median_m=float(wall_pairing.median_thickness_m),
    )
    # Load envelope before topology so Phase 2 continuity can project
    # near-hull dangling ends (same path as the scan pipeline).
    envelope_xy = _load_envelope_xy(result_dir)
    topology_result = topology_mod.build_topology(
        wall_pairing.centerline_segments,
        params=topology_params,
        envelope_xy=envelope_xy,
    )

    # Persist finished wall axes back into segments.json so the editor
    # shows Cloud2BIM-style continuous walls (not detector fragments).
    finished = topology_result.finished_wall_segments
    if finished is not None and len(finished) > 0:
        wall_pairing = walls_mod.WallPairingResult(
            walls=wall_pairing.walls,
            face_segments=walls_mod.faces_from_centerlines(
                finished, wall_pairing.median_thickness_m,
            ),
            centerline_segments=finished,
            median_thickness_m=wall_pairing.median_thickness_m,
            n_paired=wall_pairing.n_paired,
            n_unpaired=wall_pairing.n_unpaired,
            n_orphans_dropped=getattr(wall_pairing, "n_orphans_dropped", 0),
        )
        # Rewrite walls (+ faces) in segments.json while keeping other layers.
        try:
            from ..vectorize import edits as edits_mod
            payload = edits_mod.load_segments(result_dir)
            kept = [
                s for s in (payload.get("segments") or [])
                if s.get("layer") not in ("walls", "walls_faces", "rooms")
            ]
            import uuid as _uuid
            for seg in finished:
                kept.append({
                    "id": _uuid.uuid4().hex[:12],
                    "layer": "walls",
                    "x1": float(seg[0, 0]), "y1": float(seg[0, 1]),
                    "x2": float(seg[1, 0]), "y2": float(seg[1, 1]),
                })
            for seg in wall_pairing.face_segments:
                kept.append({
                    "id": _uuid.uuid4().hex[:12],
                    "layer": "walls_faces",
                    "x1": float(seg[0, 0]), "y1": float(seg[0, 1]),
                    "x2": float(seg[1, 0]), "y2": float(seg[1, 1]),
                })
            room_segs = topology_mod.rooms_to_segments(topology_result.rooms)
            for seg in room_segs:
                kept.append({
                    "id": _uuid.uuid4().hex[:12],
                    "layer": "rooms",
                    "x1": float(seg[0, 0]), "y1": float(seg[0, 1]),
                    "x2": float(seg[1, 0]), "y2": float(seg[1, 1]),
                })
            payload["segments"] = kept
            (result_dir / "segments.json").write_text(
                json.dumps(payload, indent=2)
            )
        except Exception:
            pass

    n_junctions = len(topology_result.junctions)
    n_edges = len(topology_result.edges)
    dangling = topology_mod.dangling_endpoints(topology_result)
    topology_mod.write_dangling_endpoints(result_dir, dangling)

    if not topology_result.rooms:
        raise RoomsNotClosedError(
            f"Wall network still does not enclose any rooms "
            f"({len(wall_segs)} walls, {n_junctions} junctions, "
            f"{n_edges} edges).  Close open wall ends in the editor "
            f"(draw or drag endpoints until corners meet), then click "
            f"Generate BOMA again.",
            n_walls=len(wall_segs),
            n_junctions=n_junctions,
            n_edges=n_edges,
            dangling_endpoints=dangling,
        )

    # Rooms closed — clear stale dangling markers from a prior failed attempt.
    topology_mod.write_dangling_endpoints(result_dir, [])

    from ..geometry.adapters import floor_geometry_from_vectorize
    from ..geometry.validation import validate_floor_closure
    from ..measurement import get_ruleset, list_rulesets
    from ..measurement.report import (
        build_report,
        write_json_report,
        write_pdf_report,
    )

    warnings: list[dict] = []
    openings = _openings_from_segments(opening_segs)
    columns = _columns_from_segments(column_segs)

    floor_geo = floor_geometry_from_vectorize(
        topology_result,
        wall_pairing,
        envelope_xy=envelope_xy,
        openings=openings,
        columns=columns,
        floor_id=job_id,
    )
    # Mark source so audits can tell auto vs operator-driven measurement.
    floor_geo.source = "vectorize.editor"

    (result_dir / "floor_geometry.json").write_text(
        json.dumps(floor_geo.to_json_dict(), indent=2)
    )

    floor_validation = validate_floor_closure(floor_geo)
    (result_dir / "floor_validation.json").write_text(
        json.dumps(floor_validation.to_json_dict(), indent=2)
    )
    if not floor_validation.passed:
        n_gaps = len(floor_validation.unscanned_gaps)
        gap_m2 = sum(g.area_m2 for g in floor_validation.unscanned_gaps)
        warnings.append({
            "code": "floor_closure_failed",
            "stage": "measure",
            "message": (
                f"Floor-closure checks failed: "
                f"{len(floor_validation.warnings)} warning(s)"
                + (f", {n_gaps} unscanned gap(s) totalling {gap_m2:.1f} m²"
                   if n_gaps else "")
                + " — see floor_validation.json"
            ),
        })

    measurements = [get_ruleset(name).measure(floor_geo) for name in list_rulesets()]
    report = build_report(floor_geo, measurements)
    write_json_report(report, result_dir / "measurement_report.json")
    write_pdf_report(report, result_dir / "measurement_report.pdf")

    sheet_ok = False
    if render_sheet:
        try:
            from ..sheet.dimensions import add_dimensions_to_dxf
            from ..sheet.service import render_job_sheet
            render_job_sheet(result_dir, floor=floor_geo)
            dxf_path = result_dir / "vectorized.dxf"
            if dxf_path.exists():
                add_dimensions_to_dxf(dxf_path, floor_geo)
            sheet_ok = (result_dir / "sheet.svg").exists()
        except Exception as sheet_err:
            warnings.append({
                "code": "sheet_failed",
                "stage": "sheet",
                "message": f"Sheet rendering failed ({sheet_err})",
            })

    return GenerateMeasurementResult(
        rooms_detected=len(topology_result.rooms),
        has_floor_geometry=True,
        has_measurement_report=True,
        has_sheet=sheet_ok,
        warnings=warnings,
        n_walls=len(wall_segs),
        n_junctions=n_junctions,
    )
