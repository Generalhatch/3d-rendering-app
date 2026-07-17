"""Job-level sheet orchestration: floor_geometry.json → sheet.svg + sheet.pdf.

Shared by both pipelines and the HTTP routes so a sheet re-render (after the
operator edits suite labels, title-block metadata, or enters a manual grid)
never re-runs any extraction — it reloads the persisted FloorGeometry and
re-draws.

Overrides file (``sheet_overrides.json`` in the job's result dir)::

    {
      "labels":      {"<room_id>": "Suite 301", ...},
      "meta":        {"building_name": "...", "address": "...",
                      "floor_name": "...", "north_angle_deg": 0.0},
      "manual_grid": {"rotation_deg": 0.0,
                      "u_offsets_m": [0.0, 6.0, ...],
                      "v_offsets_m": [0.0, 5.0, ...]},
      "style":       "stevenson_minimal",
      "door_swings": {"<opening_id>": {"hinge": "start", "side": "left"}}
    }

All keys optional.  A manual grid, when present, REPLACES the fitted grid
(the operator knows the real structural grid better than the detector).
``door_swings`` merges per opening — the operator flips one door at
time; an empty dict value clears that opening's override.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from ..geometry.model import FloorGeometry
from .drafting import dominant_wall_axis_rad
from .grid import ColumnGrid, bbox_aligned_grid, fit_column_grid, manual_grid
from .pdf import svg_to_pdf
from .render import SheetMetadata, SheetParams, SheetRender, floor_bounds, render_sheet

#: Sheet styles the overrides API accepts.
VALID_STYLES = ("full", "stevenson_minimal")
DEFAULT_STYLE = "stevenson_minimal"

OVERRIDES_FILENAME = "sheet_overrides.json"
SHEET_SVG_FILENAME = "sheet.svg"
SHEET_PDF_FILENAME = "sheet.pdf"


def load_overrides(result_dir: str | Path) -> dict:
    path = Path(result_dir) / OVERRIDES_FILENAME
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_overrides(result_dir: str | Path, overrides: dict) -> Path:
    """Merge-persist operator overrides (labels and door_swings merge
    per-key; meta, manual_grid and style replace wholesale)."""
    current = load_overrides(result_dir)
    if "labels" in overrides:
        merged = dict(current.get("labels", {}))
        merged.update(overrides["labels"] or {})
        # Empty-string override = clear back to the pipeline label.
        current["labels"] = {k: v for k, v in merged.items() if v}
    if "door_swings" in overrides:
        merged = dict(current.get("door_swings", {}))
        merged.update(overrides["door_swings"] or {})
        # Empty dict override = clear back to the heuristic swing.
        current["door_swings"] = {k: v for k, v in merged.items() if v}
    for key in ("meta", "manual_grid", "style"):
        if key in overrides:
            if overrides[key]:
                current[key] = overrides[key]
            else:
                current.pop(key, None)
    path = Path(result_dir) / OVERRIDES_FILENAME
    path.write_text(json.dumps(current, indent=2))
    return path


def _humanize_scan_stem(filename: str) -> str:
    stem = Path(filename).stem
    cleaned = re.sub(r"[_\-]+", " ", stem).strip()
    return cleaned.title() if cleaned else stem


def _derive_sheet_meta(floor: FloorGeometry, scan_filename: str = "") -> dict:
    """Best-effort title-block fields from geometry + job upload name."""
    building = (floor.building_name or "").strip()
    if not building and scan_filename:
        building = _humanize_scan_stem(scan_filename)
    floor_name = (floor.floor_name or "").strip()
    address = building
    return {
        "building_name": building,
        "address": address,
        "floor_name": floor_name,
        "north_angle_deg": 0.0,
    }


def _load_scan_filename(result_dir: Path) -> str:
    try:
        from ..storage import load_vectorize_job
        job = load_vectorize_job(result_dir.name)
        return job.get("scan_filename", "") or ""
    except (KeyError, Exception):
        return ""


def ensure_initial_overrides(result_dir: str | Path, floor: FloorGeometry) -> dict:
    """Seed ``sheet_overrides.json`` on first render so deliverables aren't blank."""
    result_dir = Path(result_dir)
    path = result_dir / OVERRIDES_FILENAME
    if path.exists():
        overrides = load_overrides(result_dir)
        changed = False
        if not overrides.get("style"):
            overrides["style"] = DEFAULT_STYLE
            changed = True
        if not overrides.get("meta"):
            overrides["meta"] = _derive_sheet_meta(floor, _load_scan_filename(result_dir))
            changed = True
        if changed:
            path.write_text(json.dumps(overrides, indent=2))
        return overrides

    overrides = {
        "style": DEFAULT_STYLE,
        "meta": _derive_sheet_meta(floor, _load_scan_filename(result_dir)),
    }
    path.write_text(json.dumps(overrides, indent=2))
    return overrides


def resolve_grid(
    floor: FloorGeometry, overrides: dict,
) -> ColumnGrid:
    """Manual grid wins; otherwise fit from columns; bbox fallback last."""
    mg = overrides.get("manual_grid")
    if mg:
        return manual_grid(
            u_offsets_m=mg.get("u_offsets_m", []),
            v_offsets_m=mg.get("v_offsets_m", []),
            rotation_deg=float(mg.get("rotation_deg", 0.0)),
        )
    fitted = fit_column_grid(floor.columns)
    if fitted is not None:
        return fitted
    rotation = dominant_wall_axis_rad(floor.walls)
    bounds = floor_bounds(floor, rotation)
    return bbox_aligned_grid(bounds, rotation)


def resolve_metadata(floor: FloorGeometry, overrides: dict) -> SheetMetadata:
    m = overrides.get("meta", {}) or {}
    return SheetMetadata(
        building_name=m.get("building_name", "") or floor.building_name,
        address=m.get("address", ""),
        floor_name=m.get("floor_name", "") or floor.floor_name,
        north_angle_deg=float(m.get("north_angle_deg", 0.0)),
    )


def resolve_params(overrides: dict) -> SheetParams:
    """Sheet page params from overrides (currently just the style)."""
    style = overrides.get("style") or DEFAULT_STYLE
    if style not in VALID_STYLES:
        style = "full"
    return SheetParams(style=style)


def render_job_sheet(
    result_dir: str | Path,
    floor: FloorGeometry | None = None,
    write_pdf: bool = True,
) -> SheetRender:
    """Render (or re-render) the sheet for one job's result directory.

    Loads ``floor_geometry.json`` when ``floor`` isn't passed, applies any
    persisted operator overrides, writes ``sheet.svg`` (+ ``sheet.pdf``),
    and returns the render.
    """
    result_dir = Path(result_dir)
    if floor is None:
        geo_path = result_dir / "floor_geometry.json"
        if not geo_path.exists():
            raise FileNotFoundError(
                f"floor_geometry.json not found in {result_dir} — "
                "the job has no canonical geometry to render"
            )
        floor = FloorGeometry.from_json_dict(json.loads(geo_path.read_text()))

    overrides = ensure_initial_overrides(result_dir, floor)
    render = render_sheet(
        floor,
        meta=resolve_metadata(floor, overrides),
        grid=resolve_grid(floor, overrides),
        params=resolve_params(overrides),
        label_overrides=overrides.get("labels"),
        door_swings=overrides.get("door_swings"),
    )
    (result_dir / SHEET_SVG_FILENAME).write_text(render.svg)
    if write_pdf:
        svg_to_pdf(render.svg, result_dir / SHEET_PDF_FILENAME)
    return render
