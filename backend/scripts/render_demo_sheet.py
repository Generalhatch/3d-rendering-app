"""Render a demo drafting-quality sheet from the reference-floor fixtures.

Adds synthetic openings, columns, penetrations, and a curved facade so
every Phase 3.5 feature shows on one page, renders the stevenson_minimal
style, and writes SVG + PDF to /tmp for visual comparison against the
DEMO41-Fullfloor reference.

Usage::

    cd backend && .venv/bin/python scripts/render_demo_sheet.py
    cd backend && .venv/bin/python scripts/render_demo_sheet.py --check

``--check`` prints a visual QA checklist (grid ticks, label size/color,
suite outline weight, door swings) for side-by-side comparison with the
600 5th Avenue / DEMO41 reference PDFs.
"""
from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.geometry.model import (          # noqa: E402
    BoundaryBasis, ColumnFeature, FloorGeometry, Opening, Penetration,
    Room, Wall, WallClass, WallFaceRef,
)
from app.sheet.grid import fit_column_grid   # noqa: E402
from app.sheet.pdf import svg_to_pdf         # noqa: E402
from app.sheet.render import (               # noqa: E402
    SUITE_OUTLINE_STROKE_MM,
    DOUBLE_WALL_STROKE_MM,
    SheetMetadata,
    SheetParams,
    render_sheet,
)


def build_demo_floor() -> FloorGeometry:
    ext_t, dem_t, part_t = 0.30, 0.20, 0.10

    def wall(wid, p, q, t, cls):
        return Wall(id=wid, centerline=np.array([p, q], float),
                    thickness_m=t, wall_class=cls)

    walls = [
        # Shell (30 × 18), with the right side chamfered 45° at the top.
        wall("w_bottom", (0, 0), (30, 0), ext_t, WallClass.EXTERIOR),
        wall("w_right", (30, 0), (30, 12), ext_t, WallClass.EXTERIOR),
        wall("w_chamfer", (30, 12), (24, 18), ext_t, WallClass.EXTERIOR),
        wall("w_top", (24, 18), (0, 18), ext_t, WallClass.EXTERIOR),
        wall("w_left", (0, 18), (0, 0), ext_t, WallClass.EXTERIOR),
        # Corridor walls (common corridor y ∈ [8, 10]).
        wall("w_corr_s", (0, 8), (30, 8), dem_t, WallClass.DEMISING),
        wall("w_corr_n", (0, 10), (26, 10), dem_t, WallClass.DEMISING),
        # Suite demising below the corridor.
        wall("w_dem_1", (10, 0), (10, 8), dem_t, WallClass.DEMISING),
        wall("w_dem_2", (20, 0), (20, 8), dem_t, WallClass.DEMISING),
        # Suite demising above the corridor.
        wall("w_dem_3", (12, 10), (12, 18), dem_t, WallClass.DEMISING),
        # Partitions inside suite 101.
        wall("w_p1", (5, 0), (5, 5), part_t, WallClass.PARTITION),
        wall("w_p2", (5, 5), (10, 5), part_t, WallClass.PARTITION),
    ]
    # Curved facade bump on suite 103's south wall: chords of a circle
    # (centre (25, 0), r = 3) from 180° to 0° in 15° steps, drawn INTO the
    # suite (a bowed storefront).
    arc_pts = []
    for k in range(13):
        th = math.radians(180.0 - 15.0 * k)
        arc_pts.append((25.0 + 3.0 * math.cos(th), 0.0 + 3.0 * math.sin(th)))
    for k in range(12):
        walls.append(wall(
            f"w_arc_{k}", arc_pts[k], arc_pts[k + 1], part_t,
            WallClass.PARTITION,
        ))

    def rect(x0, y0, x1, y1):
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)

    def refs(*ids):
        return [WallFaceRef(i) for i in ids]

    rooms = [
        Room(id="s101", boundary=rect(0, 0, 10, 8),
             wall_refs=refs("w_bottom", "w_dem_1", "w_corr_s", "w_left"),
             boundary_basis=BoundaryBasis.CENTERLINE,
             label="Suite 101", suite_id="101"),
        Room(id="s102", boundary=rect(10, 0, 20, 8),
             wall_refs=refs("w_bottom", "w_dem_2", "w_corr_s", "w_dem_1"),
             boundary_basis=BoundaryBasis.CENTERLINE,
             label="Suite 102", suite_id="102"),
        Room(id="s103", boundary=rect(20, 0, 30, 8),
             wall_refs=refs("w_bottom", "w_right", "w_corr_s", "w_dem_2"),
             boundary_basis=BoundaryBasis.CENTERLINE,
             label="Suite 103", suite_id="103"),
        Room(id="corr", boundary=rect(0, 8, 30, 10),
             wall_refs=refs("w_corr_s", None, "w_corr_n", None),
             boundary_basis=BoundaryBasis.CENTERLINE,
             label="Corridor", category="hallway", is_common=True),
        Room(id="s104", boundary=rect(0, 10, 12, 18),
             wall_refs=refs("w_corr_n", "w_dem_3", "w_top", "w_left"),
             boundary_basis=BoundaryBasis.CENTERLINE,
             label="Suite 104", suite_id="104"),
        Room(
            id="s105",
            boundary=np.array([
                [12, 10], [26, 10], [30, 10],
                [30, 12], [24, 18], [12, 18],
            ], float),
            wall_refs=refs("w_corr_n", None, "w_right", "w_chamfer",
                           "w_top", "w_dem_3"),
            boundary_basis=BoundaryBasis.CENTERLINE,
            label="Suite 105", suite_id="105",
        ),
    ]

    openings = [
        # Suite entry doors off the corridor (single leaf).
        Opening(id="d101", segment=np.array([[4.0, 8.0], [4.9, 8.0]]),
                width_m=0.9, kind="door", wall_id="w_corr_s"),
        Opening(id="d102", segment=np.array([[14.0, 8.0], [14.9, 8.0]]),
                width_m=0.9, kind="door", wall_id="w_corr_s"),
        Opening(id="d103", segment=np.array([[24.0, 8.0], [24.9, 8.0]]),
                width_m=0.9, kind="door", wall_id="w_corr_s"),
        Opening(id="d104", segment=np.array([[5.0, 10.0], [5.9, 10.0]]),
                width_m=0.9, kind="door", wall_id="w_corr_n"),
        Opening(id="d105", segment=np.array([[17.0, 10.0], [17.9, 10.0]]),
                width_m=0.9, kind="door", wall_id="w_corr_n"),
        # Main entrance: double door on the bottom wall.
        Opening(id="d_main", segment=np.array([[14.1, 0.0], [15.9, 0.0]]),
                width_m=1.8, kind="door", wall_id="w_bottom"),
        # Interior partition door.
        Opening(id="d_p1", segment=np.array([[5.0, 2.0], [5.0, 2.9]]),
                width_m=0.9, kind="door", wall_id="w_p1"),
        # Windows on the top wall.
        Opening(id="win1", segment=np.array([[2.0, 18.0], [6.0, 18.0]]),
                width_m=4.0, kind="window", wall_id="w_top"),
        Opening(id="win2", segment=np.array([[14.0, 18.0], [20.0, 18.0]]),
                width_m=6.0, kind="window", wall_id="w_top"),
    ]

    columns = [
        ColumnFeature(id=f"c{i}{j}", centre=(6.0 * i + 3.0, 6.0 * j + 3.0),
                      polygon=np.array([
                          [6.0 * i + 2.8, 6.0 * j + 2.8],
                          [6.0 * i + 3.2, 6.0 * j + 2.8],
                          [6.0 * i + 3.2, 6.0 * j + 3.2],
                          [6.0 * i + 2.8, 6.0 * j + 3.2],
                      ]))
        for i in range(5) for j in range(3)
        if not (i == 4 and j == 2)     # chamfer corner has no column
    ]

    penetrations = [
        Penetration(id="stair-1", kind="stair",
                    polygon=np.array([[1.0, 10.5], [4.0, 10.5],
                                      [4.0, 11.7], [1.0, 11.7]])),
        Penetration(id="elev-1", kind="elevator",
                    polygon=np.array([[26.5, 4.0], [28.5, 4.0],
                                      [28.5, 6.0], [26.5, 6.0]])),
        Penetration(id="oh-1", kind="overhang",
                    polygon=np.array([[12.0, -1.5], [18.0, -1.5],
                                      [18.0, 0.0], [12.0, 0.0]])),
    ]

    return FloorGeometry(
        floor_id="phase35-demo",
        walls=walls, rooms=rooms, openings=openings,
        columns=columns, penetrations=penetrations,
        envelope=rect(-0.15, -0.15, 30.15, 18.15),
        building_name="321 Kinzua Road",
        floor_name="Floor 1",
    )


def _checklist(svg: str) -> list[tuple[str, str, bool]]:
    """Return (category, detail, pass) tuples for visual QA."""
    root = ET.fromstring(svg)
    checks: list[tuple[str, str, bool]] = []

    grid = [e for e in root.iter() if e.get("id") == "grid"]
    ticks = [
        e for g in grid for e in g
        if (e.get("class") or "").startswith("grid-tick")
    ]
    checks.append((
        "Grid",
        f"{len(ticks)} edge ticks on four gutters (expect ≥ 8)",
        len(ticks) >= 8,
    ))
    tick_strokes = {
        line.get("stroke")
        for t in ticks for line in t if line.tag.endswith("line")
    }
    checks.append((
        "Grid",
        f"tick colour #aaaaaa (got {tick_strokes})",
        tick_strokes == {"#aaaaaa"},
    ))

    labels = [
        t for t in root.iter()
        if t.tag.endswith("text") and t.get("class") == "suite-label-name"
    ]
    label_sizes = {float(t.get("font-size", 0)) for t in labels}
    label_fills = {t.get("fill") for t in labels}
    checks.append((
        "Labels",
        f"7 mm light-gray suite labels (sizes={label_sizes}, fills={label_fills})",
        label_sizes == {7.0} and label_fills == {"#9a9a9a"},
    ))

    outlines = [
        p for g in root.iter() if g.get("id") == "suite-outlines"
        for p in g if p.tag.endswith("polygon")
    ]
    outline_weights = {float(p.get("stroke-width", 0)) for p in outlines}
    checks.append((
        "Suite outlines",
        f"heavy black {SUITE_OUTLINE_STROKE_MM} mm strokes (got {outline_weights})",
        outline_weights == {SUITE_OUTLINE_STROKE_MM},
    ))

    walls = [p for g in root.iter() if g.get("id") == "walls"
             for p in g if p.tag.endswith("path")]
    wall_weights = {float(p.get("stroke-width", 0)) for p in walls}
    checks.append((
        "Walls",
        f"thin double-line {DOUBLE_WALL_STROKE_MM} mm (got {wall_weights})",
        wall_weights == {DOUBLE_WALL_STROKE_MM},
    ))

    swings = [
        e for g in root.iter() if g.get("id") == "symbols"
        for e in g if "door-swing" in (e.get("class") or "")
    ]
    arcs = [e for s in swings for e in s if e.tag.endswith("path")]
    checks.append((
        "Door swings",
        f"{len(arcs)} swing arcs (expect ≥ 6 doors)",
        len(arcs) >= 6,
    ))

    footer_addr = next(
        (e for e in root.iter() if e.get("id") == "footer-address"), None,
    )
    checks.append((
        "Footer",
        "centred address line present",
        footer_addr is not None and bool(footer_addr.text),
    ))

    generator = next(
        (e for e in root.iter() if e.get("id") == "footer-generator"), None,
    )
    checks.append((
        "Footer",
        "no generator watermark on deliverable",
        generator is None,
    ))

    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="Print visual QA checklist instead of only writing files",
    )
    args = parser.parse_args()

    floor = build_demo_floor()
    meta = SheetMetadata(
        building_name="600 5th Avenue",
        address="600 5th Avenue",
        floor_name="Floor 32",
        date_str="2026-07-06",
    )
    grid = fit_column_grid(floor.columns)
    for name, params in (
        ("demo_sheet_minimal", SheetParams(style="stevenson_minimal")),
        ("demo_sheet_full", SheetParams(style="full")),
    ):
        render = render_sheet(floor, meta=meta, grid=grid, params=params)
        svg_path = Path(f"/tmp/{name}.svg")
        svg_path.write_text(render.svg)
        svg_to_pdf(render.svg, f"/tmp/{name}.pdf")
        print(f"{name}: 1:{render.scale_denominator} -> {svg_path} + .pdf")

        if args.check and params.style == "stevenson_minimal":
            print("\n── Visual QA checklist (vs 600 5th Avenue / DEMO41) ──")
            all_ok = True
            for category, detail, ok in _checklist(render.svg):
                mark = "OK" if ok else "DELTA"
                if not ok:
                    all_ok = False
                print(f"  [{mark}] {category}: {detail}")
            print(
                "\nAcceptable deltas: font substitution in PDF, ±0.1 mm stroke "
                "from svglib, column grid letter count when scan bbox differs."
            )
            if not all_ok:
                print("\nSome checks flagged — review /tmp/demo_sheet_minimal.svg")
                sys.exit(1)


if __name__ == "__main__":
    main()
