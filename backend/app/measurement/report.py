"""Measurement report generator: JSON + rendered PDF table.

The report combines one floor's measurements under several standards into
a single deliverable: usable / rentable / gross per suite and per floor,
every number carrying its standard + method stamp.

PDF rendering uses matplotlib (already a dependency) — one landscape page
per standard, a header block, and a table of suites + floor totals.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from ..geometry.model import FloorGeometry
from .base import SQFT_PER_M2, FloorMeasurement

# ── JSON report ───────────────────────────────────────────────────────────────

def build_report(
    floor: FloorGeometry,
    measurements: Sequence[FloorMeasurement],
) -> dict:
    """Combine measurements of one floor into a JSON-safe report dict."""
    return {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "floor_id": floor.floor_id,
        "building_name": floor.building_name,
        "floor_name": floor.floor_name,
        "units": {"area": "m2", "secondary": "sqft"},
        "geometry_summary": {
            "n_rooms": len(floor.rooms),
            "n_walls": len(floor.walls),
            "n_penetrations": len(floor.penetrations),
            "envelope_area_m2": round(floor.envelope_area_m2(), 4),
            "source": floor.source,
        },
        "measurements": [m.to_json_dict() for m in measurements],
    }


def write_json_report(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(report, indent=2))
    return path


# ── PDF report ────────────────────────────────────────────────────────────────

def _fmt(v: dict | None) -> str:
    if v is None:
        return "—"
    return f"{v['value_m2']:,.2f} m²  ({v['value_m2'] * SQFT_PER_M2:,.0f} sf)"


def write_pdf_report(report: dict, path: str | Path) -> Path:
    """Render the report as a PDF — one page per standard."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    path = Path(path)
    title_bits = [b for b in (report.get("building_name"), report.get("floor_name")) if b]
    floor_title = " — ".join(title_bits) if title_bits else report["floor_id"]

    with PdfPages(path) as pdf:
        for m in report["measurements"]:
            fig, ax = plt.subplots(figsize=(11.69, 8.27))  # A4 landscape
            ax.axis("off")

            method = f" · {m['method']}" if m.get("method") else ""
            ax.text(0.02, 0.97, f"Measurement Report — {m['standard']}{method}",
                    fontsize=16, fontweight="bold", va="top",
                    transform=ax.transAxes)
            ax.text(0.02, 0.915,
                    f"Floor: {floor_title}    Generated: {report['generated_at'][:19]}Z    "
                    f"Load factor: {m['load_factor']:.4f}",
                    fontsize=9, va="top", transform=ax.transAxes, color="#444444")

            rows: list[list[str]] = []
            for suite in m["suites"]:
                rows.append([
                    suite["suite_id"],
                    suite["label"],
                    _fmt(suite["usable"]),
                    _fmt(suite["rentable"]),
                    _fmt(suite.get("gross")),
                ])
            rows.append([
                "COMMON", "Floor common area",
                "—", _fmt(m["common_area"]), "—",
            ])
            rows.append([
                "FLOOR", "Floor totals",
                _fmt(m["floor_usable"]),
                _fmt(m["floor_rentable"]),
                _fmt(m.get("floor_gross")),
            ])

            table = ax.table(
                cellText=rows,
                colLabels=["Suite", "Label", "Usable", "Rentable", "Gross"],
                colWidths=[0.12, 0.28, 0.20, 0.20, 0.20],
                loc="upper center",
                bbox=[0.02, 0.30, 0.96, 0.55],
                cellLoc="left",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            for (row, _col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_facecolor("#2b3a55")
                    cell.set_text_props(color="white", fontweight="bold")
                elif row == len(rows):  # floor totals row
                    cell.set_facecolor("#e8ecf5")
                    cell.set_text_props(fontweight="bold")
                cell.set_edgecolor("#c9cdd6")

            notes = m.get("notes") or []
            footer = (
                f"Standard: {m['standard']}{method} — every figure above is "
                f"computed under this standard's boundary rules."
            )
            if notes:
                footer += "\nNotes: " + "; ".join(notes)
            ax.text(0.02, 0.22, footer, fontsize=8, va="top",
                    transform=ax.transAxes, color="#555555", wrap=True)

            pdf.savefig(fig)
            plt.close(fig)

    return path
