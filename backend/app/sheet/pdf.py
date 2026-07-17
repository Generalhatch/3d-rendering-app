"""SVG → PDF conversion for the sheet renderer.

Uses svglib + reportlab (pure Python — no native cairo dependency).  The
SVG stays the single source of truth: the PDF is a faithful conversion,
never a re-render, so the two deliverables cannot drift apart.
"""
from __future__ import annotations

import io
from pathlib import Path


def svg_to_pdf(svg: str, pdf_path: str | Path) -> Path:
    """Convert an SVG document string to a PDF file.

    Raises ``RuntimeError`` with an actionable message when the optional
    svglib/reportlab dependencies are missing, and ``ValueError`` when the
    SVG cannot be parsed.
    """
    try:
        from reportlab.graphics import renderPDF
        from svglib.svglib import svg2rlg
    except ImportError as exc:   # pragma: no cover — deps are in pyproject
        raise RuntimeError(
            "SVG→PDF conversion requires svglib + reportlab "
            "(pip install svglib reportlab)"
        ) from exc

    drawing = svg2rlg(io.StringIO(svg))
    if drawing is None:
        raise ValueError("could not parse SVG document for PDF conversion")

    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    renderPDF.drawToFile(drawing, str(pdf_path))
    return pdf_path
