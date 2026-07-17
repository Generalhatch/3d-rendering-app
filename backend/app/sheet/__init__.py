"""Floor plan sheet rendering (Phase 3 of the Scan-to-BOMA overhaul).

Turns the canonical :class:`app.geometry.model.FloorGeometry` into a
Stevenson-quality drawing sheet:

- :mod:`app.sheet.grid`       — column grid fitting (+ manual entry fallback)
- :mod:`app.sheet.labels`     — suite label placement with collision avoidance
- :mod:`app.sheet.render`     — SVG sheet renderer (title block, scale bar,
                                north arrow, line-weight hierarchy, grid bubbles)
- :mod:`app.sheet.pdf`        — SVG → PDF conversion
- :mod:`app.sheet.dimensions` — DIMENSION entities appended to the DXF
                                deliverable from measured wall faces
"""
