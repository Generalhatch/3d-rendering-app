"""Scan-to-CAD vectorization pipeline.

Takes a 3D point cloud (or, eventually, a pre-rasterized 2D plan slice) and
emits clean DXF vector geometry suitable for an as-built CAD deliverable.

This package is intentionally separate from :mod:`app.pipeline`, which solves
the *alignment* problem (3D scan ↔ existing DXF plan).  The two pipelines
share library dependencies (OpenCV, ezdxf, Open3D) and FastAPI plumbing but
have entirely different stages.

Module map
----------
- :mod:`.slicer`     — point cloud → 2D raster image at a chosen elevation
- :mod:`.preprocess` — adaptive threshold, morphology, contrast normalisation
- :mod:`.classical`  — raster → line segments (Hough, FLD)
- :mod:`.regularize` — merge collinear, snap to Manhattan, drop tiny segments
- :mod:`.dxf_writer` — segments → layered DXF file
- :mod:`.pipeline`   — orchestrator; the function that the API route calls
"""
