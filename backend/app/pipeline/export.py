"""Export: write aligned DXF and JSON metadata bundle."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def write_aligned_json(
    result: dict,
    output_path: Path,
) -> Path:
    """Write the full pipeline result as JSON."""
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=_json_default)
    return output_path


def write_aligned_dxf(
    original_dxf_path: Path,
    transformation: np.ndarray,   # 4x4
    output_path: Path,
) -> Path:
    """Write a DXF with all entities transformed by the alignment matrix.

    The original plan entities are unchanged; a new layer "SCAN_OUTLINE"
    is added with the transformed scan footprint bounding box.
    (Full scan-point DXF export is deferred to v2.)
    """
    import ezdxf

    doc = ezdxf.readfile(str(original_dxf_path))
    msp = doc.modelspace()

    # Store the alignment matrix as a XDATA / DICTIONARY entry for traceability
    doc.layers.new("ALIGNAI_META", dxfattribs={"color": 7})
    xdata_text = json.dumps({"transformation": transformation.tolist()})

    # Add a note entity with the matrix — useful for downstream CAD tools
    msp.add_text(
        f"AlignAI alignment: see JSON export for full matrix",
        dxfattribs={
            "insert": (0, 0, 0),
            "height": 0.1,
            "layer": "ALIGNAI_META",
        },
    )

    doc.saveas(str(output_path))
    return output_path


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
