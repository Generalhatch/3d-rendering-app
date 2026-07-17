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
    transformation: np.ndarray,   # 4x4, column-vector convention (v' = T @ v)
    output_path: Path,
) -> Path:
    """Write a DXF with all modelspace entities transformed by the alignment matrix.

    The 4x4 ``transformation`` (numpy column-vector convention) is converted
    to an :class:`ezdxf.math.Matrix44` and applied to every entity that
    supports transformation.  Entities that cannot be transformed (rare —
    e.g. malformed proxies) are left in place and counted in the meta note.
    """
    import ezdxf
    from ezdxf.math import Matrix44

    doc = ezdxf.readfile(str(original_dxf_path))
    msp = doc.modelspace()

    # ezdxf's Matrix44 uses the row-vector convention (v' = v @ M), so the
    # numpy column-vector matrix must be transposed.
    m44 = Matrix44(np.asarray(transformation, dtype=np.float64).T.flatten())

    n_transformed = 0
    n_skipped = 0
    is_identity = np.allclose(transformation, np.eye(4))
    if not is_identity:
        for entity in list(msp):
            try:
                entity.transform(m44)
                n_transformed += 1
            except Exception:
                # NotImplementedError / NonUniformScalingError / malformed
                # entities — leave untouched rather than aborting the export.
                n_skipped += 1

    # Traceability note (added AFTER the transform so it stays at the origin
    # on its own meta layer).
    if "ALIGNAI_META" not in doc.layers:
        doc.layers.new("ALIGNAI_META", dxfattribs={"color": 7})
    note = (
        f"AlignAI alignment applied to {n_transformed} entities"
        + (f" ({n_skipped} skipped)" if n_skipped else "")
        + " — see JSON export for full matrix"
    )
    msp.add_text(
        note,
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
