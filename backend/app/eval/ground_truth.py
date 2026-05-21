"""Ground-truth schema and loader for the eval harness.

A ground-truth file is a small hand-edited JSON that describes what the
pipeline *should* detect on a given scan.  Schema is deliberately minimal —
walls + rooms + columns + openings as lists of geometric primitives,
plus a runtime budget.

Example
-------

.. code-block:: json

    {
      "version": 1,
      "scan": "demo/stevenson.laz",
      "tolerance_m": 0.20,
      "runtime_budget_s": 60,
      "walls": [
        {"x1": 0.0, "y1": 0.0, "x2": 12.5, "y2": 0.0},
        {"x1": 12.5, "y1": 0.0, "x2": 12.5, "y2": 8.0}
      ],
      "rooms_expected": 8,
      "columns_expected": 4,
      "openings_expected": 6
    }

Tolerance
---------
``tolerance_m`` is the buffer radius for segment IoU.  Default 0.20 m
(wall thickness scale).  Set tighter for ML-quality evaluation (5 cm)
or looser for "is the layout right at all" sanity checks (50 cm).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np


@dataclass
class GroundTruth:
    """Hand-edited ground truth for a single scan.

    All geometry in metres, same coordinate frame as the scan.  ``walls``
    is an (N, 2, 2) numpy array after :func:`load_ground_truth`; the JSON
    on disk uses an x1/y1/x2/y2 dict per wall for readability.
    """
    scan: str
    walls: np.ndarray                       # (N, 2, 2)
    rooms_expected: int = 0
    columns_expected: int = 0
    openings_expected: int = 0
    tolerance_m: float = 0.20               # buffer radius for IoU
    runtime_budget_s: float = 60.0          # runs slower than this are failures
    # Optional extras — exact room polygons / column centres — for
    # eval-level granularity beyond just counts.  Empty by default.
    rooms_polygons: list[np.ndarray] = field(default_factory=list)
    column_centres: list[tuple[float, float]] = field(default_factory=list)

    @property
    def n_walls(self) -> int:
        return int(len(self.walls))


def load_ground_truth(path: Path) -> GroundTruth:
    """Load + validate a ground-truth JSON file."""
    path = Path(path)
    data = json.loads(path.read_text())
    if data.get("version", 1) != 1:
        raise ValueError(
            f"unsupported ground-truth version {data.get('version')}; "
            f"only v1 is implemented"
        )
    walls_raw = data.get("walls", [])
    if walls_raw:
        walls = np.array([
            [[float(w["x1"]), float(w["y1"])], [float(w["x2"]), float(w["y2"])]]
            for w in walls_raw
        ], dtype=np.float64)
    else:
        walls = np.zeros((0, 2, 2), dtype=np.float64)

    rooms_polygons = []
    for poly_raw in data.get("rooms_polygons", []):
        rooms_polygons.append(np.array(poly_raw, dtype=np.float64))

    return GroundTruth(
        scan=str(data.get("scan", "")),
        walls=walls,
        rooms_expected=int(data.get("rooms_expected", 0)),
        columns_expected=int(data.get("columns_expected", 0)),
        openings_expected=int(data.get("openings_expected", 0)),
        tolerance_m=float(data.get("tolerance_m", 0.20)),
        runtime_budget_s=float(data.get("runtime_budget_s", 60.0)),
        rooms_polygons=rooms_polygons,
        column_centres=[tuple(c) for c in data.get("column_centres", [])],
    )


def save_ground_truth(gt: GroundTruth, path: Path) -> Path:
    """Persist a ground-truth as JSON."""
    path = Path(path)
    payload = {
        "version": 1,
        "scan": gt.scan,
        "tolerance_m": gt.tolerance_m,
        "runtime_budget_s": gt.runtime_budget_s,
        "rooms_expected": gt.rooms_expected,
        "columns_expected": gt.columns_expected,
        "openings_expected": gt.openings_expected,
        "walls": [
            {"x1": float(w[0, 0]), "y1": float(w[0, 1]),
             "x2": float(w[1, 0]), "y2": float(w[1, 1])}
            for w in gt.walls
        ],
        "rooms_polygons": [p.tolist() for p in gt.rooms_polygons],
        "column_centres": [list(c) for c in gt.column_centres],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path
