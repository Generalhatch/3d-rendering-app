"""Structured pipeline warnings — every silent fallback becomes a record.

The vectorize pipeline is deliberately failure-tolerant: envelope
extraction, ceiling gating, wall pairing, room inference, measurement, and
sheet rendering are all guarded so one broken stage never fails the whole
job.  Until Phase 4 those fallbacks only left a line in the SSE stream —
gone the moment the browser tab closed.  Anyone auditing a result later
had no way to know the ceiling gate was skipped or the Manhattan filter
bailed.

Every fallback now appends a structured warning; the full list is persisted
in ``result.json`` under ``warnings`` and surfaced on the job-detail API.

Codes are stable identifiers (snake_case) so the frontend and eval tooling
can filter/aggregate without parsing message text.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# Known warning codes.  Not an enum on purpose — new fallbacks should be
# able to add codes without a lockstep frontend release — but keep this
# list current; it is the reference for consumers.
KNOWN_CODES = frozenset({
    "downsample_failed",          # voxel downsample threw; native resolution used
    "gravity_relevel_excessive_tilt",  # floor plane > max tilt; not releveled
    "gravity_relevel_failed",     # relevel threw; original orientation kept
    "adaptive_band_no_ceiling",   # no ceiling in histogram; default bands kept
    "envelope_failed",            # envelope extraction threw; no exterior lock
    "vertical_filter_fallback",   # vertical-surface filter starved/threw; cloud reloaded
    "ceiling_band_sparse",        # ceiling gate skipped (too few band points)
    "ceiling_band_failed",        # ceiling slice threw; density mask kept
    "fld_no_segments",            # FLD found nothing; fell back to Hough
    "contour_walls_failed",       # contour extraction threw; line detector used
    "manhattan_bail",             # Manhattan filter would drop ALL segments; skipped
    "wall_pairing_failed",        # wall pairing threw; centerlines unpaired
    "room_inference_failed",      # topology threw; no rooms
    "rooms_not_closed",           # topology ran but found 0 rooms — close walls in editor
    "opening_detection_failed",   # opening detector threw; no openings
    "floor_closure_failed",       # floor validation checks failed
    "measurement_failed",         # measurement report threw; not written
    "sheet_failed",               # sheet render threw; not written
    "voxel_escalated",            # dense scan: voxel raised toward accuracy ceiling
})


@dataclass
class PipelineWarning:
    code: str
    stage: str
    message: str

    def to_json_dict(self) -> dict:
        return {"code": self.code, "stage": self.stage, "message": self.message}


@dataclass
class WarningCollector:
    """Accumulates warnings across one pipeline run."""
    warnings: list[PipelineWarning] = field(default_factory=list)

    def add(self, code: str, stage: str, message: str) -> PipelineWarning:
        w = PipelineWarning(code=code, stage=stage, message=message)
        self.warnings.append(w)
        return w

    def codes(self) -> list[str]:
        return [w.code for w in self.warnings]

    def to_json_list(self) -> list[dict]:
        return [w.to_json_dict() for w in self.warnings]

    def __len__(self) -> int:
        return len(self.warnings)
