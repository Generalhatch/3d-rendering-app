"""Confidence scoring — already embedded in align.py, exposed here for reuse."""
from __future__ import annotations


def compute_confidence_pct(confidence: float) -> int:
    """Convert 0.0–1.0 float confidence to 0–100 integer."""
    return int(round(min(100, max(0, confidence * 100))))


def confidence_label(confidence_pct: int) -> str:
    if confidence_pct >= 90:
        return "high"
    elif confidence_pct >= 70:
        return "medium"
    else:
        return "low"
