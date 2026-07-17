"""Pluggable measurement engine (Phase 1).

A :class:`~app.measurement.base.Ruleset` turns a canonical
:class:`~app.geometry.model.FloorGeometry` into a
:class:`~app.measurement.base.FloorMeasurement` — usable / rentable /
gross per suite and per floor, with every number stamped with its standard
and method.

Registered rulesets:

- ``boma_2024_a`` — BOMA Office 2024, Method A (floor-by-floor load factor)
- ``boma_2024_b`` — BOMA Office 2024, Method B (single building load factor)
- ``rebny``       — REBNY (outside-face measurement, per-floor loss factor)
- ``gross``       — simple Gross Area (envelope / outside face)

IPMS is not implemented yet, but the geometry model captures both wall
faces + thickness, so IPMS 1 (external) and IPMS 2 (internal dominant
face) can be added as pure rulesets without touching any extractor.
"""
from __future__ import annotations

from typing import Callable

from .base import AreaValue, FloorMeasurement, Ruleset, SuiteMeasurement  # noqa: F401
from .rulesets import (  # noqa: F401
    BomaOffice2024MethodA,
    BomaOffice2024MethodB,
    GrossArea,
    Rebny,
)

_REGISTRY: dict[str, Callable[[], Ruleset]] = {
    "boma_2024_a": BomaOffice2024MethodA,
    "boma_2024_b": BomaOffice2024MethodB,
    "rebny": Rebny,
    "gross": GrossArea,
}


def list_rulesets() -> list[str]:
    """Names of all registered measurement rulesets."""
    return sorted(_REGISTRY)


def get_ruleset(name: str) -> Ruleset:
    """Instantiate a registered ruleset by name.  Raises KeyError if unknown."""
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown measurement ruleset {name!r} — available: {list_rulesets()}"
        )
    return _REGISTRY[name]()
