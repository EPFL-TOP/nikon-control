"""Overlap suppression for annotation boxes — GUI-free and pure.

Why this exists: a multi-class detector applies non-maximum suppression
*per class*, so one object can come back once as ``single`` and again as
``doublet``. Running the cell and debris passes also produces boxes that
overlap each other, and a human can add a box on top of a detection.

Overlap measure
---------------
``iomin`` — intersection divided by the area of the **smaller** box — not
IoU. "These two boxes overlap by more than 70%" is a statement about the
smaller one, and only ``iomin`` catches containment: a small ``single`` box
fully inside a big ``doublet`` box has ``iomin == 1.0`` but an IoU that can
easily be below 0.3, so an IoU rule would leave the duplicate in place.

Which box survives
------------------
Boxes are ranked and the best one in each overlapping cluster is kept:

1. **Protected boxes first.** A box a human drew, classified, moved or
   resized always outranks raw detector output, and is never dropped in
   favour of it. Two protected boxes that overlap are BOTH kept — deleting
   deliberate human work is not this function's job (``drop_protected``
   overrides that if a caller really wants it).
2. Then by ``prefer``: ``"area"`` (default — bigger wins, score breaks ties)
   or ``"score"`` (higher confidence wins, area breaks ties).
3. Then by input order, so the result is deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass

PREFER_AREA = "area"
PREFER_SCORE = "score"
DEFAULT_MAX_OVERLAP = 0.7


def bbox_area(bbox) -> float:
    y0, x0, y1, x1 = (float(v) for v in bbox)
    return max(0.0, y1 - y0) * max(0.0, x1 - x0)


def bbox_intersection(a, b) -> float:
    ay0, ax0, ay1, ax1 = (float(v) for v in a)
    by0, bx0, by1, bx1 = (float(v) for v in b)
    dy = min(ay1, by1) - max(ay0, by0)
    dx = min(ax1, bx1) - max(ax0, bx0)
    if dy <= 0 or dx <= 0:
        return 0.0
    return dy * dx


def bbox_iou(a, b) -> float:
    inter = bbox_intersection(a, b)
    if inter <= 0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def bbox_iomin(a, b) -> float:
    """Intersection over the SMALLER box's area — 1.0 when one contains the
    other, whatever their size difference."""
    inter = bbox_intersection(a, b)
    if inter <= 0:
        return 0.0
    smaller = min(bbox_area(a), bbox_area(b))
    return inter / smaller if smaller > 0 else 0.0


@dataclass
class OverlapCandidate:
    """One box entering suppression. ``ref`` is the caller's own handle."""

    ref: object
    bbox: list[float]
    score: float | None = None
    protected: bool = False


def select_survivors(
    candidates: list[OverlapCandidate],
    max_overlap: float = DEFAULT_MAX_OVERLAP,
    prefer: str = PREFER_AREA,
    drop_protected: bool = False,
) -> tuple[list[object], list[object]]:
    """Resolve overlapping boxes.

    Returns ``(kept_refs, dropped_refs)``. See the module docstring for the
    measure and the ranking. ``max_overlap`` is a fraction (0.7 = 70%).
    """
    if prefer not in (PREFER_AREA, PREFER_SCORE):
        raise ValueError(f"prefer must be {PREFER_AREA!r} or {PREFER_SCORE!r}")

    def rank(item: tuple[int, OverlapCandidate]):
        i, c = item
        area = bbox_area(c.bbox)
        # None score sorts lowest among unprotected boxes
        score = -1.0 if c.score is None else float(c.score)
        primary, secondary = ((area, score) if prefer == PREFER_AREA
                              else (score, area))
        # protected first, then the preference, then input order (ascending i
        # via negation, since we sort descending)
        return (0 if (c.protected and not drop_protected) else 1,
                -primary, -secondary, i)

    kept: list[tuple[int, OverlapCandidate]] = []
    dropped: list[object] = []
    for i, c in sorted(enumerate(candidates), key=rank):
        clash = None
        for _, k in kept:
            if bbox_iomin(c.bbox, k.bbox) > max_overlap:
                clash = k
                break
        if clash is None:
            kept.append((i, c))
            continue
        # never let one deliberate human box delete another
        if (c.protected and clash.protected) and not drop_protected:
            kept.append((i, c))
            continue
        dropped.append(c.ref)
    return [c.ref for _, c in kept], dropped


def count_overlaps(candidates: list[OverlapCandidate],
                   max_overlap: float = DEFAULT_MAX_OVERLAP) -> int:
    """How many boxes overlap at least one other beyond the limit.

    For reporting (e.g. an export warning) without changing anything.
    """
    n = 0
    for i, c in enumerate(candidates):
        for j, o in enumerate(candidates):
            if i != j and bbox_iomin(c.bbox, o.bbox) > max_overlap:
                n += 1
                break
    return n
