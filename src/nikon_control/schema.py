"""Annotation data model — GUI-agnostic.

This module holds the on-disk annotation schema and all pure logic for
reading, writing, migrating, and interpolating annotations. It has **no UI
dependency** (no napari, no bokeh), so it is shared by:

- the pre-annotation pipeline (``preannotate.py``), which writes JSON from
  model detections + tracking, and
- the Bokeh dashboard, which loads/edits/saves the same JSON.

An annotation describes one tracked object over time:

- ``label`` — class (single / doublet / debris / ...).
- ``keyframes`` — ordered ``(t, bbox)`` records. A static object has one
  keyframe; a moving one (drifting debris, migrating cell) has several. The
  displayed bbox at any T is linearly interpolated between neighbours and
  snaps to the nearest keyframe outside the range. A per-timepoint model
  detection maps to exactly one keyframe.
- ``t_start`` — first visible frame.
- ``t_end`` — last visible frame (``None`` = until end of recording).
- ``t_deaths`` — list of death frames (doublets can have two).

Schema version 0.5. Older files are migrated on load (see
``_upgrade_annotation``).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

SCHEMA_VERSION = "0.6"
DEFAULT_CLASSES: tuple[str, ...] = (
    "single",
    "doublet",
    "debris",
    "fission_fusion",
)

# Class a track becomes at/after its division frame (single -> doublet).
POST_DIVISION_LABEL = "doublet"


@dataclass
class Keyframe:
    t: int
    bbox: list[float]  # [y0, x0, y1, x1]


@dataclass
class Annotation:
    label: str
    keyframes: list[Keyframe] = field(default_factory=list)
    t_start: int = 0
    t_end: int | None = None
    t_deaths: list[int] = field(default_factory=list)
    t_divide: int | None = None  # frame the cell divides (single -> doublet)
    z: int = 0
    notes: str = ""
    created: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )

    @property
    def bbox(self) -> list[float]:
        """First-keyframe bbox, for callers that want a single representative."""
        if not self.keyframes:
            return [0.0, 0.0, 0.0, 0.0]
        return self.keyframes[0].bbox

    def bbox_at(self, t: int) -> list[float]:
        """Interpolated bbox at frame ``t``."""
        return _interpolate_bbox(self.keyframes, t)

    def visible_at(self, t: int) -> bool:
        end = self.t_end if self.t_end is not None else 10**9
        return self.t_start <= t <= end

    def label_at(self, t: int) -> str:
        """Effective class at frame ``t``: the base ``label`` before division,
        and ``POST_DIVISION_LABEL`` ("doublet") from the division frame on."""
        if self.t_divide is not None and t >= self.t_divide:
            return POST_DIVISION_LABEL
        return self.label


@dataclass
class AnnotationFile:
    source: str
    schema_version: str = SCHEMA_VERSION
    image_shape: list[int] = field(default_factory=list)
    axes: list[str] = field(default_factory=list)
    channels: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=lambda: list(DEFAULT_CLASSES))
    annotator: str = ""
    annotations: list[Annotation] = field(default_factory=list)


def save(ann: AnnotationFile, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ann.schema_version = SCHEMA_VERSION
    payload = {
        "source": ann.source,
        "schema_version": ann.schema_version,
        "image_shape": list(ann.image_shape),
        "axes": list(ann.axes),
        "channels": list(ann.channels),
        "classes": list(ann.classes),
        "annotator": ann.annotator,
        "annotations": [
            {
                "label": a.label,
                "keyframes": [asdict(k) for k in a.keyframes],
                "t_start": a.t_start,
                "t_end": a.t_end,
                "t_deaths": list(a.t_deaths),
                "t_divide": a.t_divide,
                "z": a.z,
                "notes": a.notes,
                "created": a.created,
            }
            for a in ann.annotations
        ],
    }
    path.write_text(json.dumps(payload, indent=2))


def _upgrade_annotation(a: dict, from_version: str) -> dict:
    """Migrate per-annotation dict from any older schema to v0.6."""
    # v0.1: drop the per-frame ``t``.
    a.pop("t", None)
    a.setdefault("t_start", 0)
    # v0.2: old ``t_end`` was the death marker.
    if from_version == "0.2":
        old_t_end = a.get("t_end")
        if old_t_end is not None and "t_death" not in a:
            a["t_death"] = old_t_end
            a["t_end"] = None
    # v0.3: scalar ``t_death`` → list ``t_deaths``.
    if "t_deaths" not in a:
        single = a.pop("t_death", None)
        a["t_deaths"] = [single] if single is not None else []
    else:
        a.pop("t_death", None)
    a.setdefault("t_end", None)
    # v0.4: ``bbox`` becomes a single keyframe at ``t_start``.
    if "keyframes" not in a:
        bbox = a.pop("bbox", None)
        if bbox is not None:
            a["keyframes"] = [{"t": a.get("t_start", 0), "bbox": list(bbox)}]
        else:
            a["keyframes"] = []
    else:
        a.pop("bbox", None)
    # v0.6: division marker (single -> doublet). Absent in older files.
    a.setdefault("t_divide", None)
    return a


def load(path: Path) -> AnnotationFile:
    path = Path(path)
    payload = json.loads(path.read_text())
    raw_anns = payload.pop("annotations", [])
    from_version = payload.get("schema_version", "0.1")
    payload["schema_version"] = SCHEMA_VERSION
    af = AnnotationFile(**payload)
    annotations: list[Annotation] = []
    for raw in raw_anns:
        upgraded = _upgrade_annotation(raw, from_version)
        kfs = [Keyframe(**kf) for kf in upgraded.pop("keyframes", [])]
        annotations.append(Annotation(keyframes=kfs, **upgraded))
    af.annotations = annotations
    return af


def _shape_to_bbox(rect) -> list[float]:
    """Extract bbox=[y0,x0,y1,x1] from a rectangle's vertex array (uses the
    last two axes as (y, x))."""
    import numpy as np

    rect = np.asarray(rect)
    ys = rect[:, -2]
    xs = rect[:, -1]
    return [float(ys.min()), float(xs.min()), float(ys.max()), float(xs.max())]


def _bbox_to_shape(bbox: list[float]) -> list[list[float]]:
    """Build a 4-vertex 2D rectangle in (y, x) order."""
    y0, x0, y1, x1 = bbox
    return [[y0, x0], [y0, x1], [y1, x1], [y1, x0]]


def _bboxes_close(a: list[float], b: list[float], atol: float = 1e-6) -> bool:
    """True if two bboxes agree component-wise within ``atol``."""
    if len(a) != len(b):
        return False
    return all(abs(float(a[i]) - float(b[i])) <= atol for i in range(len(a)))


def _interpolate_bbox(keyframes: list[Keyframe], t: int) -> list[float]:
    """Linear interpolation between surrounding keyframes; snap outside the range.

    Single keyframe → constant bbox at every T. Multiple keyframes → linear
    interpolation between the two flanking a given T; for T < first keyframe,
    snap to first; for T > last, snap to last.
    """
    if not keyframes:
        return [0.0, 0.0, 0.0, 0.0]
    if len(keyframes) == 1:
        return list(keyframes[0].bbox)
    sorted_kfs = sorted(keyframes, key=lambda k: k.t)
    if t <= sorted_kfs[0].t:
        return list(sorted_kfs[0].bbox)
    if t >= sorted_kfs[-1].t:
        return list(sorted_kfs[-1].bbox)
    for i in range(len(sorted_kfs) - 1):
        k0, k1 = sorted_kfs[i], sorted_kfs[i + 1]
        if k0.t <= t <= k1.t:
            if k1.t == k0.t:
                return list(k0.bbox)
            alpha = (t - k0.t) / (k1.t - k0.t)
            return [
                k0.bbox[j] + alpha * (k1.bbox[j] - k0.bbox[j]) for j in range(4)
            ]
    return list(sorted_kfs[-1].bbox)


def _compute_label(
    t_start: int, t_deaths: list[int], current_t: int,
    t_divide: int | None = None,
) -> str:
    """Marker text for a bbox at the current T frame.

    Birth (``↑``) shown when ``t_start > 0``; division (``⑂``) once the
    current frame is at/after ``t_divide``; deaths (``†``) one per death that
    has already happened (``d <= current_t``).
    """
    parts = []
    if t_start > 0:
        parts.append(f"↑T={t_start}")
    if t_divide is not None and current_t >= t_divide:
        parts.append(f"⑂T={t_divide}")
    past_deaths = sorted(d for d in t_deaths if d <= current_t)
    if past_deaths:
        parts.append("†T=" + ",".join(str(d) for d in past_deaths))
    return " ".join(parts)
