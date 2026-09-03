"""Simplified per-frame annotation schema — for detector training data.

This is a DELIBERATELY separate, much flatter format from
``schema.py`` (the tracked/lifecycle format used by the full dashboard).
It exists because training a single/doublet/debris classifier-detector does
not need tracks: what the model sees is one frame at a time, so a label is
just *(frame, box, class)*.

Key differences from the tracked schema:

- **Per-frame boxes.** A box belongs to exactly ONE frame, and each carries
  its own bbox and class — that is what gets exported for training.
- **A light identity link** (``SimpleBox.group``) may join the boxes of one
  physical cell across frames. It exists purely so the annotator classifies
  a cell ONCE instead of on every frame; geometry stays per-frame (so a
  drifting cell is still correct), and nothing about the exported labels
  depends on it. Boxes with ``group=None`` are standalone.
- **No lifecycle.** No t_start / t_end / t_deaths / class_changes /
  keyframes / interpolation.
- **Three training classes** (``TRAINING_CLASSES``): single, doublet,
  debris — plus the provisional ``PROVISIONAL_LABEL`` ("unlabeled") that
  fresh detections carry until a human classifies them. Unlabeled boxes are
  excluded from training exports, so every exported label is human-verified.
- **Written to a distinct filename** (``<file>.simple.json``, see
  ``simple_path_for``) and carries ``kind`` = ``SIMPLE_KIND``, so a
  simplified file can never be mistaken for a tracked one (``load_simple``
  refuses a tracked file and vice versa).

The flat ``boxes`` list maps 1:1 onto a detection training set: one image
per (source, t, channel), one annotation per box — i.e. directly
convertible to COCO. See ``docs/training-export.md``.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

SIMPLE_SCHEMA_VERSION = "1.0"

# Discriminator written into every file so the two formats can't be confused.
SIMPLE_KIND = "per-frame-boxes"

# Filename suffix, distinct from the tracked format's ".annotations.json".
SIMPLE_SUFFIX = ".simple.json"

# The classes the detector will be trained on.
TRAINING_CLASSES: tuple[str, ...] = ("single", "doublet", "debris")

# Fresh cell detections get this label; a human must replace it with one of
# TRAINING_CLASSES. Excluded from training exports, so exported labels are
# always human-verified.
PROVISIONAL_LABEL = "unlabeled"

# Full class list used for display (colour legend, category buttons).
SIMPLE_CLASSES: tuple[str, ...] = (*TRAINING_CLASSES, PROVISIONAL_LABEL)

# Default number of leading frames to annotate (settable in the dashboard):
# the model must recognise singles/doublets at an EARLY stage.
DEFAULT_N_FRAMES = 20


@dataclass
class SimpleBox:
    """One labelled box on one frame. Independent of every other box."""

    t: int
    bbox: list[float]  # [y0, x0, y1, x1] — same convention as schema.py
    label: str
    z: int = 0
    # detector confidence when the box came from a model, else None. Kept for
    # provenance/QC; not used for training.
    score: float | None = None
    # True while the box is exactly as a detector produced it. Cleared as soon
    # as a human classifies, moves, or resizes it. Re-running detection
    # replaces only ``auto`` boxes, so it can never destroy human work.
    auto: bool = False
    # Optional identity shared by the boxes of ONE physical cell across
    # frames, so a class can be set once for the whole cell. Purely an
    # annotation convenience — training uses ``t``/``bbox``/``label`` only.
    # ``None`` = a standalone box.
    group: str | None = None

    @property
    def is_training_label(self) -> bool:
        return self.label in TRAINING_CLASSES

    @property
    def is_human_verified(self) -> bool:
        return not self.auto


@dataclass
class SimpleAnnotationFile:
    source: str
    schema_version: str = SIMPLE_SCHEMA_VERSION
    kind: str = SIMPLE_KIND
    image_shape: list[int] = field(default_factory=list)
    axes: list[str] = field(default_factory=list)
    channels: list[str] = field(default_factory=list)
    # index of the channel the boxes were drawn on (brightfield, normally) —
    # the exporter needs this to render the matching plane.
    bf_channel: int = 0
    # how many leading frames this file covers (boxes only exist for t < this)
    n_frames: int = DEFAULT_N_FRAMES
    classes: list[str] = field(default_factory=lambda: list(SIMPLE_CLASSES))
    annotator: str = ""
    created: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )
    boxes: list[SimpleBox] = field(default_factory=list)

    def training_boxes(self, verified_only: bool = False) -> list[SimpleBox]:
        """Boxes usable for training — i.e. carrying a real class.

        Unlabeled (never-triaged) boxes are always excluded, so every
        exported label has been chosen by a person. ``verified_only`` further
        restricts to boxes a human explicitly touched, dropping auto-labelled
        debris that was merely left in place — useful for a high-precision
        subset.
        """
        return [b for b in self.boxes
                if b.is_training_label
                and (b.is_human_verified or not verified_only)]


def simple_path_for(nd2_path: str | Path) -> Path:
    """Sidecar path for an ND2: ``foo.nd2`` -> ``foo.simple.json``."""
    return Path(nd2_path).with_suffix(SIMPLE_SUFFIX)


def save_simple(af: SimpleAnnotationFile, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    af.schema_version = SIMPLE_SCHEMA_VERSION
    af.kind = SIMPLE_KIND
    payload = {
        "kind": af.kind,
        "schema_version": af.schema_version,
        "source": af.source,
        "image_shape": list(af.image_shape),
        "axes": list(af.axes),
        "channels": list(af.channels),
        "bf_channel": af.bf_channel,
        "n_frames": af.n_frames,
        "classes": list(af.classes),
        "annotator": af.annotator,
        "created": af.created,
        "boxes": [asdict(b) for b in af.boxes],
    }
    path.write_text(json.dumps(payload, indent=2))


def load_simple(path: str | Path) -> SimpleAnnotationFile:
    """Load a simplified file, refusing a tracked ``.annotations.json``."""
    path = Path(path)
    payload = json.loads(path.read_text())
    kind = payload.get("kind")
    if kind != SIMPLE_KIND:
        raise ValueError(
            f"'{path.name}' is not a simplified annotation file "
            f"(kind={kind!r}, expected {SIMPLE_KIND!r}). The tracked format is "
            "loaded with nikon_control.schema.load() instead."
        )
    boxes = [SimpleBox(**b) for b in payload.pop("boxes", [])]
    payload["schema_version"] = SIMPLE_SCHEMA_VERSION
    af = SimpleAnnotationFile(**payload)
    af.boxes = boxes
    # a file written before a class was added still gets the current palette
    for c in SIMPLE_CLASSES:
        if c not in af.classes:
            af.classes.append(c)
    return af


def boxes_by_frame(af: SimpleAnnotationFile) -> dict[int, list[SimpleBox]]:
    """Group boxes by frame — the shape a training exporter wants."""
    out: dict[int, list[SimpleBox]] = {}
    for b in af.boxes:
        out.setdefault(b.t, []).append(b)
    return out


def class_counts(af: SimpleAnnotationFile) -> dict[str, int]:
    """How many boxes per class — annotation progress / class balance."""
    counts: dict[str, int] = {c: 0 for c in af.classes}
    for b in af.boxes:
        counts[b.label] = counts.get(b.label, 0) + 1
    return counts


def boxes_from_annotations(anns, n_frames: int, label: str | None = None,
                           score: float | None = None,
                           group: bool = True) -> list[SimpleBox]:
    """Flatten tracked annotations into independent per-frame boxes.

    Used to reuse the *tracked* detectors (e.g. debris detection, which needs
    a temporal background) inside the simplified per-frame workflow: each
    track is expanded into one box per frame it is visible on, and the track
    identity is then discarded.

    With ``group=True`` (default) each track's boxes share a ``group`` id, so
    the annotator can re-classify or delete the whole object in one action
    while its geometry stays per-frame.

    ``anns`` is duck-typed on the tracked ``schema.Annotation`` interface
    (``t_start``, ``t_end``, ``label``, ``bbox_at(t)``) so this module stays
    independent of the tracked schema.
    """
    out: list[SimpleBox] = []
    last = n_frames - 1
    for a in anns:
        lo = max(0, int(a.t_start))
        hi = last if a.t_end is None else min(last, int(a.t_end))
        gid = uuid.uuid4().hex if group else None
        for t in range(lo, hi + 1):
            out.append(SimpleBox(t=t, bbox=[float(v) for v in a.bbox_at(t)],
                                 label=label or a.label, score=score,
                                 group=gid))
    return out


def group_counts(af: SimpleAnnotationFile) -> dict[str, int]:
    """How many distinct CELLS (groups) per class, vs boxes.

    A cell tracked across 20 frames is 20 boxes but one cell; this is the
    number that reflects how much genuinely independent data there is.
    Standalone boxes each count as their own cell.
    """
    seen: dict[str, set] = {}
    for i, b in enumerate(af.boxes):
        seen.setdefault(b.label, set()).add(b.group or f"_solo{i}")
    return {label: len(g) for label, g in seen.items()}
