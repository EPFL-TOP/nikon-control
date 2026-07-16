"""Pure controller for the annotation dashboard — no Bokeh dependency.

Owns an :class:`AnnotationFile` and mediates every edit the UI can make,
mapping each annotation to a stable string id so the view can address
boxes without caring about list order. All geometry conversion between the
schema's ``[y0, x0, y1, x1]`` bbox and Bokeh's centre/width/height Rect
representation lives here.

This module is unit-tested in full; the Bokeh view (``app.py``) only
translates widget events into calls on this class.
"""
from __future__ import annotations

import uuid

from ..schema import (
    Annotation,
    AnnotationFile,
    Keyframe,
    _compute_label,
    _interpolate_bbox,
)


def bbox_to_cwh(bbox: list[float]) -> tuple[float, float, float, float]:
    """[y0, x0, y1, x1] -> (cx, cy, w, h) for a Bokeh Rect glyph."""
    y0, x0, y1, x1 = bbox
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0, (x1 - x0), (y1 - y0))


def cwh_to_bbox(cx: float, cy: float, w: float, h: float) -> list[float]:
    """(cx, cy, w, h) -> [y0, x0, y1, x1]."""
    return [cy - h / 2.0, cx - w / 2.0, cy + h / 2.0, cx + w / 2.0]


class DashboardState:
    def __init__(self, annotation_file: AnnotationFile, n_t: int):
        self.af = annotation_file
        self.n_t = max(1, int(n_t))
        self.current_t = 0
        self._by_id: dict[str, Annotation] = {}
        self._order: list[str] = []
        for a in annotation_file.annotations:
            self._register(a)

    # ---- ids / registration -------------------------------------------
    def _register(self, ann: Annotation) -> str:
        i = uuid.uuid4().hex[:8]
        self._by_id[i] = ann
        self._order.append(i)
        return i

    def ann(self, box_id: str) -> Annotation:
        return self._by_id[box_id]

    def has(self, box_id: str) -> bool:
        return box_id in self._by_id

    def annotations(self) -> list[Annotation]:
        """Current annotations in display order (read-only snapshot)."""
        return [self._by_id[i] for i in self._order]

    @property
    def classes(self) -> list[str]:
        return self.af.classes

    # ---- time ----------------------------------------------------------
    def set_t(self, t: int) -> None:
        self.current_t = int(max(0, min(self.n_t - 1, t)))

    def _t(self, t: int | None) -> int:
        return self.current_t if t is None else int(t)

    # ---- read: boxes visible at a frame --------------------------------
    def boxes_at(self, t: int | None = None) -> list[dict]:
        """Rows for the Bokeh ColumnDataSource at frame ``t`` — one per
        currently-visible annotation.

        ``num`` is a stable 1-based track number (position in the annotation
        order) so the annotator can see a box keep its identity while
        scrubbing — i.e. confirm the track didn't jump to another cell.
        """
        t = self._t(t)
        rows: list[dict] = []
        for n, i in enumerate(self._order, start=1):
            a = self._by_id[i]
            if not a.visible_at(t):
                continue
            cx, cy, w, h = bbox_to_cwh(_interpolate_bbox(a.keyframes, t))
            rows.append(
                {
                    "id": i,
                    "num": n,
                    "label": a.label,
                    "cx": cx,
                    "cy": cy,
                    "w": w,
                    "h": h,
                    "marker": _compute_label(a.t_start, a.t_deaths, t),
                }
            )
        return rows

    # ---- write: geometry ----------------------------------------------
    def add_box(
        self, cx: float, cy: float, w: float, h: float,
        label: str, t: int | None = None, t_start: int = 0,
    ) -> str:
        """Create an annotation with its first keyframe at frame ``t``.

        ``t_start`` defaults to 0 (visible from the start of the recording),
        NOT ``t`` — a box drawn while paused on a mid-recording frame should
        stay visible when the annotator scrubs backward. Use ``mark_birth``
        for the deliberate born-later case. Because a single keyframe yields a
        constant bbox at every T, the box shows in the right place regardless
        of when it was drawn.
        """
        t = self._t(t)
        ann = Annotation(
            label=label,
            keyframes=[Keyframe(t=t, bbox=cwh_to_bbox(cx, cy, w, h))],
            t_start=t_start,
        )
        return self._register(ann)

    def update_box(
        self, box_id: str, cx: float, cy: float, w: float, h: float,
        t: int | None = None,
    ) -> None:
        """Move/resize a box at frame ``t`` — captured as a keyframe there
        (auto-keyframe: a drag at any T is authoritative at that T)."""
        t = self._t(t)
        a = self._by_id[box_id]
        a.keyframes = [k for k in a.keyframes if k.t != t]
        a.keyframes.append(Keyframe(t=t, bbox=cwh_to_bbox(cx, cy, w, h)))
        a.keyframes.sort(key=lambda k: k.t)

    def delete(self, box_id: str) -> None:
        """Delete the whole annotation (track)."""
        self._by_id.pop(box_id, None)
        if box_id in self._order:
            self._order.remove(box_id)

    # ---- write: category + lifecycle ----------------------------------
    def set_label(self, box_id: str, label: str) -> None:
        self._by_id[box_id].label = label
        if label not in self.af.classes:
            self.af.classes.append(label)

    def mark_birth(self, box_id: str, t: int | None = None) -> bool:
        """Set first-visible frame. Refuses (returns False) if it would put
        birth after the end marker, which would make the box invisible at
        every frame and thus impossible to re-select and fix."""
        a = self._by_id[box_id]
        t = self._t(t)
        if a.t_end is not None and t > a.t_end:
            return False
        a.t_start = t
        return True

    def clear_birth(self, box_id: str) -> None:
        self._by_id[box_id].t_start = 0

    def mark_end(self, box_id: str, t: int | None = None) -> bool:
        """Set last-visible frame. Refuses (returns False) if earlier than
        birth (same invisibility trap as above)."""
        a = self._by_id[box_id]
        t = self._t(t)
        if t < a.t_start:
            return False
        a.t_end = t
        return True

    def clear_end(self, box_id: str) -> None:
        self._by_id[box_id].t_end = None

    def add_death(self, box_id: str, t: int | None = None) -> None:
        a = self._by_id[box_id]
        t = self._t(t)
        if t not in a.t_deaths:
            a.t_deaths = sorted(a.t_deaths + [t])

    def pop_death(self, box_id: str) -> None:
        a = self._by_id[box_id]
        if a.t_deaths:
            a.t_deaths = sorted(a.t_deaths)[:-1]

    def clear_deaths(self, box_id: str) -> None:
        self._by_id[box_id].t_deaths = []

    def add_keyframe(self, box_id: str, t: int | None = None) -> None:
        """Pin the interpolated bbox at ``t`` as an explicit keyframe."""
        a = self._by_id[box_id]
        t = self._t(t)
        bbox = _interpolate_bbox(a.keyframes, t)
        a.keyframes = [k for k in a.keyframes if k.t != t]
        a.keyframes.append(Keyframe(t=t, bbox=bbox))
        a.keyframes.sort(key=lambda k: k.t)

    def drop_keyframe(self, box_id: str, t: int | None = None) -> bool:
        """Remove the keyframe at ``t``; refuse to leave zero keyframes."""
        a = self._by_id[box_id]
        t = self._t(t)
        remaining = [k for k in a.keyframes if k.t != t]
        if not remaining or len(remaining) == len(a.keyframes):
            return False
        a.keyframes = remaining
        return True

    # ---- reconcile edits coming back from the view --------------------
    def apply_cds_edits(
        self,
        rows: list[dict],
        default_label: str,
        t: int | None = None,
        *,
        eps: float = 0.5,
    ) -> list[str]:
        """Reconcile the view's box rows into the model at frame ``t``.

        ``rows`` is what the BoxEditTool produced — each a dict with at least
        ``cx, cy, w, h`` and possibly an ``id`` (new boxes have no/empty id).
        Behaviour:

        - New row (no id) -> ``add_box`` with ``default_label``.
        - Existing row whose geometry moved by > ``eps`` px from its
          interpolated bbox at ``t`` -> ``update_box`` (auto-keyframe). A row
          that did not move is left untouched, so scrubbing T and saving does
          not sprinkle spurious keyframes.
        - A box that was visible at ``t`` but is now absent from ``rows``
          -> ``delete`` (BoxEditTool deletion).

        Returns the list of ids for the rows, in order, so the view can write
        the ids back into its ColumnDataSource.
        """
        t = self._t(t)
        visible_before = {r["id"] for r in self.boxes_at(t)}
        seen: set[str] = set()
        out_ids: list[str] = []
        for r in rows:
            rid = r.get("id")
            is_new = (
                rid is None
                or rid == ""
                or (isinstance(rid, float) and rid != rid)  # NaN
                or rid not in self._by_id
            )
            if is_new:
                rid = self.add_box(
                    r["cx"], r["cy"], r["w"], r["h"],
                    label=r.get("label") or default_label, t=t,
                )
            else:
                cx, cy, w, h = bbox_to_cwh(
                    _interpolate_bbox(self._by_id[rid].keyframes, t)
                )
                moved = (
                    abs(cx - r["cx"]) > eps
                    or abs(cy - r["cy"]) > eps
                    or abs(w - r["w"]) > eps
                    or abs(h - r["h"]) > eps
                )
                if moved:
                    self.update_box(rid, r["cx"], r["cy"], r["w"], r["h"], t=t)
            seen.add(rid)
            out_ids.append(rid)
        for rid in visible_before - seen:
            self.delete(rid)
        return out_ids

    def set_detections(self, anns: list[Annotation], provisional_label: str) -> int:
        """Replace auto-detected (still-provisional) tracks with a fresh set,
        keeping any the human already re-classified.

        Re-running detection should refresh the untriaged ``provisional_label``
        (e.g. "cell") boxes without destroying curation work: annotations whose
        label differs from ``provisional_label`` are preserved. Returns the
        number of new annotations added.
        """
        keep = [i for i in self._order if self._by_id[i].label != provisional_label]
        self._by_id = {i: self._by_id[i] for i in keep}
        self._order = keep
        for a in anns:
            self._register(a)
        return len(anns)

    # ---- summaries / persistence --------------------------------------
    def counts(self) -> dict[str, int]:
        c: dict[str, int] = {}
        for a in self._by_id.values():
            c[a.label] = c.get(a.label, 0) + 1
        return c

    def sync_to_file(self) -> AnnotationFile:
        """Rebuild ``af.annotations`` in stable order for saving."""
        self.af.annotations = [self._by_id[i] for i in self._order]
        return self.af
