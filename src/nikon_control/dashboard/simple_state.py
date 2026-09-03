"""Controller for the SIMPLIFIED (per-frame) annotation dashboard.

Same role as ``state.py`` — all correctness logic lives here so it can be
unit-tested without a browser — but a much smaller model, because the
simplified task has no tracking and no lifecycle:

- a box belongs to exactly ONE frame, and each carries its own bbox and
  class — that is what gets exported;
- boxes of one physical cell may share a ``group`` id, so the annotator sets
  the class ONCE per cell instead of once per frame. Geometry stays per-frame
  (a drifting cell is still correct on every frame);
- there are no keyframes, no interpolation, no birth/end/death, no class
  changes;
- annotation is restricted to the first ``n_frames`` frames, since the
  detector we're training must recognise singles/doublets *early*.

It deliberately exposes the same ColumnDataSource row contract as
``DashboardState.boxes_at`` (``id, num, label, cx, cy, w, h, marker``) so
the shared figure in ``common.py`` renders either controller unchanged.
"""
from __future__ import annotations

import uuid

from ..schema_simple import (
    PROVISIONAL_LABEL,
    SimpleAnnotationFile,
    SimpleBox,
)
from .state import bbox_to_cwh, cwh_to_bbox


class SimpleState:
    def __init__(self, annotation_file: SimpleAnnotationFile, n_t: int,
                 n_frames: int | None = None):
        self.af = annotation_file
        self.n_t = max(1, int(n_t))
        requested = annotation_file.n_frames if n_frames is None else n_frames
        self.n_frames = self._clamp_n_frames(requested)
        self.current_t = 0
        self._by_id: dict[str, SimpleBox] = {}
        self._order: list[str] = []
        for b in annotation_file.boxes:
            self._register(b)

    # ---- helpers -------------------------------------------------------
    def _clamp_n_frames(self, n: int) -> int:
        return max(1, min(int(n), self.n_t))

    def _register(self, box: SimpleBox) -> str:
        box_id = uuid.uuid4().hex
        self._by_id[box_id] = box
        self._order.append(box_id)
        return box_id

    def _t(self, t: int | None) -> int:
        return self.current_t if t is None else int(t)

    # ---- cell identity (light tracking) --------------------------------
    def group_of(self, box_id: str) -> str:
        """Effective group of a box. A standalone box is its own singleton
        group, so every box belongs to exactly one 'cell'."""
        return self._by_id[box_id].group or box_id

    def ids_in_group(self, box_id: str) -> list[str]:
        """Every box id belonging to the same cell, in insertion order."""
        gid = self.group_of(box_id)
        return [i for i in self._order if self.group_of(i) == gid]

    def _group_numbers(self) -> dict[str, int]:
        """Stable 1-based number per cell, in the order cells first appear.

        Derived from insertion order (no extra state), so a cell keeps its
        number while the annotator scrubs frames — the identity feedback that
        makes 'classify once' legible.
        """
        nums: dict[str, int] = {}
        for i in self._order:
            gid = self.group_of(i)
            if gid not in nums:
                nums[gid] = len(nums) + 1
        return nums

    def scope_ids(self, box_id: str, scope: str = "cell",
                  t: int | None = None) -> list[str]:
        """Box ids an action should apply to.

        - ``"frame"`` — only this box (this frame).
        - ``"cell"`` — every box of this cell, on all frames (the default:
          classify once).
        - ``"forward"`` — this cell from frame ``t`` onward, for a cell whose
          class changes mid-window (e.g. a single that divides).
        """
        if scope == "frame":
            return [box_id]
        ids = self.ids_in_group(box_id)
        if scope == "cell":
            return ids
        if scope == "forward":
            t0 = self._by_id[box_id].t if t is None else int(t)
            return [i for i in ids if self._by_id[i].t >= t0]
        raise ValueError(f"unknown scope {scope!r}")

    @property
    def max_t(self) -> int:
        """Last annotatable frame (annotation is limited to the first N)."""
        return self.n_frames - 1

    @property
    def classes(self) -> list[str]:
        return list(self.af.classes)

    def box(self, box_id: str) -> SimpleBox:
        return self._by_id[box_id]

    def has(self, box_id: str) -> bool:
        return box_id in self._by_id

    def boxes(self) -> list[SimpleBox]:
        return [self._by_id[i] for i in self._order]

    # ---- frame range ---------------------------------------------------
    def set_t(self, t: int) -> None:
        """Set the current frame, clamped into the annotatable range."""
        self.current_t = max(0, min(int(t), self.max_t))

    def set_n_frames(self, n: int) -> int:
        """Change how many leading frames are annotated.

        Returns the number of already-annotated boxes that now fall OUTSIDE
        the range. They are kept (never silently discarded — that would throw
        away work), just not shown; the view warns about them and
        ``drop_out_of_range`` removes them on an explicit user action.
        """
        self.n_frames = self._clamp_n_frames(n)
        if self.current_t > self.max_t:
            self.current_t = self.max_t
        return self.out_of_range_count()

    def out_of_range_count(self) -> int:
        return sum(1 for b in self._by_id.values() if b.t > self.max_t)

    def drop_out_of_range(self) -> int:
        """Delete boxes beyond the frame limit. Returns how many went."""
        doomed = [i for i, b in self._by_id.items() if b.t > self.max_t]
        for i in doomed:
            self.delete(i)
        return len(doomed)

    # ---- read ----------------------------------------------------------
    def boxes_at(self, t: int | None = None) -> list[dict]:
        """CDS rows for frame ``t`` — only boxes that belong to that frame.

        ``num`` numbers the boxes within this frame (they're independent, so
        there is no cross-frame identity to preserve). ``marker`` shows the
        detector score for boxes that came from a model, so the annotator can
        triage low-confidence ones; it's empty for hand-drawn boxes.
        """
        t = self._t(t)
        if t > self.max_t or t < 0:
            # Outside the annotatable range: boxes there are kept in the model
            # (shrinking N must not destroy work) but are never shown.
            return []
        nums = self._group_numbers()
        rows: list[dict] = []
        for i in self._order:
            b = self._by_id[i]
            if b.t != t:
                continue
            cx, cy, w, h = bbox_to_cwh(b.bbox)
            rows.append({
                "id": i,
                "num": nums[self.group_of(i)],
                "label": b.label,
                "cx": cx,
                "cy": cy,
                "w": w,
                "h": h,
                "marker": "" if b.score is None else f"{b.score:.2f}",
            })
        return rows

    # ---- write: geometry -----------------------------------------------
    def add_box(self, cx: float, cy: float, w: float, h: float,
                label: str, t: int | None = None,
                score: float | None = None, auto: bool = False,
                group: str | None = None) -> str:
        """Create a box on frame ``t`` (defaults to the current frame).

        ``auto=True`` marks it as raw detector output — see ``set_detections``.
        """
        t = self._t(t)
        return self._register(
            SimpleBox(t=t, bbox=cwh_to_bbox(cx, cy, w, h), label=label,
                      score=score, auto=auto, group=group)
        )

    def update_box(self, box_id: str, cx: float, cy: float,
                   w: float, h: float) -> None:
        """Move/resize a box. Frame-local — no keyframes to reconcile.

        Moving a box counts as human vetting, so it stops being ``auto``.
        """
        b = self._by_id[box_id]
        b.bbox = cwh_to_bbox(cx, cy, w, h)
        b.auto = False

    def delete(self, box_id: str, scope: str = "frame",
               t: int | None = None) -> int:
        """Delete this box (default) or the whole cell (``scope="cell"``).

        Returns how many boxes were removed.
        """
        ids = self.scope_ids(box_id, scope, t) if scope != "frame" else [box_id]
        for i in ids:
            self._by_id.pop(i, None)
            if i in self._order:
                self._order.remove(i)
        return len(ids)

    def propagate_forward(self, box_id: str) -> int:
        """Copy this box to the later frames of the window as the same cell.

        For a hand-drawn box (or frames the detector missed): the class and
        geometry are copied to every later frame in range that this cell
        doesn't already occupy, sharing one ``group`` so a later
        re-classification still hits all of them at once. Individual frames
        can then be nudged to follow drift. Returns how many were added.
        """
        src = self._by_id[box_id]
        gid = src.group
        if gid is None:
            gid = uuid.uuid4().hex
            src.group = gid
        taken = {self._by_id[i].t for i in self.ids_in_group(box_id)}
        added = 0
        for t in range(src.t + 1, self.max_t + 1):
            if t in taken:
                continue
            self._register(SimpleBox(t=t, bbox=list(src.bbox),
                                     label=src.label, z=src.z,
                                     auto=False, group=gid))
            added += 1
        return added

    def resize(self, box_id: str, w: float, h: float) -> None:
        """Set width/height, keeping the centre (counts as human vetting)."""
        b = self._by_id[box_id]
        cx, cy, _, _ = bbox_to_cwh(b.bbox)
        b.bbox = cwh_to_bbox(cx, cy, max(1.0, float(w)), max(1.0, float(h)))
        b.auto = False

    def scale(self, box_id: str, factor: float) -> None:
        b = self._by_id[box_id]
        cx, cy, w, h = bbox_to_cwh(b.bbox)
        self.resize(box_id, w * factor, h * factor)

    def size_of(self, box_id: str) -> tuple[float, float]:
        _, _, w, h = bbox_to_cwh(self._by_id[box_id].bbox)
        return w, h

    # ---- write: classification -----------------------------------------
    def set_label(self, box_id: str, label: str, scope: str = "cell",
                  t: int | None = None) -> int:
        """Classify a box and, by default, the whole cell across frames.

        This is the point of the light tracking: the annotator picks a class
        once per cell rather than once per frame. Use ``scope="frame"`` for a
        one-off correction, or ``scope="forward"`` when a cell changes class
        part-way through the window. Returns how many boxes changed.
        """
        ids = self.scope_ids(box_id, scope, t)
        for i in ids:
            b = self._by_id[i]
            b.label = label
            b.auto = False  # a human chose this class
        if label not in self.af.classes:
            self.af.classes.append(label)
        return len(ids)

    # ---- view reconciliation -------------------------------------------
    def apply_cds_edits(self, rows: list[dict], default_label: str,
                        t: int | None = None, *, eps: float = 0.5) -> list[str]:
        """Reconcile BoxEditTool rows into the model at frame ``t``.

        New row (no id) -> ``add_box``; moved row -> ``update_box``; a box
        that was on this frame but is gone from ``rows`` -> ``delete``.
        Returns the row ids in order so the view can write them back.
        """
        t = self._t(t)
        present_before = {r["id"] for r in self.boxes_at(t)}
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
                rid = self.add_box(r["cx"], r["cy"], r["w"], r["h"],
                                   label=r.get("label") or default_label, t=t)
            else:
                cx, cy, w, h = bbox_to_cwh(self._by_id[rid].bbox)
                moved = (abs(cx - r["cx"]) > eps or abs(cy - r["cy"]) > eps
                         or abs(w - r["w"]) > eps or abs(h - r["h"]) > eps)
                if moved:
                    self.update_box(rid, r["cx"], r["cy"], r["w"], r["h"])
            seen.add(rid)
            out_ids.append(rid)
        for rid in present_before - seen:
            self.delete(rid)
        return out_ids

    # ---- detection -----------------------------------------------------
    def set_detections(self, boxes: list[SimpleBox],
                       t_range: tuple[int, int] | None = None) -> int:
        """Insert fresh detections, keeping everything a human touched.

        Only ``auto`` boxes (raw detector output nobody has classified,
        moved, or resized) — and, when ``t_range`` is given, only those
        inside it — are cleared out. So re-running detection refreshes
        untriaged boxes and can never destroy curation work, whatever class
        those boxes carry. Returns how many boxes were added.
        """
        def is_stale(b: SimpleBox) -> bool:
            if not b.auto:
                return False  # human-touched: keep
            if t_range is None:
                return True
            return t_range[0] <= b.t < t_range[1]

        keep = [i for i in self._order if not is_stale(self._by_id[i])]
        self._by_id = {i: self._by_id[i] for i in keep}
        self._order = keep
        added = 0
        for b in boxes:
            if b.t <= self.max_t:
                self._register(b)
                added += 1
        return added

    # ---- summaries / persistence ---------------------------------------
    def counts(self) -> dict[str, int]:
        c: dict[str, int] = {}
        for b in self._by_id.values():
            c[b.label] = c.get(b.label, 0) + 1
        return c

    def counts_at(self, t: int | None = None) -> dict[str, int]:
        t = self._t(t)
        c: dict[str, int] = {}
        for b in self._by_id.values():
            if b.t == t:
                c[b.label] = c.get(b.label, 0) + 1
        return c

    def n_cells(self) -> int:
        """Distinct cells (groups), not boxes."""
        return len(self._group_numbers())

    def cell_counts(self) -> dict[str, int]:
        """Distinct CELLS per class — the number that reflects how much
        genuinely independent data there is (one cell across 20 frames is 20
        boxes but one cell)."""
        seen: dict[str, set] = {}
        for i in self._order:
            seen.setdefault(self._by_id[i].label, set()).add(self.group_of(i))
        return {label: len(g) for label, g in seen.items()}

    def unlabeled_cells(self) -> int:
        return self.cell_counts().get(PROVISIONAL_LABEL, 0)

    def unlabeled_count(self) -> int:
        """Boxes still awaiting a human class — annotation progress."""
        return sum(1 for b in self._by_id.values()
                   if b.label == PROVISIONAL_LABEL)

    def frames_with_boxes(self) -> list[int]:
        return sorted({b.t for b in self._by_id.values()})

    def sync_to_file(self) -> SimpleAnnotationFile:
        """Rebuild ``af.boxes`` (frame-ordered) for saving."""
        self.af.boxes = sorted(
            (self._by_id[i] for i in self._order),
            key=lambda b: (b.t, b.bbox[0], b.bbox[1]),
        )
        self.af.n_frames = self.n_frames
        return self.af
