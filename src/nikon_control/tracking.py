"""Link per-frame detections into tracks across time.

A *track* is one object followed through consecutive frames. Each track
becomes one :class:`~nikon_control.schema.Annotation` whose keyframes are
the (simplified) trajectory of its bounding box.

``IoUTracker`` is the default: greedy/optimal box-overlap association
(SORT-style, minus the Kalman filter — unnecessary when cells barely
move). It is deterministic and dependency-light. The ``Tracker`` protocol
lets a heavier tracker (e.g. for deforming debris) drop in later without
touching the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

from .detector import Detection
from .schema import Annotation, Keyframe


@dataclass
class Track:
    frames: list[int] = field(default_factory=list)
    bboxes: list[list[float]] = field(default_factory=list)  # [y0,x0,y1,x1]
    scores: list[float] = field(default_factory=list)

    @property
    def t_start(self) -> int:
        return self.frames[0]

    @property
    def t_last(self) -> int:
        return self.frames[-1]


class Tracker(Protocol):
    def track(
        self, per_frame: Sequence[Sequence[Detection]]
    ) -> list[Track]:
        """Link detections (indexed by frame) into tracks."""
        ...


def iou(a: list[float], b: list[float]) -> float:
    """IoU of two ``[y0, x0, y1, x1]`` boxes."""
    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b
    iy0, ix0 = max(ay0, by0), max(ax0, bx0)
    iy1, ix1 = min(ay1, by1), min(ax1, bx1)
    ih, iw = max(0.0, iy1 - iy0), max(0.0, ix1 - ix0)
    inter = ih * iw
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ay1 - ay0) * max(0.0, ax1 - ax0)
    area_b = max(0.0, by1 - by0) * max(0.0, bx1 - bx0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class IoUTracker:
    """Associate detections frame-to-frame by IoU (optimal assignment).

    Parameters
    ----------
    iou_threshold:
        Minimum overlap to link a detection to an existing track.
    max_age:
        How many consecutive frames a track may go undetected before it is
        closed. >0 bridges the occasional missed detection so a track isn't
        split in two.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 3):
        self.iou_threshold = iou_threshold
        self.max_age = max_age

    def track(self, per_frame: Sequence[Sequence[Detection]]) -> list[Track]:
        try:
            from scipy.optimize import linear_sum_assignment
            _have_scipy = True
        except ImportError:
            _have_scipy = False

        finished: list[Track] = []
        # active: list of dicts {track, last_bbox, misses}
        active: list[dict] = []

        for t, dets in enumerate(per_frame):
            dets = list(dets)
            if active and dets:
                cost = [[1.0 - iou(a["last_bbox"], d.bbox) for d in dets]
                        for a in active]
                if _have_scipy:
                    import numpy as np

                    rows, cols = linear_sum_assignment(np.array(cost))
                    pairs = list(zip(rows.tolist(), cols.tolist()))
                else:
                    pairs = _greedy_assign(cost)
            else:
                pairs = []

            matched_tracks: set[int] = set()
            matched_dets: set[int] = set()
            for ai, di in pairs:
                if iou(active[ai]["last_bbox"], dets[di].bbox) >= self.iou_threshold:
                    tr = active[ai]["track"]
                    tr.frames.append(t)
                    tr.bboxes.append(dets[di].bbox)
                    tr.scores.append(dets[di].score)
                    active[ai]["last_bbox"] = dets[di].bbox
                    active[ai]["misses"] = 0
                    matched_tracks.add(ai)
                    matched_dets.add(di)

            # age / close unmatched tracks
            still_active: list[dict] = []
            for ai, a in enumerate(active):
                if ai in matched_tracks:
                    still_active.append(a)
                else:
                    a["misses"] += 1
                    if a["misses"] > self.max_age:
                        finished.append(a["track"])
                    else:
                        still_active.append(a)
            active = still_active

            # start new tracks for unmatched detections
            for di, d in enumerate(dets):
                if di in matched_dets:
                    continue
                tr = Track(frames=[t], bboxes=[d.bbox], scores=[d.score])
                active.append({"track": tr, "last_bbox": d.bbox, "misses": 0})

        finished.extend(a["track"] for a in active)
        # tracks come out in closing order; sort by start frame for stability
        finished.sort(key=lambda tr: (tr.t_start, tr.bboxes[0][0], tr.bboxes[0][1]))
        return finished


def _center(bbox: list[float]) -> tuple[float, float]:
    y0, x0, y1, x1 = bbox
    return ((y0 + y1) / 2.0, (x0 + x1) / 2.0)


class VelocityTracker:
    """Associate detections by predicted position — for fast movers (debris).

    IoU fails when an object moves further than its own size per frame (zero
    overlap). This tracker predicts each track's next centre as
    ``last_centre + velocity`` and matches the nearest detection within
    ``max_dist`` of that prediction, so a debris particle crossing the FOV
    in a straight line links with a modest gate even at hundreds of px/frame.

    Parameters
    ----------
    max_dist: max distance (px) between a detection and a track's predicted
        centre to link them.
    max_age: frames a track may go unmatched before it closes.
    """

    def __init__(
        self,
        max_dist: float = 150.0,
        max_age: int = 2,
        base_gate: float = 100.0,
        vel_factor: float = 2.0,
    ):
        # Gating is velocity-adaptive to avoid the "static debris that jumps"
        # artefact: an established track's match gate is
        # ``base_gate + vel_factor*speed`` (capped at ``max_dist``), so a
        # stationary blob (speed≈0) uses a tight ``base_gate`` and cannot grab
        # a far transient when its own detection momentarily drops, while a
        # genuinely fast track keeps a wide gate. A brand-new track (velocity
        # not yet known) uses the full ``max_dist`` so a fast crosser's first
        # step can still link.
        self.max_dist = max_dist
        self.max_age = max_age
        self.base_gate = base_gate
        self.vel_factor = vel_factor

    def _gate(self, a: dict) -> float:
        import math

        if len(a["track"].frames) < 2:  # velocity not yet established
            return self.max_dist
        speed = math.hypot(a["vy"], a["vx"])
        return min(self.max_dist, self.base_gate + self.vel_factor * speed)

    def track(self, per_frame: Sequence[Sequence[Detection]]) -> list[Track]:
        import math

        try:
            from scipy.optimize import linear_sum_assignment
            _have_scipy = True
        except ImportError:
            _have_scipy = False

        finished: list[Track] = []
        active: list[dict] = []  # {track, cy, cx, vy, vx, misses}

        for t, dets in enumerate(per_frame):
            dets = list(dets)
            centers = [_center(d.bbox) for d in dets]
            pairs: list[tuple[int, int]] = []
            if active and dets:
                cost = []
                for a in active:
                    py, px = a["cy"] + a["vy"], a["cx"] + a["vx"]
                    cost.append([math.dist((py, px), c) for c in centers])
                if _have_scipy:
                    import numpy as np

                    rows, cols = linear_sum_assignment(np.array(cost))
                    pairs = list(zip(rows.tolist(), cols.tolist()))
                else:
                    pairs = _greedy_assign(cost)

            matched_a: set[int] = set()
            matched_d: set[int] = set()
            for ai, di in pairs:
                py, px = active[ai]["cy"] + active[ai]["vy"], active[ai]["cx"] + active[ai]["vx"]
                if math.dist((py, px), centers[di]) <= self._gate(active[ai]):
                    a = active[ai]
                    cy, cx = centers[di]
                    a["vy"], a["vx"] = cy - a["cy"], cx - a["cx"]
                    a["cy"], a["cx"] = cy, cx
                    a["misses"] = 0
                    a["track"].frames.append(t)
                    a["track"].bboxes.append(dets[di].bbox)
                    a["track"].scores.append(dets[di].score)
                    matched_a.add(ai)
                    matched_d.add(di)

            still: list[dict] = []
            for ai, a in enumerate(active):
                if ai in matched_a:
                    still.append(a)
                else:
                    a["misses"] += 1
                    if a["misses"] > self.max_age:
                        finished.append(a["track"])
                    else:
                        # coast on last velocity while unmatched
                        a["cy"] += a["vy"]
                        a["cx"] += a["vx"]
                        still.append(a)
            active = still

            for di, d in enumerate(dets):
                if di in matched_d:
                    continue
                cy, cx = centers[di]
                active.append({
                    "track": Track(frames=[t], bboxes=[d.bbox], scores=[d.score]),
                    "cy": cy, "cx": cx, "vy": 0.0, "vx": 0.0, "misses": 0,
                })

        finished.extend(a["track"] for a in active)
        finished.sort(key=lambda tr: (tr.t_start, tr.bboxes[0][0], tr.bboxes[0][1]))
        return finished


def _greedy_assign(cost: list[list[float]]) -> list[tuple[int, int]]:
    """Fallback assignment when scipy is unavailable: greedily take the
    lowest-cost (track, det) pairs without reuse."""
    triples = sorted(
        ((cost[a][d], a, d) for a in range(len(cost)) for d in range(len(cost[0]))),
        key=lambda x: x[0],
    )
    used_a: set[int] = set()
    used_d: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, a, d in triples:
        if a in used_a or d in used_d:
            continue
        used_a.add(a)
        used_d.add(d)
        pairs.append((a, d))
    return pairs


def _rdp_simplify(
    frames: list[int], bboxes: list[list[float]], tol: float
) -> list[int]:
    """Ramer–Douglas–Peucker on a bbox trajectory.

    Returns the indices to keep. A keyframe survives if dropping it would
    move the linearly-interpolated bbox at its frame by more than ``tol``
    pixels in any of the four coordinates. Static objects collapse to their
    two endpoints (then dedup to one); moving objects keep their inflections.
    """
    n = len(frames)
    if n <= 2:
        return list(range(n))
    keep = {0, n - 1}

    def recurse(lo: int, hi: int) -> None:
        t0, t1 = frames[lo], frames[hi]
        b0, b1 = bboxes[lo], bboxes[hi]
        worst, worst_i = 0.0, -1
        for i in range(lo + 1, hi):
            if t1 == t0:
                dev = 0.0
            else:
                alpha = (frames[i] - t0) / (t1 - t0)
                dev = max(
                    abs(bboxes[i][j] - (b0[j] + alpha * (b1[j] - b0[j])))
                    for j in range(4)
                )
            if dev > worst:
                worst, worst_i = dev, i
        if worst > tol and worst_i != -1:
            keep.add(worst_i)
            recurse(lo, worst_i)
            recurse(worst_i, hi)

    recurse(0, n - 1)
    return sorted(keep)


def tracks_to_annotations(
    tracks: Sequence[Track],
    *,
    label: str,
    n_frames: int,
    min_len: int = 1,
    simplify_tol: float = 2.0,
) -> list[Annotation]:
    """Convert tracks into schema Annotations.

    - Tracks shorter than ``min_len`` frames are dropped (likely noise).
    - Keyframes are RDP-simplified with ``simplify_tol`` px so static objects
      collapse toward a single keyframe.
    - ``t_end`` is set to the last detected frame when the track ends before
      the recording does (object left/died); left ``None`` if it runs to the
      end (visible throughout).
    """
    anns: list[Annotation] = []
    for tr in tracks:
        if len(tr.frames) < min_len:
            continue
        keep = _rdp_simplify(tr.frames, tr.bboxes, simplify_tol)
        kfs = [Keyframe(t=tr.frames[i], bbox=list(tr.bboxes[i])) for i in keep]
        # collapse a static run (endpoints nearly identical) to one keyframe
        if len(kfs) == 2 and _bbox_max_dev(kfs[0].bbox, kfs[1].bbox) <= simplify_tol:
            kfs = [kfs[0]]
        t_end = None if tr.t_last >= n_frames - 1 else tr.t_last
        anns.append(
            Annotation(
                label=label,
                keyframes=kfs,
                t_start=tr.t_start,
                t_end=t_end,
            )
        )
    return anns


def _bbox_max_dev(a: list[float], b: list[float]) -> float:
    return max(abs(a[j] - b[j]) for j in range(4))
