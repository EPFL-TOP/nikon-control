"""Compare model predictions against annotations — GUI-free and pure.

Used by the review dashboard to point attention at the images most likely to
be wrong: where the model and the annotation disagree on **how many** objects
there are, or on **what class** they are.

A disagreement is not automatically an annotation error — it can equally be a
model error. That is the point: both are worth a human look, and they are the
only images where looking can change anything.

Matching is greedy by IoU (best pair first), which is the standard detection
convention: each annotation matches at most one prediction and vice versa.
IoU is right here — unlike duplicate suppression, this asks "are these the
same object?", not "does one contain the other".
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .overlap import bbox_iou

DEFAULT_MATCH_IOU = 0.5


@dataclass
class Box:
    """One box to compare — an annotation or a prediction."""

    bbox: list[float]  # [y0, x0, y1, x1]
    label: str
    score: float | None = None


@dataclass
class Agreement:
    """How one image's annotations and predictions line up."""

    n_gt: int = 0
    n_pred: int = 0
    # (gt index, pred index) pairs that matched geometrically
    matched: list[tuple[int, int]] = field(default_factory=list)
    # matched pairs whose CLASS differs, with both labels
    class_mismatch: list[tuple[int, int, str, str]] = field(default_factory=list)
    # annotations the model did not find / predictions with no annotation
    missed: list[int] = field(default_factory=list)
    extra: list[int] = field(default_factory=list)

    @property
    def n_agree(self) -> int:
        """Matched pairs that also agree on the class."""
        return len(self.matched) - len(self.class_mismatch)

    @property
    def count_delta(self) -> int:
        """Predicted minus annotated box count (0 = same number)."""
        return self.n_pred - self.n_gt

    @property
    def score(self) -> int:
        """Total discrepancies — 0 means full agreement.

        Class mismatches, missed annotations and spurious predictions all
        count once, so sorting by this puts the worst images first.
        """
        return len(self.class_mismatch) + len(self.missed) + len(self.extra)

    @property
    def agrees(self) -> bool:
        return self.score == 0

    def reasons(self) -> list[str]:
        """Short human-readable reasons, for a status line."""
        out = []
        for _, _, gt_label, pred_label in self.class_mismatch:
            out.append(f"annotated {gt_label} → model says {pred_label}")
        if self.missed:
            out.append(f"{len(self.missed)} annotated box(es) the model missed")
        if self.extra:
            out.append(f"{len(self.extra)} box(es) the model found but "
                       "nobody annotated")
        return out


def match_boxes(gt: list[Box], pred: list[Box],
                iou_threshold: float = DEFAULT_MATCH_IOU
                ) -> list[tuple[int, int, float]]:
    """Greedy one-to-one matching by IoU, best pair first.

    Returns ``(gt_index, pred_index, iou)`` triples. Deterministic: ties fall
    back to the input order.
    """
    pairs = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            v = bbox_iou(g.bbox, p.bbox)
            if v >= iou_threshold:
                pairs.append((gi, pi, v))
    pairs.sort(key=lambda t: (-t[2], t[0], t[1]))
    used_g: set[int] = set()
    used_p: set[int] = set()
    out = []
    for gi, pi, v in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        out.append((gi, pi, v))
    return out


def compare(gt: list[Box], pred: list[Box],
            iou_threshold: float = DEFAULT_MATCH_IOU) -> Agreement:
    """Compare one image's annotations against one model's predictions."""
    matches = match_boxes(gt, pred, iou_threshold)
    ag = Agreement(n_gt=len(gt), n_pred=len(pred))
    for gi, pi, _ in matches:
        ag.matched.append((gi, pi))
        if gt[gi].label != pred[pi].label:
            ag.class_mismatch.append((gi, pi, gt[gi].label, pred[pi].label))
    matched_g = {gi for gi, _ in ag.matched}
    matched_p = {pi for _, pi in ag.matched}
    ag.missed = [i for i in range(len(gt)) if i not in matched_g]
    ag.extra = [i for i in range(len(pred)) if i not in matched_p]
    return ag


def confusion(agreements: list[tuple[str, Agreement]]) -> dict[tuple[str, str], int]:
    """Dataset-level ``(annotated, predicted) -> count`` for mismatches."""
    out: dict[tuple[str, str], int] = {}
    for _, ag in agreements:
        for _, _, gt_label, pred_label in ag.class_mismatch:
            key = (gt_label, pred_label)
            out[key] = out.get(key, 0) + 1
    return out


def totals(agreements: list[tuple[str, Agreement]]) -> dict[str, int]:
    """Aggregate counts across a dataset."""
    t = {"images": 0, "images_disagreeing": 0, "agree": 0,
         "class_mismatch": 0, "missed": 0, "extra": 0,
         "annotated": 0, "predicted": 0}
    for _, ag in agreements:
        t["images"] += 1
        if not ag.agrees:
            t["images_disagreeing"] += 1
        t["agree"] += ag.n_agree
        t["class_mismatch"] += len(ag.class_mismatch)
        t["missed"] += len(ag.missed)
        t["extra"] += len(ag.extra)
        t["annotated"] += ag.n_gt
        t["predicted"] += ag.n_pred
    return t
