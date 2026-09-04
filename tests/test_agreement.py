"""Tests for model-vs-annotation comparison."""
from nikon_control.agreement import (
    Box,
    compare,
    confusion,
    match_boxes,
    totals,
)


def _b(y0, x0, y1, x1, label="single", score=None):
    return Box(bbox=[y0, x0, y1, x1], label=label, score=score)


def test_perfect_agreement():
    gt = [_b(0, 0, 10, 10, "single"), _b(50, 50, 60, 60, "debris")]
    pred = [_b(0, 0, 10, 10, "single"), _b(50, 50, 60, 60, "debris")]
    ag = compare(gt, pred)
    assert ag.agrees and ag.score == 0
    assert ag.n_agree == 2 and ag.count_delta == 0
    assert ag.reasons() == []


def test_class_mismatch_is_reported_with_both_labels():
    gt = [_b(0, 0, 10, 10, "single")]
    pred = [_b(0, 0, 10, 10, "doublet")]
    ag = compare(gt, pred)
    assert not ag.agrees and ag.score == 1
    assert ag.class_mismatch == [(0, 0, "single", "doublet")]
    assert ag.n_agree == 0
    assert ag.missed == [] and ag.extra == []
    assert ag.reasons() == ["annotated single → model says doublet"]


def test_missed_annotation():
    gt = [_b(0, 0, 10, 10), _b(80, 80, 90, 90)]
    pred = [_b(0, 0, 10, 10)]
    ag = compare(gt, pred)
    assert ag.missed == [1] and ag.extra == []
    assert ag.count_delta == -1 and ag.score == 1
    assert "missed" in ag.reasons()[0]


def test_extra_prediction():
    gt = [_b(0, 0, 10, 10)]
    pred = [_b(0, 0, 10, 10), _b(80, 80, 90, 90)]
    ag = compare(gt, pred)
    assert ag.extra == [1] and ag.missed == []
    assert ag.count_delta == 1 and ag.score == 1
    assert "nobody annotated" in ag.reasons()[0]


def test_matching_is_one_to_one_and_greedy():
    """Two predictions on one annotation: only the best match counts, the
    other becomes 'extra' rather than double-matching."""
    gt = [_b(0, 0, 10, 10)]
    pred = [_b(0, 0, 10, 10), _b(1, 1, 11, 11)]
    m = match_boxes(gt, pred)
    assert len(m) == 1 and m[0][:2] == (0, 0)   # the exact overlap wins
    ag = compare(gt, pred)
    assert ag.extra == [1]


def test_below_iou_threshold_counts_as_missed_plus_extra():
    gt = [_b(0, 0, 10, 10)]
    pred = [_b(8, 8, 18, 18)]           # small overlap
    ag = compare(gt, pred, iou_threshold=0.5)
    assert ag.matched == [] and ag.missed == [0] and ag.extra == [0]
    assert ag.score == 2


def test_empty_sides():
    assert compare([], []).agrees
    only_gt = compare([_b(0, 0, 10, 10)], [])
    assert only_gt.missed == [0] and only_gt.count_delta == -1
    only_pred = compare([], [_b(0, 0, 10, 10)])
    assert only_pred.extra == [0] and only_pred.count_delta == 1


def test_matching_is_deterministic():
    gt = [_b(0, 0, 10, 10), _b(0, 0, 10, 10)]
    pred = [_b(0, 0, 10, 10), _b(0, 0, 10, 10)]
    assert match_boxes(gt, pred) == match_boxes(gt, pred)


def test_confusion_and_totals_aggregate():
    a = compare([_b(0, 0, 10, 10, "single")], [_b(0, 0, 10, 10, "doublet")])
    b = compare([_b(0, 0, 10, 10, "single")], [_b(0, 0, 10, 10, "doublet")])
    c = compare([_b(0, 0, 10, 10, "debris")], [_b(0, 0, 10, 10, "debris")])
    ags = [("i1", a), ("i2", b), ("i3", c)]
    assert confusion(ags) == {("single", "doublet"): 2}
    t = totals(ags)
    assert t["images"] == 3 and t["images_disagreeing"] == 2
    assert t["class_mismatch"] == 2 and t["agree"] == 1
    assert t["annotated"] == 3 and t["predicted"] == 3


def test_score_orders_worst_first():
    mild = compare([_b(0, 0, 10, 10, "single")],
                   [_b(0, 0, 10, 10, "doublet")])
    bad = compare([_b(0, 0, 10, 10, "single"), _b(40, 40, 50, 50, "single")],
                  [_b(0, 0, 10, 10, "doublet"), _b(80, 80, 90, 90, "debris")])
    assert bad.score > mild.score
