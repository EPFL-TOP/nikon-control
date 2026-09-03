"""Tests for SimpleState — the per-frame (untracked) controller."""
from nikon_control.dashboard.simple_state import SimpleState
from nikon_control.schema_simple import (
    PROVISIONAL_LABEL,
    SimpleAnnotationFile,
    SimpleBox,
)


def _st(boxes=None, n_t=100, n_frames=20):
    af = SimpleAnnotationFile(source="x.nd2", n_frames=n_frames,
                              boxes=boxes or [])
    return SimpleState(af, n_t=n_t)


def test_boxes_are_frame_local():
    """A box on frame 0 must NOT appear on frame 1 — no tracking."""
    st = _st()
    st.set_t(0)
    st.add_box(100, 100, 40, 40, "single")
    assert len(st.boxes_at(0)) == 1
    assert st.boxes_at(1) == []
    assert st.boxes_at(5) == []


def test_same_cell_two_frames_are_independent():
    st = _st()
    i0 = st.add_box(100, 100, 40, 40, "single", t=0)
    i1 = st.add_box(100, 100, 40, 40, "single", t=1)
    assert i0 != i1
    st.set_label(i1, "doublet")
    assert st.boxes_at(0)[0]["label"] == "single"
    assert st.boxes_at(1)[0]["label"] == "doublet"


def test_num_is_per_cell_and_stable_across_frames():
    """A cell keeps its number while scrubbing — the identity feedback that
    makes 'classify once per cell' legible."""
    st = _st()
    st.add_box(10, 10, 4, 4, "single", t=0, group="g1")
    st.add_box(20, 20, 4, 4, "single", t=0)          # standalone -> own cell
    st.add_box(11, 11, 4, 4, "single", t=1, group="g1")  # same cell, later frame
    assert [r["num"] for r in st.boxes_at(0)] == [1, 2]
    assert [r["num"] for r in st.boxes_at(1)] == [1]  # cell 1 again, not 3


def test_classify_once_applies_to_the_whole_cell():
    """The whole point of the light tracking: one click labels every frame."""
    st = _st()
    ids = [st.add_box(10, 10, 4, 4, PROVISIONAL_LABEL, t=t, group="g1",
                      auto=True) for t in range(5)]
    solo = st.add_box(99, 99, 4, 4, PROVISIONAL_LABEL, t=0, auto=True)
    n = st.set_label(ids[2], "doublet")            # click any frame
    assert n == 5
    assert all(st.box(i).label == "doublet" for i in ids)
    assert st.box(solo).label == PROVISIONAL_LABEL  # other cells untouched
    assert st.unlabeled_count() == 1


def test_scope_frame_and_forward():
    st = _st()
    ids = [st.add_box(10, 10, 4, 4, "single", t=t, group="g1")
           for t in range(6)]
    # a one-off correction on a single frame
    assert st.set_label(ids[0], "debris", scope="frame") == 1
    assert [st.box(i).label for i in ids] == (
        ["debris"] + ["single"] * 5)
    # a cell that divides mid-window: doublet from frame 3 on
    assert st.set_label(ids[3], "doublet", scope="forward") == 3
    assert [st.box(i).label for i in ids] == (
        ["debris", "single", "single", "doublet", "doublet", "doublet"])


def test_geometry_stays_per_frame_within_a_cell():
    """Moving a box must not move its siblings — cells drift."""
    st = _st()
    a = st.add_box(10, 10, 4, 4, "single", t=0, group="g1")
    b = st.add_box(10, 10, 4, 4, "single", t=1, group="g1")
    st.update_box(b, 80, 80, 4, 4)
    assert st.boxes_at(0)[0]["cx"] == 10
    assert st.boxes_at(1)[0]["cx"] == 80
    assert st.group_of(a) == st.group_of(b)


def test_delete_scope_frame_vs_cell():
    st = _st()
    ids = [st.add_box(10, 10, 4, 4, "single", t=t, group="g1")
           for t in range(4)]
    assert st.delete(ids[1]) == 1                    # just that frame
    assert len(st.boxes()) == 3
    assert st.delete(ids[0], scope="cell") == 3      # the whole cell
    assert st.boxes() == []


def test_propagate_forward_shares_one_cell():
    st = _st(n_frames=5)
    i = st.add_box(10, 10, 4, 4, "single", t=0)
    assert st.propagate_forward(i) == 4              # frames 1..4
    assert [b.t for b in st.boxes()] == [0, 1, 2, 3, 4]
    gids = {b.group for b in st.boxes()}
    assert len(gids) == 1 and None not in gids
    # and a later re-classification still hits all of them
    assert st.set_label(i, "doublet") == 5
    assert {b.label for b in st.boxes()} == {"doublet"}


def test_propagate_forward_skips_frames_already_occupied():
    st = _st(n_frames=4)
    i = st.add_box(10, 10, 4, 4, "single", t=0, group="g1")
    st.add_box(50, 50, 4, 4, "single", t=2, group="g1")   # already there
    assert st.propagate_forward(i) == 2                   # only t=1 and t=3
    assert st.boxes_at(2)[0]["cx"] == 50                  # not overwritten


def test_frame_limit_clamps_navigation():
    st = _st(n_frames=20)
    assert st.max_t == 19
    st.set_t(50)
    assert st.current_t == 19
    st.set_t(-5)
    assert st.current_t == 0


def test_n_frames_clamped_to_recording_length():
    st = _st(n_t=5, n_frames=20)
    assert st.n_frames == 5 and st.max_t == 4


def test_shrinking_range_keeps_boxes_but_reports_them():
    """Reducing N must not silently destroy annotation work."""
    st = _st(n_frames=20)
    st.add_box(10, 10, 4, 4, "single", t=0)
    st.add_box(10, 10, 4, 4, "single", t=15)
    far = st.set_n_frames(10)
    assert far == 1                      # reported
    assert len(st.boxes()) == 2          # but kept
    assert st.boxes_at(15) == []         # just not shown
    assert st.drop_out_of_range() == 1   # explicit removal
    assert len(st.boxes()) == 1


def test_marker_shows_score_only_for_auto_boxes():
    st = _st()
    st.add_box(10, 10, 4, 4, PROVISIONAL_LABEL, t=0, score=0.873)
    st.add_box(20, 20, 4, 4, "single", t=0)
    rows = st.boxes_at(0)
    assert rows[0]["marker"] == "0.87"
    assert rows[1]["marker"] == ""


def test_apply_cds_edits_adds_moves_deletes_on_this_frame_only():
    st = _st()
    other = st.add_box(50, 50, 10, 10, "single", t=1)  # different frame
    i = st.add_box(100, 100, 40, 40, "single", t=0)
    st.set_t(0)
    # move the existing box and add a brand-new one
    ids = st.apply_cds_edits(
        [{"id": i, "cx": 120, "cy": 100, "w": 40, "h": 40},
         {"id": "", "cx": 300, "cy": 300, "w": 20, "h": 20}],
        default_label=PROVISIONAL_LABEL,
    )
    assert len(ids) == 2
    assert st.boxes_at(0)[0]["cx"] == 120
    assert st.boxes_at(0)[1]["label"] == PROVISIONAL_LABEL
    # the box on frame 1 is untouched by an edit at frame 0
    assert st.has(other)
    # dropping a row deletes it
    st.apply_cds_edits([{"id": i, "cx": 120, "cy": 100, "w": 40, "h": 40}],
                       default_label=PROVISIONAL_LABEL)
    assert len(st.boxes_at(0)) == 1
    assert st.has(other)


def _auto(t, label=PROVISIONAL_LABEL):
    return SimpleBox(t=t, bbox=[0, 0, 5, 5], label=label, auto=True)


def test_set_detections_keeps_human_labels():
    st = _st()
    keep = st.add_box(10, 10, 4, 4, "doublet", t=0)  # human-classified
    st.add_box(20, 20, 4, 4, PROVISIONAL_LABEL, t=0, auto=True)  # untriaged
    added = st.set_detections([_auto(0), _auto(1)], t_range=(0, 20))
    assert added == 2
    assert st.has(keep)
    labels = [b.label for b in st.boxes()]
    assert labels.count("doublet") == 1
    assert labels.count(PROVISIONAL_LABEL) == 2  # old untriaged one replaced


def test_redetect_never_destroys_a_human_drawn_box_of_the_same_class():
    """Re-running debris detection must not wipe hand-drawn debris.

    Provenance (auto), not the label, decides what a refresh may replace —
    otherwise auto-labelled 'debris' and human 'debris' are indistinguishable.
    """
    st = _st()
    mine = st.add_box(10, 10, 4, 4, "debris", t=0)          # hand-drawn
    st.add_box(20, 20, 4, 4, "debris", t=0, auto=True)      # detector output
    added = st.set_detections([_auto(0, "debris")], t_range=(0, 20))
    assert added == 1
    assert st.has(mine)                     # human box survived
    assert len(st.boxes()) == 2             # auto one was replaced, not added


def test_human_actions_clear_the_auto_flag():
    st = _st()
    for action in ("label", "move", "resize"):
        i = st.add_box(10, 10, 4, 4, PROVISIONAL_LABEL, t=0, auto=True)
        assert st.box(i).auto is True
        if action == "label":
            st.set_label(i, "single")
        elif action == "move":
            st.update_box(i, 50, 50, 4, 4)
        else:
            st.resize(i, 8, 8)
        assert st.box(i).auto is False, action
        # ...and a refresh therefore leaves it alone
        st.set_detections([], t_range=(0, 20))
        assert st.has(i), action


def test_set_detections_drops_boxes_beyond_the_limit():
    st = _st(n_frames=5)
    added = st.set_detections([_auto(0), _auto(99)])
    assert added == 1


def test_counts_and_progress():
    st = _st()
    st.add_box(10, 10, 4, 4, "single", t=0)
    st.add_box(20, 20, 4, 4, "single", t=1)
    st.add_box(30, 30, 4, 4, PROVISIONAL_LABEL, t=1)
    assert st.counts()["single"] == 2
    assert st.counts_at(1)["single"] == 1
    assert st.unlabeled_count() == 1
    assert st.frames_with_boxes() == [0, 1]


def test_resize_and_scale_keep_centre():
    st = _st()
    i = st.add_box(100, 200, 40, 40, "single", t=0)
    st.resize(i, 80, 60)
    r = st.boxes_at(0)[0]
    assert (r["cx"], r["cy"], r["w"], r["h"]) == (100, 200, 80, 60)
    st.scale(i, 0.5)
    r = st.boxes_at(0)[0]
    assert (r["cx"], r["cy"], r["w"], r["h"]) == (100, 200, 40, 30)
    assert st.size_of(i) == (40, 30)


def test_sync_to_file_sorts_by_frame_and_persists_n_frames():
    st = _st(n_frames=20)
    st.add_box(10, 10, 4, 4, "single", t=7)
    st.add_box(10, 10, 4, 4, "single", t=2)
    st.set_n_frames(12)
    af = st.sync_to_file()
    assert [b.t for b in af.boxes] == [2, 7]
    assert af.n_frames == 12
