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


def test_num_is_per_frame():
    st = _st()
    st.add_box(10, 10, 4, 4, "single", t=0)
    st.add_box(20, 20, 4, 4, "single", t=0)
    st.add_box(30, 30, 4, 4, "single", t=1)
    assert [r["num"] for r in st.boxes_at(0)] == [1, 2]
    assert [r["num"] for r in st.boxes_at(1)] == [1]


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
