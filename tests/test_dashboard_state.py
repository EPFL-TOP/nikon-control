from nikon_control.dashboard.state import (
    DashboardState,
    bbox_to_cwh,
    cwh_to_bbox,
)
from nikon_control.schema import Annotation, AnnotationFile, Keyframe


def _af(anns=None):
    return AnnotationFile(
        source="x.nd2",
        classes=["cell", "single", "doublet"],
        annotations=anns or [],
    )


def test_bbox_cwh_roundtrip():
    bbox = [10.0, 20.0, 110.0, 220.0]  # y0,x0,y1,x1
    cx, cy, w, h = bbox_to_cwh(bbox)
    assert (cx, cy, w, h) == (120.0, 60.0, 200.0, 100.0)
    assert cwh_to_bbox(cx, cy, w, h) == bbox


def test_add_box_keyframe_at_t_but_visible_from_start():
    st = DashboardState(_af(), n_t=100)
    st.set_t(5)
    i = st.add_box(cx=100, cy=100, w=50, h=50, label="cell")
    a = st.ann(i)
    assert a.label == "cell"
    # keyframe sits at the draw frame, but the box is visible from t=0 so it
    # does not vanish when the annotator scrubs backward
    assert a.keyframes[0].t == 5
    assert a.t_start == 0
    assert a.visible_at(0) and a.visible_at(5)
    # single keyframe => constant bbox at every T
    assert st.boxes_at(0) and st.boxes_at(99)


def test_add_box_explicit_t_start():
    st = DashboardState(_af(), n_t=100)
    i = st.add_box(1, 1, 2, 2, "cell", t=10, t_start=10)
    assert st.ann(i).t_start == 10


def test_boxes_at_hides_out_of_range():
    a = Annotation(label="cell", keyframes=[Keyframe(0, [0, 0, 10, 10])],
                   t_start=3, t_end=8)
    st = DashboardState(_af([a]), n_t=20)
    assert st.boxes_at(0) == []      # before birth
    assert len(st.boxes_at(5)) == 1  # inside window
    assert st.boxes_at(9) == []      # after end


def test_update_box_autokeyframes_at_t():
    st = DashboardState(_af(), n_t=100)
    st.set_t(0)
    i = st.add_box(100, 100, 40, 40, "cell")
    # move it at a later frame -> should add a second keyframe
    st.set_t(20)
    st.update_box(i, 150, 150, 40, 40)
    a = st.ann(i)
    assert len(a.keyframes) == 2
    ts = sorted(k.t for k in a.keyframes)
    assert ts == [0, 20]
    # interpolation halfway should be between the two centres
    rows = st.boxes_at(10)
    assert len(rows) == 1
    assert 100 < rows[0]["cx"] < 150


def test_update_box_same_t_replaces_keyframe():
    st = DashboardState(_af(), n_t=100)
    st.set_t(4)
    i = st.add_box(100, 100, 40, 40, "cell")
    st.update_box(i, 120, 120, 40, 40)  # same t=4
    a = st.ann(i)
    assert len(a.keyframes) == 1
    assert a.keyframes[0].bbox == cwh_to_bbox(120, 120, 40, 40)


def test_resize_keeps_center_and_affects_all_keyframes():
    st = DashboardState(_af(), n_t=100)
    st.set_t(0)
    i = st.add_box(100, 100, 40, 40, "cell")   # keyframe at t=0
    st.set_t(20)
    st.update_box(i, 300, 300, 40, 40)          # 2nd keyframe at t=20 (moved)
    st.resize(i, 80, 60)                         # resize whole track
    a = st.ann(i)
    assert len(a.keyframes) == 2                 # resize didn't add a keyframe
    for k in a.keyframes:
        w = k.bbox[3] - k.bbox[1]
        h = k.bbox[2] - k.bbox[0]
        assert (round(w), round(h)) == (80, 60)
    # centres preserved: kf0 ~ (100,100), kf1 ~ (300,300)
    k0 = sorted(a.keyframes, key=lambda k: k.t)[0]
    cy0 = (k0.bbox[0] + k0.bbox[2]) / 2
    cx0 = (k0.bbox[1] + k0.bbox[3]) / 2
    assert (round(cy0), round(cx0)) == (100, 100)


def test_size_of_returns_first_keyframe_wh():
    st = DashboardState(_af(), n_t=10)
    i = st.add_box(50, 50, 120, 90, "cell")
    assert st.size_of(i) == (120.0, 90.0)


def test_scale_grows_and_shrinks_about_center():
    st = DashboardState(_af(), n_t=10)
    i = st.add_box(200, 200, 100, 100, "cell")  # center (200,200)
    st.scale(i, 1.1)
    w, h = st.size_of(i)
    assert (round(w), round(h)) == (110, 110)
    st.scale(i, 0.9)  # back down (~99)
    w2, h2 = st.size_of(i)
    assert round(w2) == round(110 * 0.9)
    # centre unchanged
    b = st.ann(i).keyframes[0].bbox
    assert (round((b[1] + b[3]) / 2), round((b[0] + b[2]) / 2)) == (200, 200)


def test_delete_removes_annotation():
    st = DashboardState(_af(), n_t=10)
    i = st.add_box(1, 1, 2, 2, "cell")
    st.delete(i)
    assert st.boxes_at(0) == []
    assert st.counts() == {}


def test_set_label_and_counts():
    st = DashboardState(_af(), n_t=10)
    a = st.add_box(1, 1, 2, 2, "cell")
    b = st.add_box(5, 5, 2, 2, "cell")
    st.set_label(a, "single")
    st.set_label(b, "doublet")
    assert st.counts() == {"single": 1, "doublet": 1}


def test_set_label_registers_new_class():
    st = DashboardState(_af(), n_t=10)
    i = st.add_box(1, 1, 2, 2, "cell")
    st.set_label(i, "mitotic")
    assert "mitotic" in st.classes


def test_lifecycle_birth_end_deaths():
    st = DashboardState(_af(), n_t=100)
    st.set_t(7)
    i = st.add_box(1, 1, 2, 2, "cell")
    assert st.mark_birth(i) is True  # birth at current t=7
    st.set_t(50)
    assert st.mark_end(i) is True
    st.set_t(30)
    st.add_death(i)
    st.set_t(40)
    st.add_death(i)
    a = st.ann(i)
    assert a.t_start == 7
    assert a.t_end == 50
    assert a.t_deaths == [30, 40]
    st.pop_death(i)
    assert a.t_deaths == [30]
    st.clear_deaths(i)
    assert a.t_deaths == []
    st.clear_birth(i)
    assert a.t_start == 0
    st.clear_end(i)
    assert a.t_end is None


def test_mark_end_refuses_before_birth():
    st = DashboardState(_af(), n_t=100)
    st.set_t(20)
    i = st.add_box(1, 1, 2, 2, "cell")
    st.mark_birth(i, t=20)          # birth at 20
    st.set_t(5)
    assert st.mark_end(i) is False  # end before birth -> refused
    assert st.ann(i).t_end is None  # unchanged
    # the box stays visible/selectable (not trapped invisible)
    assert st.ann(i).visible_at(20)


def test_mark_birth_refuses_after_end():
    st = DashboardState(_af(), n_t=100)
    i = st.add_box(1, 1, 2, 2, "cell")
    st.mark_end(i, t=30)
    assert st.mark_birth(i, t=40) is False  # birth after end -> refused
    assert st.ann(i).t_start == 0


def test_add_death_dedups():
    st = DashboardState(_af(), n_t=100)
    i = st.add_box(1, 1, 2, 2, "cell")
    st.set_t(10)
    st.add_death(i)
    st.add_death(i)  # same frame again
    assert st.ann(i).t_deaths == [10]


def test_drop_keyframe_refuses_last():
    st = DashboardState(_af(), n_t=100)
    st.set_t(0)
    i = st.add_box(1, 1, 2, 2, "cell")
    assert st.drop_keyframe(i, t=0) is False  # only keyframe -> refused
    assert len(st.ann(i).keyframes) == 1


def test_apply_cds_edits_adds_new_box():
    st = DashboardState(_af(), n_t=50)
    st.set_t(3)
    rows = [{"id": None, "cx": 100, "cy": 100, "w": 40, "h": 40}]
    out = st.apply_cds_edits(rows, default_label="cell")
    assert len(out) == 1 and out[0] in st._by_id
    a = st.ann(out[0])
    # new boxes are visible from the start (t_start=0), keyframe at draw frame
    assert a.label == "cell" and a.t_start == 0
    assert a.keyframes[0].t == 3


def test_apply_cds_edits_move_adds_keyframe_only_when_moved():
    st = DashboardState(_af(), n_t=50)
    st.set_t(0)
    i = st.add_box(100, 100, 40, 40, "cell")
    # re-send the SAME geometry at a later frame -> must NOT add a keyframe
    st.set_t(10)
    rows = st.boxes_at(10)
    st.apply_cds_edits(rows, default_label="cell")
    assert len(st.ann(i).keyframes) == 1  # unchanged
    # now actually move it -> keyframe added
    rows = st.boxes_at(10)
    rows[0]["cx"] += 25
    st.apply_cds_edits(rows, default_label="cell")
    assert len(st.ann(i).keyframes) == 2


def test_apply_cds_edits_deletes_missing():
    st = DashboardState(_af(), n_t=50)
    st.set_t(0)
    i = st.add_box(100, 100, 40, 40, "cell")
    st.apply_cds_edits([], default_label="cell")  # user deleted the box
    assert i not in st._by_id


def test_apply_cds_edits_delete_only_affects_visible_frame():
    # a box NOT visible at t must not be deleted just because it's absent
    a = Annotation(label="cell", keyframes=[Keyframe(0, [0, 0, 10, 10])],
                   t_start=0, t_end=5)
    st = DashboardState(_af([a]), n_t=50)
    st.set_t(20)  # 'a' is not visible here
    assert st.boxes_at(20) == []
    st.apply_cds_edits([], default_label="cell", t=20)
    # 'a' still exists — it was simply not on screen at t=20
    assert len(st._by_id) == 1


def test_boxes_at_assigns_stable_track_numbers():
    st = DashboardState(_af(), n_t=10)
    a = st.add_box(10, 10, 4, 4, "cell")
    b = st.add_box(50, 50, 4, 4, "cell")
    rows = {r["id"]: r["num"] for r in st.boxes_at(0)}
    assert rows[a] == 1 and rows[b] == 2
    # numbers are stable across frames (same track keeps its number)
    rows2 = {r["id"]: r["num"] for r in st.boxes_at(5)}
    assert rows2[a] == 1 and rows2[b] == 2


def test_set_detections_keeps_curated_replaces_provisional():
    st = DashboardState(_af(), n_t=20)
    prov = st.add_box(10, 10, 4, 4, "cell")     # untriaged
    curated = st.add_box(50, 50, 4, 4, "cell")
    st.set_label(curated, "single")             # human curated this one
    # a fresh detection run produces two new provisional tracks
    new = [
        Annotation(label="cell", keyframes=[Keyframe(0, [0, 0, 5, 5])]),
        Annotation(label="cell", keyframes=[Keyframe(0, [9, 9, 14, 14])]),
    ]
    added = st.set_detections(new, provisional_label="cell")
    assert added == 2
    labels = sorted(a.label for a in st.sync_to_file().annotations)
    # the curated 'single' survived; the old provisional 'cell' was dropped
    assert labels == ["cell", "cell", "single"]
    assert not st.has(prov)      # old provisional removed
    assert st.has(curated)       # curated kept


def test_sync_to_file_preserves_order():
    st = DashboardState(_af(), n_t=10)
    a = st.add_box(1, 1, 2, 2, "cell")
    b = st.add_box(5, 5, 2, 2, "cell")
    af = st.sync_to_file()
    assert len(af.annotations) == 2
    assert af.annotations[0] is st.ann(a)
    assert af.annotations[1] is st.ann(b)
