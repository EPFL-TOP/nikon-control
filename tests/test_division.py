"""Tests for time-dependent class changes (division, fission/fusion)."""
from nikon_control.schema import (
    POST_DIVISION_LABEL,
    Annotation,
    AnnotationFile,
    ClassChange,
    Keyframe,
    _compute_label,
    load,
    save,
)
from nikon_control.dashboard.state import DashboardState


def _af(anns=None):
    return AnnotationFile(source="x.nd2",
                          classes=["single", "doublet", "fission_fusion"],
                          annotations=anns or [])


def _cell(**kw):
    return Annotation(label="single", keyframes=[Keyframe(0, [0, 0, 10, 10])],
                      **kw)


def test_label_at_flips_at_change():
    a = _cell(class_changes=[ClassChange(30, "doublet")])
    assert a.label_at(29) == "single"
    assert a.label_at(30) == "doublet"
    assert a.label_at(100) == "doublet"


def test_label_at_multiple_changes_latest_wins():
    a = _cell(class_changes=[ClassChange(30, "doublet"),
                             ClassChange(60, "fission_fusion")])
    assert a.label_at(10) == "single"
    assert a.label_at(30) == "doublet"
    assert a.label_at(59) == "doublet"
    assert a.label_at(60) == "fission_fusion"


def test_label_at_no_change_is_base():
    assert _cell().label_at(999) == "single"


def test_compute_label_shows_changes_and_deaths():
    ccs = [ClassChange(30, "doublet")]
    assert _compute_label(0, [], 10, ccs) == ""
    assert _compute_label(0, [], 30, ccs) == "⑂→doubletT=30"
    assert _compute_label(5, [60], 60,
                          [ClassChange(30, "fission_fusion")]) == \
        "↑T=5 ⑂→fission_fusionT=30 †T=60"


def test_roundtrip_preserves_class_changes(tmp_path):
    af = _af([_cell(class_changes=[ClassChange(42, "doublet"),
                                   ClassChange(70, "fission_fusion")])])
    p = tmp_path / "d.annotations.json"
    save(af, p)
    loaded = load(p)
    cc = loaded.annotations[0].class_changes
    assert [(c.t, c.label) for c in cc] == [(42, "doublet"), (70, "fission_fusion")]
    assert loaded.schema_version == "0.7"


def test_migrate_v06_t_divide_to_class_change(tmp_path):
    import json
    payload = {
        "schema_version": "0.6",
        "source": "x.nd2", "image_shape": [], "axes": [], "channels": [],
        "classes": ["single"], "annotator": "",
        "annotations": [{
            "label": "single",
            "keyframes": [{"t": 0, "bbox": [1, 2, 3, 4]}],
            "t_start": 0, "t_end": None, "t_deaths": [], "t_divide": 40,
            "z": 0, "notes": "", "created": "2026-07-01T00:00:00",
        }],
    }
    p = tmp_path / "v06.annotations.json"
    p.write_text(json.dumps(payload))
    loaded = load(p)
    a = loaded.annotations[0]
    assert [(c.t, c.label) for c in a.class_changes] == [(40, POST_DIVISION_LABEL)]
    assert a.label_at(50) == "doublet"
    assert not hasattr(a, "t_divide")


def test_dashboard_division_and_fission_fusion():
    st = DashboardState(_af(), n_t=100)
    st.set_t(0)
    i = st.add_box(100, 100, 40, 40, "single")
    st.set_t(40)
    st.mark_division(i)
    assert st.boxes_at(20)[0]["label"] == "single"
    assert st.boxes_at(40)[0]["label"] == "doublet"
    st.set_t(70)
    st.mark_fission_fusion(i)
    assert st.boxes_at(70)[0]["label"] == "fission_fusion"
    st.clear_class_changes(i)
    assert st.boxes_at(70)[0]["label"] == "single"
