import json

from nikon_control.annotate import (
    Annotation,
    AnnotationFile,
    DEFAULT_CLASSES,
    Keyframe,
    SCHEMA_VERSION,
    _bbox_to_shape,
    _compute_label,
    _interpolate_bbox,
    _shape_to_bbox,
    load,
    save,
)


def _kf(t, bbox):
    return Keyframe(t=t, bbox=bbox)


def test_save_load_roundtrip(tmp_path):
    af = AnnotationFile(
        source="/path/to/file.nd2",
        image_shape=[60, 3, 1024, 1024],
        axes=["T", "C", "Y", "X"],
        channels=["BF", "GFP", "DAPI"],
        annotator="ch",
        annotations=[
            Annotation(
                label="single",
                keyframes=[_kf(0, [10.0, 20.0, 100.0, 200.0])],
            ),
            Annotation(
                label="doublet",
                keyframes=[
                    _kf(5, [300.0, 400.0, 380.0, 480.0]),
                    _kf(25, [310.0, 420.0, 390.0, 500.0]),
                ],
                t_start=5,
                t_end=50,
                t_deaths=[42, 47],
            ),
        ],
    )
    p = tmp_path / "x.annotations.json"
    save(af, p)
    loaded = load(p)

    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.source == af.source
    assert loaded.classes == list(DEFAULT_CLASSES)
    assert len(loaded.annotations) == 2

    a0 = loaded.annotations[0]
    assert a0.label == "single"
    assert len(a0.keyframes) == 1
    assert a0.keyframes[0].bbox == [10.0, 20.0, 100.0, 200.0]
    assert a0.bbox == [10.0, 20.0, 100.0, 200.0]  # property = first keyframe
    assert a0.t_deaths == []

    a1 = loaded.annotations[1]
    assert a1.label == "doublet"
    assert len(a1.keyframes) == 2
    assert a1.keyframes[0].t == 5
    assert a1.keyframes[1].t == 25
    assert a1.t_start == 5
    assert a1.t_end == 50
    assert a1.t_deaths == [42, 47]


def test_default_classes_includes_fission_fusion():
    assert "fission_fusion" in DEFAULT_CLASSES


def test_compute_label_is_time_aware():
    assert _compute_label(0, [], 0) == ""
    assert _compute_label(5, [], 0) == "↑T=5"
    assert _compute_label(5, [], 100) == "↑T=5"
    assert _compute_label(0, [42], 0) == ""
    assert _compute_label(0, [42], 41) == ""
    assert _compute_label(0, [42], 42) == "†T=42"
    assert _compute_label(0, [42], 100) == "†T=42"
    assert _compute_label(5, [42], 0) == "↑T=5"
    assert _compute_label(5, [42], 50) == "↑T=5 †T=42"


def test_compute_label_with_multiple_deaths():
    deaths = [42, 50]
    assert _compute_label(0, deaths, 41) == ""
    assert _compute_label(0, deaths, 42) == "†T=42"
    assert _compute_label(0, deaths, 49) == "†T=42"
    assert _compute_label(0, deaths, 50) == "†T=42,50"
    assert _compute_label(0, deaths, 99) == "†T=42,50"
    assert _compute_label(5, deaths, 50) == "↑T=5 †T=42,50"


def test_bbox_shape_roundtrip():
    shape = _bbox_to_shape([10.0, 20.0, 100.0, 200.0])
    assert len(shape) == 4
    assert all(len(v) == 2 for v in shape)
    assert _shape_to_bbox(shape) == [10.0, 20.0, 100.0, 200.0]


def test_interpolate_single_keyframe_is_constant():
    kfs = [_kf(10, [0.0, 0.0, 10.0, 20.0])]
    # Before, at, and after the keyframe — all snap to the single value.
    assert _interpolate_bbox(kfs, 0) == [0.0, 0.0, 10.0, 20.0]
    assert _interpolate_bbox(kfs, 10) == [0.0, 0.0, 10.0, 20.0]
    assert _interpolate_bbox(kfs, 500) == [0.0, 0.0, 10.0, 20.0]


def test_interpolate_between_two_keyframes():
    kfs = [_kf(0, [0.0, 0.0, 10.0, 10.0]), _kf(10, [20.0, 30.0, 40.0, 50.0])]
    # At the keyframes themselves
    assert _interpolate_bbox(kfs, 0) == [0.0, 0.0, 10.0, 10.0]
    assert _interpolate_bbox(kfs, 10) == [20.0, 30.0, 40.0, 50.0]
    # Midpoint should be midpoint
    assert _interpolate_bbox(kfs, 5) == [10.0, 15.0, 25.0, 30.0]
    # Quarter-way
    out = _interpolate_bbox(kfs, 2)
    assert out[0] == 4.0  # 0 + 0.2 * (20 - 0)


def test_interpolate_snaps_outside_range():
    kfs = [_kf(10, [0.0, 0.0, 10.0, 10.0]), _kf(20, [20.0, 20.0, 30.0, 30.0])]
    # Before the first keyframe → snap to first
    assert _interpolate_bbox(kfs, 0) == [0.0, 0.0, 10.0, 10.0]
    # After the last keyframe → snap to last
    assert _interpolate_bbox(kfs, 99) == [20.0, 20.0, 30.0, 30.0]


def test_interpolate_with_unsorted_keyframes():
    kfs = [_kf(20, [20.0, 20.0, 30.0, 30.0]), _kf(0, [0.0, 0.0, 10.0, 10.0])]
    # Interpolation should still work — keyframes get sorted internally.
    assert _interpolate_bbox(kfs, 10) == [10.0, 10.0, 20.0, 20.0]


def test_load_migrates_v0_1(tmp_path):
    payload = {
        "schema_version": "0.1",
        "source": "/some/path.nd2",
        "image_shape": [60, 3, 1024, 1024],
        "axes": ["T", "C", "Y", "X"],
        "channels": ["BF"],
        "classes": ["single", "doublet"],
        "annotator": "",
        "annotations": [
            {
                "t": 5,
                "z": 0,
                "bbox": [10.0, 20.0, 100.0, 200.0],
                "label": "single",
                "notes": "",
                "created": "2026-05-18T14:00:00",
            }
        ],
    }
    p = tmp_path / "old.annotations.json"
    p.write_text(json.dumps(payload))

    loaded = load(p)

    assert loaded.schema_version == SCHEMA_VERSION
    a = loaded.annotations[0]
    assert len(a.keyframes) == 1
    assert a.keyframes[0].bbox == [10.0, 20.0, 100.0, 200.0]
    assert a.bbox == [10.0, 20.0, 100.0, 200.0]
    assert a.t_deaths == []


def test_load_migrates_v0_2(tmp_path):
    payload = {
        "schema_version": "0.2",
        "source": "/some/path.nd2",
        "image_shape": [60, 3, 1024, 1024],
        "axes": ["T", "C", "Y", "X"],
        "channels": ["BF"],
        "classes": ["single"],
        "annotator": "",
        "annotations": [
            {
                "bbox": [1.0, 2.0, 3.0, 4.0],
                "label": "single",
                "t_start": 7,
                "t_end": 42,
                "z": 0,
                "notes": "",
                "created": "2026-05-18T14:00:00",
            }
        ],
    }
    p = tmp_path / "v0_2.annotations.json"
    p.write_text(json.dumps(payload))

    loaded = load(p)

    a = loaded.annotations[0]
    assert a.t_start == 7
    assert a.t_end is None
    assert a.t_deaths == [42]
    # v0.2 bbox migrates into a single keyframe at t_start
    assert len(a.keyframes) == 1
    assert a.keyframes[0].t == 7
    assert a.keyframes[0].bbox == [1.0, 2.0, 3.0, 4.0]


def test_load_migrates_v0_3(tmp_path):
    payload = {
        "schema_version": "0.3",
        "source": "/some/path.nd2",
        "image_shape": [60, 3, 1024, 1024],
        "axes": ["T", "C", "Y", "X"],
        "channels": ["BF"],
        "classes": ["single"],
        "annotator": "",
        "annotations": [
            {
                "bbox": [1.0, 2.0, 3.0, 4.0],
                "label": "single",
                "t_start": 0,
                "t_end": None,
                "t_death": 42,
                "z": 0,
                "notes": "",
                "created": "2026-05-22T14:00:00",
            },
            {
                "bbox": [5.0, 6.0, 7.0, 8.0],
                "label": "single",
                "t_start": 0,
                "t_end": None,
                "t_death": None,
                "z": 0,
                "notes": "",
                "created": "2026-05-22T14:00:00",
            },
        ],
    }
    p = tmp_path / "v0_3.annotations.json"
    p.write_text(json.dumps(payload))

    loaded = load(p)

    assert loaded.annotations[0].t_deaths == [42]
    assert loaded.annotations[1].t_deaths == []
    # Migration also folds bbox into keyframes
    assert len(loaded.annotations[0].keyframes) == 1
    assert loaded.annotations[0].keyframes[0].bbox == [1.0, 2.0, 3.0, 4.0]


def test_load_migrates_v0_4(tmp_path):
    payload = {
        "schema_version": "0.4",
        "source": "/some/path.nd2",
        "image_shape": [60, 3, 1024, 1024],
        "axes": ["T", "C", "Y", "X"],
        "channels": ["BF"],
        "classes": ["single", "doublet"],
        "annotator": "",
        "annotations": [
            {
                "bbox": [10.0, 20.0, 100.0, 200.0],
                "label": "single",
                "t_start": 3,
                "t_end": None,
                "t_deaths": [42],
                "z": 0,
                "notes": "",
                "created": "2026-06-11T14:00:00",
            }
        ],
    }
    p = tmp_path / "v0_4.annotations.json"
    p.write_text(json.dumps(payload))

    loaded = load(p)

    assert loaded.schema_version == SCHEMA_VERSION
    a = loaded.annotations[0]
    assert len(a.keyframes) == 1
    assert a.keyframes[0].t == 3  # placed at t_start
    assert a.keyframes[0].bbox == [10.0, 20.0, 100.0, 200.0]
    assert a.t_start == 3
    assert a.t_deaths == [42]
    assert not hasattr(a, "bbox") or a.bbox == [10.0, 20.0, 100.0, 200.0]
