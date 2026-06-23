import json

from nikon_control.annotate import (
    Annotation,
    AnnotationFile,
    DEFAULT_CLASSES,
    SCHEMA_VERSION,
    _bbox_to_shape,
    _compute_label,
    _shape_to_bbox,
    load,
    save,
)


def test_save_load_roundtrip(tmp_path):
    af = AnnotationFile(
        source="/path/to/file.nd2",
        image_shape=[60, 3, 1024, 1024],
        axes=["T", "C", "Y", "X"],
        channels=["BF", "GFP", "DAPI"],
        annotator="ch",
        annotations=[
            Annotation(bbox=[10.0, 20.0, 100.0, 200.0], label="single"),
            Annotation(
                bbox=[300.0, 400.0, 380.0, 480.0],
                label="doublet",
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
    assert loaded.annotations[0].bbox == [10.0, 20.0, 100.0, 200.0]
    assert loaded.annotations[0].t_start == 0
    assert loaded.annotations[0].t_end is None
    assert loaded.annotations[0].t_deaths == []
    assert loaded.annotations[1].label == "doublet"
    assert loaded.annotations[1].t_start == 5
    assert loaded.annotations[1].t_end == 50
    assert loaded.annotations[1].t_deaths == [42, 47]


def test_default_classes_includes_fission_fusion():
    assert "fission_fusion" in DEFAULT_CLASSES


def test_compute_label_is_time_aware():
    # No markers when nothing is set
    assert _compute_label(0, [], 0) == ""
    # Birth marker is always shown when t_start > 0 (independent of current_t)
    assert _compute_label(5, [], 0) == "↑T=5"
    assert _compute_label(5, [], 100) == "↑T=5"
    # Single death: hidden before t_death, shown at and after
    assert _compute_label(0, [42], 0) == ""
    assert _compute_label(0, [42], 41) == ""
    assert _compute_label(0, [42], 42) == "†T=42"
    assert _compute_label(0, [42], 100) == "†T=42"
    # Both birth and death
    assert _compute_label(5, [42], 0) == "↑T=5"
    assert _compute_label(5, [42], 50) == "↑T=5 †T=42"


def test_compute_label_with_multiple_deaths():
    # Doublet: two cells dying at different times
    deaths = [42, 50]
    assert _compute_label(0, deaths, 41) == ""        # nothing yet
    assert _compute_label(0, deaths, 42) == "†T=42"   # first death only
    assert _compute_label(0, deaths, 49) == "†T=42"   # second still pending
    assert _compute_label(0, deaths, 50) == "†T=42,50"  # both now in the past
    assert _compute_label(0, deaths, 99) == "†T=42,50"
    # Birth + two deaths
    assert _compute_label(5, deaths, 50) == "↑T=5 †T=42,50"


def test_bbox_shape_roundtrip():
    shape = _bbox_to_shape([10.0, 20.0, 100.0, 200.0])
    assert len(shape) == 4
    assert all(len(v) == 2 for v in shape)
    assert _shape_to_bbox(shape) == [10.0, 20.0, 100.0, 200.0]


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
    assert len(loaded.annotations) == 1
    a = loaded.annotations[0]
    assert a.bbox == [10.0, 20.0, 100.0, 200.0]
    assert a.t_start == 0
    assert a.t_end is None
    assert a.t_deaths == []
    assert not hasattr(a, "t")
    assert not hasattr(a, "t_death")


def test_load_migrates_v0_2(tmp_path):
    # v0.2 used t_end as the death frame. v0.3+ splits these; t_end becomes
    # "last visible" and the old value moves into t_deaths.
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

    assert loaded.schema_version == SCHEMA_VERSION
    a = loaded.annotations[0]
    assert a.t_start == 7
    assert a.t_end is None
    assert a.t_deaths == [42]


def test_load_migrates_v0_3(tmp_path):
    # v0.3 had a scalar `t_death`; v0.4 generalises to a list `t_deaths`.
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

    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.annotations[0].t_deaths == [42]
    assert loaded.annotations[1].t_deaths == []
    assert not hasattr(loaded.annotations[0], "t_death")
