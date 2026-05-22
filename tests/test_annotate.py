import json

from nikon_control.annotate import (
    Annotation,
    AnnotationFile,
    DEFAULT_CLASSES,
    SCHEMA_VERSION,
    _bbox_to_shape,
    _life_label,
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
                bbox=[300.0, 400.0, 380.0, 480.0], label="doublet", t_end=42
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
    assert loaded.annotations[1].label == "doublet"
    assert loaded.annotations[1].t_end == 42


def test_life_label_combinations():
    assert _life_label(0, -1) == ""
    assert _life_label(5, -1) == "↑T=5"
    assert _life_label(0, 42) == "†T=42"
    assert _life_label(5, 42) == "↑T=5 †T=42"


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
    assert not hasattr(a, "t")
