from pathlib import Path

from nikon_control.annotate import (
    Annotation,
    AnnotationFile,
    DEFAULT_CLASSES,
    SCHEMA_VERSION,
    _bbox_to_shape,
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
            Annotation(t=5, z=0, bbox=[10.0, 20.0, 100.0, 200.0], label="single"),
            Annotation(t=5, z=0, bbox=[300.0, 400.0, 380.0, 480.0], label="doublet"),
        ],
    )
    p = tmp_path / "x.annotations.json"
    save(af, p)
    loaded = load(p)

    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.source == af.source
    assert loaded.image_shape == af.image_shape
    assert loaded.axes == af.axes
    assert loaded.channels == af.channels
    assert loaded.classes == list(DEFAULT_CLASSES)
    assert len(loaded.annotations) == 2
    assert loaded.annotations[0].bbox == [10.0, 20.0, 100.0, 200.0]
    assert loaded.annotations[0].label == "single"
    assert loaded.annotations[1].label == "doublet"


def test_bbox_shape_roundtrip_with_time():
    axes = ["T", "Y", "X"]
    shape = _bbox_to_shape(t=7, z=0, bbox=[10.0, 20.0, 100.0, 200.0], axes=axes)
    assert len(shape) == 4  # 4 vertices
    assert all(len(v) == 3 for v in shape)  # 3 dims each

    t, z, bbox = _shape_to_bbox(shape, axes)
    assert t == 7
    assert z == 0
    assert bbox == [10.0, 20.0, 100.0, 200.0]


def test_bbox_shape_roundtrip_with_time_and_z():
    axes = ["T", "Z", "Y", "X"]
    shape = _bbox_to_shape(t=3, z=2, bbox=[5.0, 15.0, 50.0, 150.0], axes=axes)
    t, z, bbox = _shape_to_bbox(shape, axes)
    assert t == 3
    assert z == 2
    assert bbox == [5.0, 15.0, 50.0, 150.0]


def test_bbox_shape_roundtrip_2d_only():
    axes = ["Y", "X"]
    shape = _bbox_to_shape(t=0, z=0, bbox=[1.0, 2.0, 3.0, 4.0], axes=axes)
    t, z, bbox = _shape_to_bbox(shape, axes)
    assert t == 0
    assert z == 0
    assert bbox == [1.0, 2.0, 3.0, 4.0]
