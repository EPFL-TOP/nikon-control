"""Tests for the simplified per-frame annotation schema."""
import json

import pytest

from nikon_control.schema import Annotation, Keyframe
from nikon_control.schema_simple import (
    PROVISIONAL_LABEL,
    SIMPLE_KIND,
    SIMPLE_SUFFIX,
    TRAINING_CLASSES,
    SimpleAnnotationFile,
    SimpleBox,
    boxes_by_frame,
    boxes_from_annotations,
    class_counts,
    load_simple,
    save_simple,
    simple_path_for,
)


def _af(boxes=None):
    return SimpleAnnotationFile(source="x.nd2", n_frames=20,
                                boxes=boxes or [])


def test_sidecar_path_is_distinct_from_tracked():
    p = simple_path_for("/data/pos1.nd2")
    assert p.name == "pos1" + SIMPLE_SUFFIX
    assert p.name != "pos1.annotations.json"


def test_roundtrip(tmp_path):
    af = _af([SimpleBox(t=0, bbox=[1, 2, 11, 12], label="single"),
              SimpleBox(t=3, bbox=[5, 6, 25, 26], label="doublet", score=0.9)])
    p = tmp_path / ("a" + SIMPLE_SUFFIX)
    save_simple(af, p)
    loaded = load_simple(p)
    assert loaded.kind == SIMPLE_KIND
    assert [(b.t, b.label) for b in loaded.boxes] == [(0, "single"), (3, "doublet")]
    assert loaded.boxes[1].score == 0.9
    assert loaded.n_frames == 20


def test_load_simple_refuses_a_tracked_file(tmp_path):
    """A tracked .annotations.json must not load as a simplified one."""
    p = tmp_path / "t.annotations.json"
    p.write_text(json.dumps({
        "schema_version": "0.7", "source": "x.nd2", "annotations": [],
    }))
    with pytest.raises(ValueError, match="not a simplified annotation file"):
        load_simple(p)


def test_training_boxes_excludes_unlabeled():
    af = _af([SimpleBox(t=0, bbox=[0, 0, 1, 1], label="single"),
              SimpleBox(t=0, bbox=[0, 0, 1, 1], label=PROVISIONAL_LABEL),
              SimpleBox(t=1, bbox=[0, 0, 1, 1], label="debris")])
    assert len(af.boxes) == 3
    assert [b.label for b in af.training_boxes()] == ["single", "debris"]
    assert PROVISIONAL_LABEL not in TRAINING_CLASSES


def test_boxes_by_frame_groups():
    af = _af([SimpleBox(t=0, bbox=[0, 0, 1, 1], label="single"),
              SimpleBox(t=0, bbox=[2, 2, 3, 3], label="doublet"),
              SimpleBox(t=5, bbox=[0, 0, 1, 1], label="debris")])
    g = boxes_by_frame(af)
    assert sorted(g) == [0, 5]
    assert len(g[0]) == 2 and len(g[5]) == 1


def test_class_counts():
    af = _af([SimpleBox(t=0, bbox=[0, 0, 1, 1], label="single"),
              SimpleBox(t=1, bbox=[0, 0, 1, 1], label="single"),
              SimpleBox(t=1, bbox=[0, 0, 1, 1], label="debris")])
    c = class_counts(af)
    assert c["single"] == 2 and c["debris"] == 1 and c["doublet"] == 0


def test_boxes_from_annotations_flattens_a_track():
    """A tracked annotation becomes one independent box per visible frame."""
    a = Annotation(label="debris", t_start=2, t_end=4,
                   keyframes=[Keyframe(2, [0, 0, 10, 10]),
                              Keyframe(4, [20, 20, 30, 30])])
    boxes = boxes_from_annotations([a], n_frames=20, label="debris")
    assert [b.t for b in boxes] == [2, 3, 4]
    assert all(b.label == "debris" for b in boxes)
    # frame 3 is interpolated between the two keyframes
    assert boxes[1].bbox == [10.0, 10.0, 20.0, 20.0]


def test_boxes_from_annotations_clips_to_n_frames():
    a = Annotation(label="debris", t_start=0, t_end=None,
                   keyframes=[Keyframe(0, [0, 0, 5, 5])])
    boxes = boxes_from_annotations([a], n_frames=4)
    assert [b.t for b in boxes] == [0, 1, 2, 3]
