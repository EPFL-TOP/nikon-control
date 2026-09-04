"""Tests for the review dashboard controller (operates on COCO output)."""
import json

import pytest

from nikon_control.dashboard.review_state import (
    FILTER_ALL,
    FILTER_EMPTY,
    FILTER_REVIEWED,
    FILTER_UNREVIEWED,
    FILTER_UNVERIFIED,
    ReviewState,
    bbox_to_cwh,
    cwh_to_bbox,
    load_splits,
)

CATS = [{"id": 1, "name": "single"}, {"id": 2, "name": "doublet"},
        {"id": 3, "name": "debris"}]


def _payload(images, anns):
    return {"images": images, "annotations": anns, "categories": CATS}


def _splits():
    train = _payload(
        [{"id": 1, "file_name": "a_t0000.tif", "height": 64, "width": 64,
          "source": "a.nd2", "frame": 0},
         {"id": 2, "file_name": "a_t0001.tif", "height": 64, "width": 64,
          "source": "a.nd2", "frame": 1}],
        [{"id": 1, "image_id": 1, "category_id": 1,
          "bbox": [10, 20, 30, 40], "area": 1200, "iscrowd": 0,
          "auto": True, "score": 0.8, "cell": "c1"},
         {"id": 2, "image_id": 1, "category_id": 3,
          "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0,
          "auto": False, "score": None, "cell": None}],
    )
    val = _payload(
        [{"id": 1, "file_name": "b_t0000.tif", "height": 64, "width": 64,
          "source": "b.nd2", "frame": 0}],
        [{"id": 1, "image_id": 1, "category_id": 2,
          "bbox": [5, 5, 20, 20], "area": 400, "iscrowd": 0,
          "auto": False, "score": None, "cell": "c9"}],
    )
    return {"train": train, "val": val}


def test_bbox_conversion_roundtrip():
    assert bbox_to_cwh([10, 20, 30, 40]) == (25.0, 40.0, 30.0, 40.0)
    assert cwh_to_bbox(25, 40, 30, 40) == [10.0, 20.0, 30.0, 40.0]


def test_ids_are_namespaced_by_split():
    """Both splits have an annotation id 1 — they must not collide."""
    st = ReviewState(_splits())
    assert st.has("train:1") and st.has("val:1")
    assert st.boxes(0)[0]["id"] == "train:1"
    st.goto(2)
    assert st.split_of() == "val"
    assert st.boxes()[0]["id"] == "val:1"


def test_images_flattened_across_splits():
    st = ReviewState(_splits())
    assert len(st.images) == 3
    assert st.split_counts() == {"train": 2, "val": 1}
    assert st.classes == ["single", "doublet", "debris"]


def test_boxes_expose_class_and_unverified_marker():
    st = ReviewState(_splits())
    rows = st.boxes(0)
    assert [r["label"] for r in rows] == ["single", "debris"]
    assert rows[0]["marker"] == "0.80 ?"    # score + unverified
    assert rows[1]["marker"] == ""          # human-drawn, no score
    assert (rows[0]["cx"], rows[0]["cy"]) == (25.0, 40.0)


def test_set_class_also_verifies():
    st = ReviewState(_splits())
    st.set_class("train:1", "doublet")
    assert st.boxes(0)[0]["label"] == "doublet"
    assert st._anns["train:1"]["auto"] is False   # reviewing verifies
    assert st.unverified_count() == 0
    assert st.dirty


def test_set_class_rejects_unknown_class():
    st = ReviewState(_splits())
    with pytest.raises(ValueError, match="unknown class"):
        st.set_class("train:1", "banana")


def test_delete_removes_from_payload_too():
    """A deleted box must be gone from what gets written, not just the index."""
    st = ReviewState(_splits())
    st.delete("train:1")
    assert not st.has("train:1")
    assert len(st.boxes(0)) == 1
    ids = [a["id"] for a in st.splits["train"]["annotations"]]
    assert ids == [2]


def test_add_box_gets_a_fresh_id_and_lands_in_the_payload():
    st = ReviewState(_splits())
    key = st.add_box(30, 30, 10, 10, "debris", idx=1)
    assert key not in ("train:1", "train:2")
    st.goto(1)
    rows = st.boxes()
    assert len(rows) == 1 and rows[0]["label"] == "debris"
    added = [a for a in st.splits["train"]["annotations"]
             if a["id"] == int(key.split(":")[1])][0]
    assert added["image_id"] == 2          # attached to the right image
    assert added["auto"] is False          # human-drawn
    assert added["bbox"] == [25.0, 25.0, 10.0, 10.0]


def test_geometry_edits_mark_verified_and_keep_area_consistent():
    st = ReviewState(_splits())
    st.move_to("train:1", 100, 100)
    assert st.center_of("train:1") == (100, 100)
    assert st.size_of("train:1") == (30, 40)     # size kept
    st.nudge("train:1", 10, -10)
    assert st.center_of("train:1") == (110, 90)
    st.resize("train:1", 20, 20)
    a = st._anns["train:1"]
    assert a["area"] == 400 and a["auto"] is False
    st.scale("train:1", 2.0)
    assert st.size_of("train:1") == (40, 40)


def test_filters():
    st = ReviewState(_splits())
    assert st.set_filter(FILTER_ALL) == 3
    assert st.set_filter(FILTER_UNVERIFIED) == 1      # only train image 1
    assert st.visible() == [0]
    assert st.set_filter(FILTER_EMPTY) == 1           # train image 2
    assert st.visible() == [1]
    assert st.set_filter("doublet") == 1              # by class -> val image
    assert st.visible() == [2]


def test_filter_jumps_to_first_match_and_step_stays_within_filter():
    st = ReviewState(_splits())
    st.set_filter("doublet")
    assert st.current == 2          # jumped to the matching image
    st.step(-1)
    assert st.current == 2          # no earlier match: stays put
    st.set_filter(FILTER_ALL)
    st.goto(0)
    st.step(1)
    assert st.current == 1


def test_review_progress_persists_on_the_image_entry():
    st = ReviewState(_splits())
    assert st.review_progress() == (0, 3)
    st.mark_reviewed(True, idx=0)
    assert st.review_progress() == (1, 3)
    assert st.splits["train"]["images"][0]["reviewed"] is True
    assert st.set_filter(FILTER_UNREVIEWED) == 2
    assert st.set_filter(FILTER_REVIEWED) == 1


def test_counts():
    st = ReviewState(_splits())
    assert st.counts() == {"single": 1, "doublet": 1, "debris": 1}
    assert st.unverified_count() == 1


def test_save_writes_both_splits_and_backs_up_once(tmp_path):
    ann = tmp_path / "annotations"
    ann.mkdir()
    sp = _splits()
    for name, payload in sp.items():
        (ann / f"{name}.json").write_text(json.dumps(payload))

    st = ReviewState(load_splits(tmp_path), tmp_path)
    st.set_class("train:1", "debris")
    written = st.save()
    assert {p.name for p in written} == {"train.json", "val.json"}
    assert not st.dirty
    # the edit landed on disk
    reread = ReviewState(load_splits(tmp_path))
    assert reread.boxes(0)[0]["label"] == "debris"
    # the ORIGINAL is preserved, and a second save doesn't clobber the backup
    bak = json.loads((ann / "train.json.bak").read_text())
    assert bak["annotations"][0]["category_id"] == 1   # pre-edit
    st.set_class("train:2", "single")
    st.save()
    bak2 = json.loads((ann / "train.json.bak").read_text())
    assert bak2["annotations"][0]["category_id"] == 1  # still the original


def test_load_splits_rejects_a_non_dataset(tmp_path):
    with pytest.raises(FileNotFoundError, match="nikon-control-export"):
        load_splits(tmp_path)


# ---- overlap suppression -------------------------------------------------

def _overlap_splits():
    """One image with a small 'single' fully inside a big 'doublet'."""
    train = _payload(
        [{"id": 1, "file_name": "a_t0000.tif", "height": 200, "width": 200,
          "source": "a.nd2", "frame": 0}],
        [{"id": 1, "image_id": 1, "category_id": 2,          # doublet, big
          "bbox": [0, 0, 100, 100], "area": 10000, "iscrowd": 0,
          "auto": True, "score": 0.6, "cell": "c1"},
         {"id": 2, "image_id": 1, "category_id": 1,          # single, inside
          "bbox": [10, 10, 30, 30], "area": 900, "iscrowd": 0,
          "auto": True, "score": 0.95, "cell": "c2"}],
    )
    return {"train": train}


def test_review_detects_and_suppresses_overlap():
    st = ReviewState(_overlap_splits())
    assert st.overlap_count(0.7) == 2
    assert st.has_overlaps()
    assert st.suppress_overlaps(0.7) == 1
    assert [r["label"] for r in st.boxes()] == ["doublet"]   # bigger kept
    assert st.overlap_count(0.7) == 0
    # and it really left the payload
    assert [a["id"] for a in st.splits["train"]["annotations"]] == [1]


def test_review_overlap_filter():
    from nikon_control.dashboard.review_state import FILTER_OVERLAP
    st = ReviewState(_overlap_splits())
    assert st.set_filter(FILTER_OVERLAP) == 1
    st.suppress_overlaps(0.7)
    assert st.set_filter(FILTER_OVERLAP) == 0


def test_review_suppression_protects_human_boxes():
    sp = _overlap_splits()
    sp["train"]["annotations"][1]["auto"] = False     # a human drew the single
    st = ReviewState(sp)
    assert st.suppress_overlaps(0.7) == 1
    assert [r["label"] for r in st.boxes()] == ["single"]   # human box kept
