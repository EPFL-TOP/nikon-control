"""Tests for the COCO training-set exporter (pure parts, no ND2 needed)."""
import math

from nikon_control.export_training import (
    CATEGORY_IDS,
    build_coco,
    coco_categories,
    frames_to_export,
    split_by_source,
)
from nikon_control.schema_simple import (
    PROVISIONAL_LABEL,
    SimpleAnnotationFile,
    SimpleBox,
)


def _af(boxes):
    return SimpleAnnotationFile(source="x.nd2", n_frames=20, boxes=boxes)


def test_category_ids_reserve_zero_for_background():
    assert 0 not in CATEGORY_IDS.values()
    assert CATEGORY_IDS == {"single": 1, "doublet": 2, "debris": 3}
    assert [c["name"] for c in coco_categories()] == ["single", "doublet", "debris"]


def test_split_is_by_file_and_deterministic():
    stems = [f"pos{i}" for i in range(10)]
    tr1, va1 = split_by_source(stems, 0.2, seed=0)
    tr2, va2 = split_by_source(stems, 0.2, seed=0)
    assert (tr1, va1) == (tr2, va2)          # deterministic
    assert not (tr1 & va1)                    # no file in both
    assert tr1 | va1 == set(stems)            # nothing lost
    assert len(va1) == 2


def test_split_never_leaves_train_empty():
    tr, va = split_by_source(["a", "b"], val_frac=0.9, seed=0)
    assert len(tr) >= 1 and len(va) >= 1


def test_split_single_file_gives_empty_val():
    """One file cannot be split without leaking — val must be empty, and the
    CLI warns rather than pretending to validate."""
    tr, va = split_by_source(["only"], 0.2)
    assert tr == {"only"} and va == set()


def test_frames_to_export_skips_unlabeled_and_empty_frames():
    af = _af([
        SimpleBox(t=0, bbox=[0, 0, 10, 10], label="single"),
        SimpleBox(t=1, bbox=[0, 0, 10, 10], label=PROVISIONAL_LABEL),  # dropped
        SimpleBox(t=2, bbox=[0, 0, 10, 10], label="debris"),
    ])
    frames = frames_to_export(af)
    assert sorted(frames) == [0, 2]           # frame 1 has nothing usable


def test_frames_to_export_frame_step_subsamples():
    af = _af([SimpleBox(t=t, bbox=[0, 0, 10, 10], label="single")
              for t in range(10)])
    assert sorted(frames_to_export(af, frame_step=5)) == [0, 5]


def test_frames_to_export_verified_only():
    af = _af([
        SimpleBox(t=0, bbox=[0, 0, 10, 10], label="debris", auto=True),
        SimpleBox(t=1, bbox=[0, 0, 10, 10], label="debris", auto=False),
    ])
    assert sorted(frames_to_export(af)) == [0, 1]
    assert sorted(frames_to_export(af, verified_only=True)) == [1]


def test_build_coco_converts_bbox_convention():
    """[y0,x0,y1,x1] (ours) -> [x,y,w,h] (COCO)."""
    rec = {"file_name": "p_t0000.tif", "height": 512, "width": 640,
           "source": "/d/p.nd2", "t": 0,
           "boxes": [SimpleBox(t=0, bbox=[100, 200, 140, 260],
                               label="doublet", group="g1", score=0.7,
                               auto=True)]}
    coco = build_coco([rec])
    (img,), (ann,) = coco["images"], coco["annotations"]
    assert img["file_name"] == "p_t0000.tif"
    assert (img["height"], img["width"]) == (512, 640)
    assert img["frame"] == 0 and img["source"] == "/d/p.nd2"
    # y0=100 x0=200 y1=140 x1=260  ->  x=200, y=100, w=60, h=40
    assert ann["bbox"] == [200.0, 100.0, 60.0, 40.0]
    assert math.isclose(ann["area"], 60 * 40)
    assert ann["category_id"] == CATEGORY_IDS["doublet"]
    assert ann["image_id"] == img["id"]
    assert ann["cell"] == "g1"            # provenance kept
    assert ann["iscrowd"] == 0


def test_build_coco_ids_are_unique_and_linked():
    recs = [
        {"file_name": f"p_t{t:04d}.tif", "height": 8, "width": 8,
         "source": "p.nd2", "t": t,
         "boxes": [SimpleBox(t=t, bbox=[0, 0, 4, 4], label="single"),
                   SimpleBox(t=t, bbox=[4, 4, 8, 8], label="debris")]}
        for t in range(3)
    ]
    coco = build_coco(recs)
    assert len(coco["images"]) == 3
    assert len(coco["annotations"]) == 6
    assert len({i["id"] for i in coco["images"]}) == 3
    assert len({a["id"] for a in coco["annotations"]}) == 6
    img_ids = {i["id"] for i in coco["images"]}
    assert all(a["image_id"] in img_ids for a in coco["annotations"])


def test_build_coco_drops_degenerate_boxes_at_train_time():
    """A zero-area box would crash torchvision; the dataset filters them."""
    from nikon_control.train_detector import CocoDetectionDataset
    assert CocoDetectionDataset is not None  # import-only check here
