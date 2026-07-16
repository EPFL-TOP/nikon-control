import numpy as np

from nikon_control.detector import (
    Detection,
    DebrisDetector,
    bbox_center,
    point_in_bbox,
)
from nikon_control.tracking import VelocityTracker, tracks_to_annotations


def _d(cy, cx, half=15):
    return Detection(bbox=[cy - half, cx - half, cy + half, cx + half], score=1.0)


def test_velocity_tracker_links_fast_linear_mover():
    # object moving +200px/frame in x, +150 in y — IoU would be 0 between
    # frames, but the velocity tracker should keep it as one track
    per_frame = [[_d(100 + 150 * t, 100 + 200 * t)] for t in range(6)]
    tracks = VelocityTracker(max_dist=500).track(per_frame)
    assert len(tracks) == 1
    assert tracks[0].frames == [0, 1, 2, 3, 4, 5]


def test_velocity_tracker_keeps_two_crossing_objects_separate():
    # two movers; each consistent in its own direction
    per_frame = [
        [_d(100 + 120 * t, 100 + 120 * t), _d(1000 - 120 * t, 100 + 120 * t)]
        for t in range(5)
    ]
    tracks = VelocityTracker(max_dist=400).track(per_frame)
    assert len(tracks) == 2
    assert all(len(tr.frames) == 5 for tr in tracks)


def test_velocity_tracker_static_track_does_not_jump_to_far_transient():
    # a static blob at (1000,1000); at t=3 its own detection is missing and a
    # transient appears 350px away (within max_dist but far). Adaptive gating
    # must NOT let the established static track jump to it (the "static debris
    # that moves" bug).
    per_frame = [
        [_d(1000, 1000)],
        [_d(1000, 1000)],
        [_d(1000, 1000)],
        [_d(1000, 650)],   # static missing this frame; transient 350px away
        [_d(1000, 1000)],
    ]
    tracks = VelocityTracker(max_dist=500, base_gate=100).track(per_frame)
    static = max(tracks, key=lambda t: len(t.frames))
    xs = [(b[1] + b[3]) / 2 for b in static.bboxes]
    # the static track's x-centres stay near 1000 — it never jumps to ~650
    assert all(abs(x - 1000) < 60 for x in xs)


def test_velocity_tracker_bridges_one_missed_frame():
    per_frame = [
        [_d(100, 100)],
        [_d(200, 200)],
        [],              # missed
        [_d(400, 400)],
    ]
    tracks = VelocityTracker(max_dist=400, max_age=2).track(per_frame)
    assert len(tracks) == 1
    assert tracks[0].frames == [0, 1, 3]


def test_debris_detector_finds_moving_blob_not_static_background():
    rng = np.random.default_rng(0)
    H = W = 200
    base = (rng.normal(1000, 5, (H, W))).astype(np.float32)
    # a static bright square (like a fixed structure) — should be suppressed
    base[20:40, 20:40] += 400
    planes = []
    for t in range(8):
        fr = base.copy()
        # a moving bright blob crossing the frame
        cy, cx = 100 + 8 * t, 30 + 20 * t
        fr[cy - 8:cy + 8, cx - 8:cx + 8] += 600
        planes.append(fr)

    det = DebrisDetector(sigma=1.5, k=3.0, min_area=30)
    per_frame = det.detect_stack(planes)
    # every frame should find the mover
    assert all(len(dets) >= 1 for dets in per_frame)
    # and the detection should follow the mover, not sit on the static square
    for t, dets in enumerate(per_frame):
        cy_expected = 100 + 8 * t
        centers_y = [ (d.bbox[0] + d.bbox[2]) / 2 for d in dets ]
        assert any(abs(cy - cy_expected) < 20 for cy in centers_y)


def test_bbox_center_and_point_in_bbox():
    assert bbox_center([10, 20, 30, 60]) == (20.0, 40.0)
    assert point_in_bbox(20, 40, [10, 20, 30, 60])
    assert not point_in_bbox(5, 40, [10, 20, 30, 60])
    # margin extends the box
    assert point_in_bbox(8, 40, [10, 20, 30, 60], margin=5)


def test_detect_series_exclude_fn_drops_detections_inside_region():
    rng = np.random.default_rng(1)
    H = W = 200
    base = rng.normal(1000, 5, (H, W)).astype(np.float32)
    planes = []
    for t in range(6):
        fr = base.copy()
        # a bright blob sitting at a FIXED location that we will exclude
        fr[95:115, 95:115] += 700
        planes.append(fr)
    det = DebrisDetector(sigma=1.5, k=3.0, min_area=30)
    # exclude anything centred inside the box around (105,105)
    excl = lambda t, b: point_in_bbox(*bbox_center(b), [90, 90, 120, 120])
    per_frame = det.detect_series(lambda i: planes[i], range(len(planes)),
                                  exclude_fn=excl)
    assert all(len(d) == 0 for d in per_frame)  # all excluded


def test_debris_min_area_drops_small_blobs():
    rng = np.random.default_rng(2)
    H = W = 200
    base = rng.normal(1000, 5, (H, W)).astype(np.float32)
    planes = []
    for t in range(6):
        fr = base.copy()
        fr[100:104, 100:104] += 700   # tiny 4x4 blob (area ~16)
        planes.append(fr)
    # min_area well above the blob size -> nothing detected
    det = DebrisDetector(sigma=1.0, k=3.0, min_area=200)
    per_frame = det.detect_stack(planes)
    assert all(len(d) == 0 for d in per_frame)


def test_debris_end_to_end_to_annotations():
    per_frame = [[_d(100 + 200 * t, 100 + 150 * t)] for t in range(5)]
    tracks = VelocityTracker(max_dist=500).track(per_frame)
    anns = tracks_to_annotations(tracks, label="debris", n_frames=5, min_len=3)
    assert len(anns) == 1
    assert anns[0].label == "debris"
    assert anns[0].t_start == 0
    assert len(anns[0].keyframes) >= 2  # a moving track keeps inflections
