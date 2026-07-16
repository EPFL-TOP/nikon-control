from nikon_control.detector import Detection
from nikon_control.tracking import (
    IoUTracker,
    _rdp_simplify,
    iou,
    tracks_to_annotations,
)


def _d(y0, x0, y1, x1, score=0.9):
    return Detection(bbox=[y0, x0, y1, x1], score=score)


def test_iou_basic():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    # half-overlap in x: intersection 5x10=50, union 100+100-50=150
    assert abs(iou([0, 0, 10, 10], [0, 5, 10, 15]) - (50 / 150)) < 1e-9


def test_tracker_links_static_cell_across_frames():
    # one near-static cell over 5 frames
    per_frame = [[_d(100, 100, 210, 210)] for _ in range(5)]
    tracks = IoUTracker(iou_threshold=0.3).track(per_frame)
    assert len(tracks) == 1
    assert tracks[0].frames == [0, 1, 2, 3, 4]


def test_tracker_separates_two_cells():
    per_frame = [
        [_d(0, 0, 50, 50), _d(200, 200, 260, 260)] for _ in range(4)
    ]
    tracks = IoUTracker(iou_threshold=0.3).track(per_frame)
    assert len(tracks) == 2
    assert all(len(t.frames) == 4 for t in tracks)


def test_tracker_bridges_missed_frame_with_max_age():
    # detection missing at t=2, present t=0,1,3,4
    per_frame = [
        [_d(10, 10, 60, 60)],
        [_d(10, 10, 60, 60)],
        [],
        [_d(10, 10, 60, 60)],
        [_d(10, 10, 60, 60)],
    ]
    tracks = IoUTracker(iou_threshold=0.3, max_age=3).track(per_frame)
    assert len(tracks) == 1  # not split by the gap
    assert tracks[0].frames == [0, 1, 3, 4]


def test_tracker_splits_when_gap_exceeds_max_age():
    per_frame = [
        [_d(10, 10, 60, 60)],
        [],
        [],
        [],
        [_d(10, 10, 60, 60)],
    ]
    tracks = IoUTracker(iou_threshold=0.3, max_age=2).track(per_frame)
    assert len(tracks) == 2


def test_rdp_collapses_static_trajectory():
    frames = list(range(6))
    bboxes = [[0, 0, 10, 10]] * 6
    keep = _rdp_simplify(frames, bboxes, tol=2.0)
    assert keep == [0, 5]  # endpoints only


def test_rdp_keeps_moving_inflection():
    # linear drift then a jump — the corner must survive
    frames = [0, 1, 2, 3, 4]
    bboxes = [
        [0, 0, 10, 10],
        [0, 1, 10, 11],
        [0, 2, 10, 12],
        [0, 20, 10, 30],  # jump
        [0, 21, 10, 31],
    ]
    keep = _rdp_simplify(frames, bboxes, tol=2.0)
    assert 3 in keep or 2 in keep  # the inflection is preserved


def test_tracks_to_annotations_static_is_single_keyframe():
    per_frame = [[_d(100, 100, 210, 210)] for _ in range(10)]
    tracks = IoUTracker(iou_threshold=0.3).track(per_frame)
    anns = tracks_to_annotations(
        tracks, label="cell", n_frames=10, simplify_tol=2.0
    )
    assert len(anns) == 1
    assert len(anns[0].keyframes) == 1  # collapsed to one
    assert anns[0].t_start == 0
    assert anns[0].t_end is None  # ran to the last frame


def test_tracks_to_annotations_sets_t_end_when_track_ends_early():
    per_frame = [[_d(0, 0, 50, 50)] for _ in range(5)] + [[] for _ in range(5)]
    tracks = IoUTracker(iou_threshold=0.3, max_age=1).track(per_frame)
    anns = tracks_to_annotations(tracks, label="cell", n_frames=10)
    assert len(anns) == 1
    assert anns[0].t_end == 4  # last detected frame, before the recording ends


def test_min_len_drops_noise():
    per_frame = [[_d(0, 0, 20, 20)]] + [[] for _ in range(5)]
    tracks = IoUTracker(iou_threshold=0.3, max_age=0).track(per_frame)
    anns = tracks_to_annotations(tracks, label="cell", n_frames=6, min_len=2)
    assert anns == []
