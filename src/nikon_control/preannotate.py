"""Pre-annotation pipeline: ND2 -> detections -> tracks -> annotation JSON.

Runs the trained cell detector on every timepoint of the brightfield
channel, links detections into tracks, and writes a v0.5 annotation file
that the dashboard opens for human review. Each detected object becomes a
provisional annotation labelled ``provisional_label`` (default "cell") —
the human then re-classifies it as single / doublet / etc.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .detector import CellDetector, DebrisDetector
from .schema import DEFAULT_CLASSES, Annotation, AnnotationFile, save
from .tracking import IoUTracker, VelocityTracker, tracks_to_annotations


def annotations_from_per_frame(
    per_frame, tracker, *, label: str, n_t: int,
    min_len: int = 1, simplify_tol: float = 5.0, t_offset: int = 0,
) -> list[Annotation]:
    """Track per-frame detections and convert the tracks to annotations."""
    tracks = tracker.track(per_frame)
    if t_offset:
        for tr in tracks:
            tr.frames = [fr + t_offset for fr in tr.frames]
    return tracks_to_annotations(
        tracks, label=label, n_frames=n_t,
        min_len=min_len, simplify_tol=simplify_tol,
    )


def detect_debris(
    plane_fn,
    bf_channel: int,
    n_t: int,
    *,
    t_range: tuple[int, int] | None = None,
    bg_stride: int = 8,
    sigma: float = 2.0,
    k: float = 4.0,
    min_area: int = 150,
    max_area: int | None = None,
    bbox_pad: int = 20,
    max_dist: float = 500.0,
    max_age: int = 2,
    min_len: int = 3,
    simplify_tol: float = 4.0,
    label: str = "debris",
    progress_cb=None,
    exclude_fn=None,
) -> list[Annotation]:
    """Motion-based debris detection + velocity tracking over a frame range.

    Builds the median background from every ``bg_stride``-th frame in the
    range (bounded memory), detects transients per frame, and links them
    with a velocity tracker (debris moves too fast for IoU). ``min_len``
    drops brief flickers; raise it to ignore more noise.
    """
    lo, hi = (0, n_t) if t_range is None else t_range
    hi = min(hi, n_t)
    frames = list(range(lo, hi))
    bg_frames = frames[::max(1, bg_stride)] or frames
    det = DebrisDetector(sigma=sigma, k=k, min_area=min_area,
                         max_area=max_area, bbox_pad=bbox_pad)
    per_frame = det.detect_series(
        lambda t: plane_fn(t, bf_channel), frames, bg_frames=bg_frames,
        progress_cb=progress_cb, exclude_fn=exclude_fn,
    )
    tracker = VelocityTracker(max_dist=max_dist, max_age=max_age)
    return annotations_from_per_frame(
        per_frame, tracker, label=label, n_t=n_t,
        min_len=min_len, simplify_tol=simplify_tol, t_offset=lo,
    )


def detect_and_track(
    detector: CellDetector,
    plane_fn,
    bf_channel: int,
    n_t: int,
    *,
    provisional_label: str = "cell",
    iou_threshold: float = 0.3,
    max_age: int = 3,
    min_len: int = 1,
    simplify_tol: float = 5.0,
    t_range: tuple[int, int] | None = None,
    progress_cb=None,
) -> list[Annotation]:
    """Run detect+track over frames and return annotations.

    ``plane_fn(t, c) -> 2D ndarray`` supplies frames; sharing this signature
    lets the dashboard reuse its already-open ND2 (and a cached detector)
    instead of re-opening the file. ``progress_cb(done, total)`` is called
    after each frame so a GUI can show progress.
    """
    lo, hi = (0, n_t) if t_range is None else t_range
    hi = min(hi, n_t)
    per_frame = []
    for t in range(lo, hi):
        per_frame.append(detector.detect_frame(plane_fn(t, bf_channel)))
        if progress_cb is not None:
            progress_cb(t - lo + 1, hi - lo)
    tracker = IoUTracker(iou_threshold=iou_threshold, max_age=max_age)
    return annotations_from_per_frame(
        per_frame, tracker, label=provisional_label, n_t=n_t,
        min_len=min_len, simplify_tol=simplify_tol, t_offset=lo,
    )


def _find_bf_channel(channels: list[str]) -> int:
    """Index of the brightfield channel by name, else 0."""
    for i, name in enumerate(channels):
        if name and "bf" in name.lower():
            return i
    for i, name in enumerate(channels):
        if name and ("bright" in name.lower() or "trans" in name.lower()):
            return i
    return 0


def _plane_extractor(arr, axes: list[str]):
    """Return a function plane(t, c) -> 2D ndarray for the given axis order."""
    def plane(t: int, c: int) -> np.ndarray:
        idx: list = []
        for ax in axes:
            if ax == "T":
                idx.append(t)
            elif ax == "C":
                idx.append(c)
            elif ax in ("Y", "X"):
                idx.append(slice(None))
            else:  # Z or anything else -> first plane
                idx.append(0)
        return np.asarray(arr[tuple(idx)])

    return plane


def preannotate_nd2(
    nd2_path: str | Path,
    weights_path: str | Path,
    *,
    json_out: str | Path | None = None,
    bf_channel: int | None = None,
    provisional_label: str = "cell",
    score_threshold: float = 0.5,
    device: str | None = None,
    iou_threshold: float = 0.3,
    max_age: int = 3,
    min_len: int = 1,
    simplify_tol: float = 2.0,
    t_range: tuple[int, int] | None = None,
    progress: bool = True,
) -> AnnotationFile:
    """Detect + track cells in an ND2 and return (and optionally save) an
    :class:`AnnotationFile`.

    ``t_range`` limits the frames processed (useful for a quick test).
    """
    import nd2

    nd2_path = Path(nd2_path)
    if json_out is None:
        json_out = nd2_path.with_suffix(".annotations.json")

    f = nd2.ND2File(str(nd2_path))
    try:
        sizes = dict(f.sizes)
        axes = list(sizes.keys())
        arr = f.to_dask()
        try:
            channels = [str(c.channel.name) for c in (f.metadata.channels or [])]
        except Exception:
            channels = []
        n_t = sizes.get("T", 1)
        n_c = sizes.get("C", 1)
        if bf_channel is None:
            bf_channel = _find_bf_channel(channels) if channels else 0
        bf_channel = min(bf_channel, n_c - 1)

        plane = _plane_extractor(arr, axes)
        lo, hi = (0, n_t) if t_range is None else t_range
        hi = min(hi, n_t)

        detector = CellDetector(
            weights_path,
            device=device,
            score_threshold=score_threshold,
        )

        def _progress(done, total):
            if progress and (done - 1) % 20 == 0:
                print(f"  frame {done}/{total}")

        anns = detect_and_track(
            detector, plane, bf_channel, n_t,
            provisional_label=provisional_label,
            iou_threshold=iou_threshold, max_age=max_age,
            min_len=min_len, simplify_tol=simplify_tol,
            t_range=(lo, hi), progress_cb=_progress,
        )

        classes = [provisional_label] + [
            c for c in DEFAULT_CLASSES if c != provisional_label
        ]
        af = AnnotationFile(
            source=str(nd2_path),
            image_shape=list(arr.shape),
            axes=axes,
            channels=channels,
            classes=classes,
            annotations=anns,
        )
        if json_out:
            save(af, Path(json_out))
        return af
    finally:
        f.close()


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        prog="nikon-control-preannotate",
        description="Pre-annotate an ND2 with the trained cell detector.",
    )
    p.add_argument("nd2", type=Path)
    p.add_argument("weights", type=Path, help="path to cell_detection_model.pth")
    p.add_argument("-o", "--out", type=Path, default=None)
    p.add_argument("--score", type=float, default=0.5)
    p.add_argument("--bf-channel", type=int, default=None)
    p.add_argument("--device", default=None, help="cuda / cpu (auto if omitted)")
    p.add_argument(
        "--t-range", type=int, nargs=2, default=None, metavar=("START", "STOP")
    )
    args = p.parse_args()

    af = preannotate_nd2(
        args.nd2,
        args.weights,
        json_out=args.out,
        bf_channel=args.bf_channel,
        score_threshold=args.score,
        device=args.device,
        t_range=tuple(args.t_range) if args.t_range else None,
    )
    print(
        f"pre-annotated {len(af.annotations)} track(s) "
        f"-> {args.out or Path(args.nd2).with_suffix('.annotations.json')}"
    )


if __name__ == "__main__":
    main()
