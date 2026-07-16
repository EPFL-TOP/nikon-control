"""Cell detection wrapping the trained torchvision Faster R-CNN.

The model in ``cell_detection_model.pth`` is a
``fasterrcnn_resnet50_fpn`` with a single foreground class ("cell"),
box-detection only. It was saved as a training checkpoint — the weights
live under the ``model_state_dict`` key alongside ``epoch``/``loss``/
``optimizer_state_dict``. ``load_checkpoint_state_dict`` handles both that
wrapped form and a raw ``state_dict``.

Bounding boxes are returned in the annotation-schema convention
``[y0, x0, y1, x1]`` (torchvision emits ``[x0, y0, x1, y1]``; we convert
here so nothing downstream has to think about it).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Detection:
    bbox: list[float]  # [y0, x0, y1, x1] in pixels
    score: float


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    y0, x0, y1, x1 = bbox
    return ((y0 + y1) / 2.0, (x0 + x1) / 2.0)


def point_in_bbox(cy: float, cx: float, bbox: list[float], margin: float = 0.0) -> bool:
    y0, x0, y1, x1 = bbox
    return (y0 - margin) <= cy <= (y1 + margin) and (x0 - margin) <= cx <= (x1 + margin)


def load_checkpoint_state_dict(obj) -> dict:
    """Return the model ``state_dict`` from either a wrapped training
    checkpoint (``{'model_state_dict': ...}`` / ``{'state_dict': ...}``) or a
    raw state_dict."""
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def infer_num_classes(state_dict: dict) -> int:
    """Read the number of classes (incl. background) from the box predictor."""
    w = state_dict["roi_heads.box_predictor.cls_score.weight"]
    return int(w.shape[0])


def _resolve_device(torch, requested: str | None) -> str:
    """Resolve the compute device, falling back with a loud warning rather
    than the cryptic 'Torch not compiled with CUDA enabled' assertion.

    - None  -> cuda if available, else cpu (mps is not auto-selected because
      some torchvision detection ops are unimplemented on MPS).
    - 'cuda' when unavailable (e.g. on a Mac) -> cpu, with a warning.
    - 'mps' when unavailable -> cpu, with a warning.
    """
    if requested is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    requested = requested.lower()
    if requested == "cuda" and not torch.cuda.is_available():
        print("WARNING: device 'cuda' requested but CUDA is unavailable "
              "(e.g. on macOS) — falling back to CPU.")
        return "cpu"
    if requested == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        print("WARNING: device 'mps' requested but unavailable — "
              "falling back to CPU.")
        return "cpu"
    return requested


def normalize_plane(
    plane: np.ndarray, pct: tuple[float, float] = (0.5, 99.5)
) -> np.ndarray:
    """Percentile-normalize a 2D plane to float32 [0, 1].

    Percentile clipping (rather than min/max) is robust to hot pixels and
    the bright well-border glue in brightfield frames.
    """
    p = plane.astype(np.float32)
    lo, hi = np.percentile(p, pct)
    out = np.clip((p - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    return out.astype(np.float32)


class DebrisDetector:
    """Motion-based debris detector — no trained model.

    Debris moves while cells stay roughly put, so a temporal-median
    background (built from the frames themselves) captures the static
    scene — including the cell and any fixed bubbles — and subtracting it
    leaves the moving/transient objects. Works on a whole stack at once
    (needs the median), unlike ``CellDetector`` which is per-frame.

    Parameters
    ----------
    sigma: gaussian smoothing of the residual before thresholding.
    k: threshold at ``mean + k*std`` of the residual.
    min_area / max_area: connected-component area gate (px).
    """

    def __init__(
        self,
        *,
        sigma: float = 2.0,
        k: float = 4.0,
        min_area: int = 60,
        max_area: int | None = None,
        bbox_pad: int = 20,
    ):
        self.sigma = sigma
        self.k = k
        self.min_area = min_area
        self.max_area = max_area
        self.bbox_pad = bbox_pad

    def _detect_against_bg(self, fr, bg) -> list[Detection]:
        from skimage import filters, measure

        H, W = fr.shape[-2], fr.shape[-1]
        resid = filters.gaussian(np.abs(fr.astype(np.float32) - bg), sigma=self.sigma)
        mask = resid > (resid.mean() + self.k * resid.std())
        # small-object removal is handled by the per-region area filter below
        pad = self.bbox_pad
        dets: list[Detection] = []
        for p in measure.regionprops(measure.label(mask)):
            if p.area < self.min_area:
                continue
            if self.max_area is not None and p.area > self.max_area:
                continue
            y0, x0, y1, x1 = p.bbox
            # pad the tight component bbox so the ROI comfortably encloses
            # the debris (residual thresholding tends to clip to the core),
            # clamped to the image.
            y0 = max(0.0, y0 - pad); x0 = max(0.0, x0 - pad)
            y1 = min(float(H), y1 + pad); x1 = min(float(W), x1 + pad)
            dets.append(
                Detection(bbox=[float(y0), float(x0), float(y1), float(x1)],
                          score=1.0)
            )
        return dets

    def detect_series(
        self, plane_fn, frames, bg_frames=None, progress_cb=None, exclude_fn=None
    ) -> list[list[Detection]]:
        """Detect over ``frames`` (list of frame indices), computing the
        median background from ``bg_frames`` (default: ``frames``).

        ``plane_fn(t) -> 2D ndarray`` loads one frame at a time, so only the
        background subsample is held in memory — a full 217-frame movie need
        not be loaded at once. ``progress_cb(done, total)`` is called per
        frame. ``exclude_fn(t, bbox) -> bool`` drops detections (e.g. debris
        whose centre lies inside a cell). Returns per-frame detections
        aligned to ``frames``.
        """
        frames = list(frames)
        bg_frames = list(bg_frames) if bg_frames is not None else frames
        bg = np.median(
            np.stack([np.asarray(plane_fn(t)).astype(np.float32) for t in bg_frames]),
            axis=0,
        )
        out: list[list[Detection]] = []
        for i, t in enumerate(frames, start=1):
            dets = self._detect_against_bg(np.asarray(plane_fn(t)), bg)
            if exclude_fn is not None:
                dets = [d for d in dets if not exclude_fn(t, d.bbox)]
            out.append(dets)
            if progress_cb is not None:
                progress_cb(i, len(frames))
        return out

    def detect_stack(self, planes) -> list[list[Detection]]:
        """Convenience wrapper: detect over an in-memory list of 2D planes."""
        planes = list(planes)
        return self.detect_series(lambda i: planes[i], range(len(planes)))


class CellDetector:
    """Runs the trained cell detector on 2D brightfield frames."""

    def __init__(
        self,
        weights_path: str | Path,
        *,
        device: str | None = None,
        score_threshold: float = 0.5,
        percentiles: tuple[float, float] = (0.5, 99.5),
        num_classes: int | None = None,
    ):
        import torch
        from torchvision.models.detection import fasterrcnn_resnet50_fpn

        self.score_threshold = score_threshold
        self.percentiles = percentiles
        self.device = _resolve_device(torch, device)

        ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        state = load_checkpoint_state_dict(ckpt)
        if num_classes is None:
            num_classes = infer_num_classes(state)
        self.num_classes = num_classes

        model = fasterrcnn_resnet50_fpn(
            weights=None, weights_backbone=None, num_classes=num_classes
        )
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint did not load cleanly: "
                f"{len(missing)} missing, {len(unexpected)} unexpected keys. "
                "Is this the right architecture / checkpoint?"
            )
        model.eval().to(device)
        self._model = model
        self._torch = torch

    def detect_frame(self, bf_plane: np.ndarray) -> list[Detection]:
        """Detect cells in a single 2D brightfield plane."""
        torch = self._torch
        img = normalize_plane(bf_plane, self.percentiles)
        tensor = torch.from_numpy(img)[None].repeat(3, 1, 1).to(self.device)
        with torch.no_grad():
            out = self._model([tensor])[0]
        boxes = out["boxes"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        dets: list[Detection] = []
        for (x0, y0, x1, y1), s in zip(boxes, scores):
            if s < self.score_threshold:
                continue
            dets.append(
                Detection(bbox=[float(y0), float(x0), float(y1), float(x1)],
                          score=float(s))
            )
        return dets

    def detect_stack(self, bf_frames) -> list[list[Detection]]:
        """Detect over an iterable of 2D planes; returns per-frame detections."""
        return [self.detect_frame(np.asarray(f)) for f in bf_frames]
