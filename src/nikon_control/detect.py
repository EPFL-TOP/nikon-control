"""Phase 0 placeholder brightfield cell detector.

A real BF segmenter (Cellpose, Stardist, or a trained model) lands in Phase 1.
This crude high-pass + Otsu pipeline exists only so the end-to-end loop can be
visually sanity-checked on a real 10x BF image.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    centroids: np.ndarray
    labels: np.ndarray
    n: int


def detect_cells_bf(image: np.ndarray, min_area_px: int = 50) -> Detection:
    from skimage import filters, measure, morphology

    if image.ndim != 2:
        raise ValueError(f"expected 2D image, got shape {image.shape}")

    img = image.astype(np.float32)
    blur = filters.gaussian(img, sigma=2)
    detail = np.abs(img - blur)
    mask = detail > filters.threshold_otsu(detail)
    mask = morphology.closing(mask, morphology.disk(3))

    raw = measure.label(mask)
    kept = [p for p in measure.regionprops(raw) if p.area >= min_area_px]

    if not kept:
        return Detection(
            centroids=np.empty((0, 2)),
            labels=np.zeros_like(raw),
            n=0,
        )

    keep = np.zeros(raw.max() + 1, dtype=bool)
    for p in kept:
        keep[p.label] = True
    labels = np.where(keep[raw], raw, 0)
    centroids = np.array([p.centroid for p in kept])
    return Detection(centroids=centroids, labels=labels, n=len(kept))
