from __future__ import annotations

from pathlib import Path

import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    """Load a 2D image from a .nd2 or .tif/.tiff file.

    Multi-dim files (channels, z, time) are collapsed to the first plane —
    Phase 0 only needs a single 10x BF frame.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".nd2":
        try:
            import nd2  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Reading .nd2 files requires the 'nd2' extra: "
                "pip install 'nikon-control[nd2]'"
            ) from exc
        with nd2.ND2File(str(path)) as f:
            arr = f.asarray()
    elif suffix in (".tif", ".tiff"):
        import tifffile

        arr = tifffile.imread(str(path))
    else:
        raise ValueError(f"unsupported image format: {suffix!r}")

    while arr.ndim > 2:
        arr = arr[0]
    return arr
