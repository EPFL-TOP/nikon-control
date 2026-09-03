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


def plane_extractor(arr, axes: list[str]):
    """Build ``plane(t, c) -> 2D ndarray`` for an ND2's (dask) array."""

    def plane(t: int, c: int) -> np.ndarray:
        idx: list = []
        for ax in axes:
            if ax == "T":
                idx.append(int(t))
            elif ax == "C":
                idx.append(int(c))
            elif ax in ("Y", "X"):
                idx.append(slice(None))
            else:
                idx.append(0)
        return np.asarray(arr[tuple(idx)])

    return plane


def bf_channel_index(channels: list[str]) -> int:
    """Index of the brightfield channel (the one detection runs on)."""
    return next((i for i, c in enumerate(channels) if "bf" in c.lower()), 0)


def open_nd2(path: str | Path) -> dict:
    """Open an ND2 time series and return what callers need from it.

    Used by both dashboards and by the training exporter. The caller owns
    ``file`` and must close it.
    """
    import nd2

    f = nd2.ND2File(str(path))
    sizes = dict(f.sizes)
    axes = list(sizes.keys())
    arr = f.to_dask()
    try:
        channels = [str(cc.channel.name) for cc in (f.metadata.channels or [])]
    except Exception:
        channels = []
    n_t = sizes.get("T", 1)
    n_c = sizes.get("C", 1)
    if not channels:
        channels = [f"C{i}" for i in range(n_c)]
    return {
        "file": f,
        "arr": arr,
        "axes": axes,
        "sizes": sizes,
        "channels": channels,
        "n_t": n_t,
        "n_c": n_c,
        "H": sizes.get("Y", arr.shape[-2]),
        "W": sizes.get("X", arr.shape[-1]),
        "bf_index": bf_channel_index(channels),
        "plane": plane_extractor(arr, axes),
    }
