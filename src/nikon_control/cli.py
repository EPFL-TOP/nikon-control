from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(
        prog="nikon-control",
        description=(
            "Phase 0: load a 10x BF image and run a placeholder cell "
            "detection. Saves an overlay PNG when --out is given."
        ),
    )
    p.add_argument("image", type=Path, help="path to a .nd2 or .tif/.tiff file")
    p.add_argument("--out", type=Path, help="optional path for an overlay PNG")
    p.add_argument(
        "--min-area",
        type=int,
        default=50,
        help="minimum connected-component area in pixels (default: 50)",
    )
    args = p.parse_args()

    from .detect import detect_cells_bf
    from .io import load_image

    img = load_image(args.image)
    print(f"image: {args.image}")
    print(f"  shape: {img.shape}")
    print(f"  dtype: {img.dtype}")
    print(f"  range: [{img.min()}, {img.max()}]")

    det = detect_cells_bf(img, min_area_px=args.min_area)
    print(f"  detected (placeholder): {det.n} objects")

    if args.out:
        _save_overlay(img, det, args.out)
        print(f"  overlay -> {args.out}")


def _save_overlay(img, det, path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    vmin, vmax = np.percentile(img, [1, 99])
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    if det.n:
        ax.scatter(
            det.centroids[:, 1],
            det.centroids[:, 0],
            s=30,
            edgecolors="red",
            facecolors="none",
            linewidths=1,
        )
    ax.set_title(f"{det.n} objects (placeholder)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
