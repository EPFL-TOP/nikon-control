"""ND2 annotation tool for cell-detection training data.

Opens an ND2 file in napari with one shape layer per class. The user draws
rectangles around cells in the appropriate class layer; bounding boxes are
saved to a JSON file next to the ND2.

CLI:

    nikon-control-annotate path/to/file.nd2
    nikon-control-annotate path/to/file.nd2 -o some/annotations.json
    nikon-control-annotate path/to/file.nd2 --classes single doublet dividing

The schema is documented at the bottom of this module's docstring.

Schema (stable, version 0.1):

    {
        "schema_version": "0.1",
        "source": "<path to nd2>",
        "image_shape": [...],          # ND2 shape, e.g. [60, 3, 1024, 1024]
        "axes": ["T", "C", "Y", "X"],  # axis order matching image_shape
        "channels": ["BF", "GFP", ...],
        "classes": ["single", "doublet", "debris"],
        "annotator": "",
        "annotations": [
            {
                "t": 0, "z": 0,
                "bbox": [y0, x0, y1, x1],   # pixel coords, y/x order
                "label": "single",
                "notes": "",
                "created": "2026-05-18T14:00:00"
            },
            ...
        ]
    }
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

SCHEMA_VERSION = "0.1"
DEFAULT_CLASSES: tuple[str, ...] = ("single", "doublet", "debris")
_LAYER_COLORS = ("red", "yellow", "cyan", "magenta", "lime")


@dataclass
class Annotation:
    t: int
    z: int
    bbox: list[float]  # [y0, x0, y1, x1]
    label: str
    notes: str = ""
    created: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )


@dataclass
class AnnotationFile:
    source: str
    schema_version: str = SCHEMA_VERSION
    image_shape: list[int] = field(default_factory=list)
    axes: list[str] = field(default_factory=list)
    channels: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=lambda: list(DEFAULT_CLASSES))
    annotator: str = ""
    annotations: list[Annotation] = field(default_factory=list)


def save(ann: AnnotationFile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(ann)
    payload["annotations"] = [asdict(a) for a in ann.annotations]
    path.write_text(json.dumps(payload, indent=2))


def load(path: Path) -> AnnotationFile:
    payload = json.loads(path.read_text())
    raw_anns = payload.pop("annotations", [])
    af = AnnotationFile(**payload)
    af.annotations = [Annotation(**a) for a in raw_anns]
    return af


def _shape_to_bbox(
    rect, axes: list[str]
) -> tuple[int, int, list[float]]:
    """Extract (t, z, bbox=[y0,x0,y1,x1]) from a napari rectangle vertex array."""
    import numpy as np

    rect = np.asarray(rect)
    yi = axes.index("Y")
    xi = axes.index("X")
    t = int(round(float(rect[0, axes.index("T")]))) if "T" in axes else 0
    z = int(round(float(rect[0, axes.index("Z")]))) if "Z" in axes else 0
    ys = rect[:, yi]
    xs = rect[:, xi]
    return t, z, [float(ys.min()), float(xs.min()), float(ys.max()), float(xs.max())]


def _bbox_to_shape(t: int, z: int, bbox: list[float], axes: list[str]) -> list[list[float]]:
    """Build a 4-vertex rectangle in napari's per-axis coordinate order."""
    y0, x0, y1, x1 = bbox
    ndim = len(axes)
    def vertex(y, x):
        v = [0.0] * ndim
        if "T" in axes:
            v[axes.index("T")] = float(t)
        if "Z" in axes:
            v[axes.index("Z")] = float(z)
        v[axes.index("Y")] = float(y)
        v[axes.index("X")] = float(x)
        return v
    return [vertex(y0, x0), vertex(y0, x1), vertex(y1, x1), vertex(y1, x0)]


def open_for_annotation(
    nd2_path: Path,
    json_out: Path | None = None,
    classes: list[str] | None = None,
) -> None:
    """Open an ND2 file in napari and block until the window closes.

    On close (or when the Save button is clicked), annotations are written to
    `json_out` (default: <nd2>.annotations.json).
    """
    try:
        import napari
        import nd2 as nd2lib
        from magicgui import magicgui
    except ImportError as exc:
        raise ImportError(
            "Annotation requires the 'annotate' extra: "
            "pip install -e '.[annotate]'"
        ) from exc

    if json_out is None:
        json_out = nd2_path.with_suffix(".annotations.json")

    if json_out.exists():
        state = load(json_out)
        if classes is not None:
            for c in classes:
                if c not in state.classes:
                    state.classes.append(c)
    else:
        state = AnnotationFile(
            source=str(nd2_path),
            classes=list(classes) if classes else list(DEFAULT_CLASSES),
        )

    f = nd2lib.ND2File(str(nd2_path))
    try:
        arr = f.to_dask()
        sizes = f.sizes  # ordered dict like {'T': 60, 'C': 3, 'Y': 1024, 'X': 1024}
        axes = list(sizes.keys())
        state.image_shape = list(arr.shape)
        state.axes = axes
        try:
            state.channels = [
                str(c.channel.name) for c in (f.metadata.channels or [])
            ]
        except Exception:
            state.channels = []
        channel_idx = axes.index("C") if "C" in axes else None
        channel_names = state.channels or None

        viewer = napari.Viewer(title=f"annotate: {nd2_path.name}")
        if channel_idx is not None:
            viewer.add_image(arr, channel_axis=channel_idx, name=channel_names)
        else:
            viewer.add_image(arr, name=nd2_path.stem)

        non_channel_axes = [a for a in axes if a != "C"]

        class_layers: dict[str, "napari.layers.Shapes"] = {}
        for i, cls in enumerate(state.classes):
            existing = [
                _bbox_to_shape(a.t, a.z, a.bbox, non_channel_axes)
                for a in state.annotations
                if a.label == cls
            ]
            class_layers[cls] = viewer.add_shapes(
                existing if existing else None,
                shape_type="rectangle",
                edge_color=_LAYER_COLORS[i % len(_LAYER_COLORS)],
                face_color="transparent",
                edge_width=2,
                name=cls,
                ndim=len(non_channel_axes),
            )

        def _collect() -> list[Annotation]:
            out: list[Annotation] = []
            now = datetime.now().isoformat(timespec="seconds")
            for cls, layer in class_layers.items():
                for rect in layer.data:
                    t, z, bbox = _shape_to_bbox(rect, non_channel_axes)
                    out.append(
                        Annotation(t=t, z=z, bbox=bbox, label=cls, created=now)
                    )
            return out

        @magicgui(call_button="Save annotations")
        def save_widget() -> None:
            state.annotations = _collect()
            save(state, json_out)
            print(f"saved {len(state.annotations)} annotations -> {json_out}")

        viewer.window.add_dock_widget(save_widget, name="Save", area="right")
        napari.run()

        state.annotations = _collect()
        save(state, json_out)
    finally:
        f.close()


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        prog="nikon-control-annotate",
        description="Open an ND2 in napari for cell-detection annotation.",
    )
    p.add_argument("nd2", type=Path, help="path to an ND2 file")
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="JSON output path (default: <nd2>.annotations.json)",
    )
    p.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help=f"class labels (default: {' '.join(DEFAULT_CLASSES)})",
    )
    args = p.parse_args()
    open_for_annotation(args.nd2, args.out, args.classes)


if __name__ == "__main__":
    main()
