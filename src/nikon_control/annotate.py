"""ND2 annotation tool for cell-detection training data.

Opens an ND2 file in napari with one shape layer per class. The user draws
rectangles around cells in the appropriate class layer; bounding boxes are
saved to a JSON file next to the ND2.

CLI:

    nikon-control-annotate path/to/file.nd2
    nikon-control-annotate path/to/file.nd2 -o some/annotations.json
    nikon-control-annotate path/to/file.nd2 --classes single doublet dividing

Lifecycle model: each bbox is **persistent across the whole recording** by
default (cells don't move much, no need to redraw per frame). If a cell
dies, select its bbox and click "Mark death of selected at current T" — the
bbox's `t_end` is set to that frame, recording the cell's last alive frame.

Schema (stable, version 0.2):

    {
        "schema_version": "0.2",
        "source": "<path to nd2>",
        "image_shape": [...],
        "axes": ["T", "C", "Y", "X"],
        "channels": ["BF", "GFP", ...],
        "classes": ["single", "doublet", "debris"],
        "annotator": "",
        "annotations": [
            {
                "bbox": [y0, x0, y1, x1],
                "label": "single",
                "t_start": 0,
                "t_end": null,            # null = alive until end of recording
                "z": 0,
                "notes": "",
                "created": "2026-05-18T14:00:00"
            },
            ...
        ]
    }

v0.1 files (with a per-annotation ``t`` field) are auto-migrated on load:
``t`` is dropped, the annotation is treated as alive for the whole recording.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

SCHEMA_VERSION = "0.2"
DEFAULT_CLASSES: tuple[str, ...] = ("single", "doublet", "debris")
_LAYER_COLORS = ("red", "yellow", "cyan", "magenta", "lime")
_NO_DEATH = -1  # sentinel stored in napari properties for "alive until end"


@dataclass
class Annotation:
    bbox: list[float]  # [y0, x0, y1, x1]
    label: str
    t_start: int = 0
    t_end: int | None = None
    z: int = 0
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
    ann.schema_version = SCHEMA_VERSION
    payload = asdict(ann)
    payload["annotations"] = [asdict(a) for a in ann.annotations]
    path.write_text(json.dumps(payload, indent=2))


def _upgrade_annotation(a: dict) -> dict:
    """Migrate per-annotation dict from any older schema to v0.2."""
    a.pop("t", None)  # v0.1 had a single per-frame t; we no longer track it
    a.setdefault("t_start", 0)
    a.setdefault("t_end", None)
    return a


def load(path: Path) -> AnnotationFile:
    payload = json.loads(path.read_text())
    raw_anns = payload.pop("annotations", [])
    payload["schema_version"] = SCHEMA_VERSION
    af = AnnotationFile(**payload)
    af.annotations = [Annotation(**_upgrade_annotation(a)) for a in raw_anns]
    return af


def _shape_to_bbox(rect) -> list[float]:
    """Extract bbox=[y0,x0,y1,x1] from a 2D napari rectangle (4 vertices, 2D)."""
    import numpy as np

    rect = np.asarray(rect)
    ys = rect[:, -2]
    xs = rect[:, -1]
    return [float(ys.min()), float(xs.min()), float(ys.max()), float(xs.max())]


def _bbox_to_shape(bbox: list[float]) -> list[list[float]]:
    """Build a 4-vertex 2D rectangle in (y, x) order."""
    y0, x0, y1, x1 = bbox
    return [[y0, x0], [y0, x1], [y1, x1], [y1, x0]]


def open_for_annotation(
    nd2_path: Path,
    json_out: Path | None = None,
    classes: list[str] | None = None,
) -> None:
    """Open an ND2 file in napari and block until the window closes."""
    try:
        import napari
        import nd2 as nd2lib
        import numpy as np
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
        sizes = f.sizes
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
        non_channel_axes = [a for a in axes if a != "C"]
        t_dim_in_viewer = (
            non_channel_axes.index("T") if "T" in non_channel_axes else None
        )

        viewer = napari.Viewer(title=f"annotate: {nd2_path.name}")
        if channel_idx is not None:
            viewer.add_image(
                arr, channel_axis=channel_idx, name=state.channels or None
            )
        else:
            viewer.add_image(arr, name=nd2_path.stem)

        class_layers: dict[str, "napari.layers.Shapes"] = {}
        for i, cls in enumerate(state.classes):
            existing_shapes: list[list[list[float]]] = []
            existing_t_ends: list[int] = []
            for a in state.annotations:
                if a.label == cls:
                    existing_shapes.append(_bbox_to_shape(a.bbox))
                    existing_t_ends.append(
                        a.t_end if a.t_end is not None else _NO_DEATH
                    )
            props = {
                "t_end": np.array(existing_t_ends, dtype=int)
                if existing_t_ends
                else np.array([], dtype=int)
            }
            layer = viewer.add_shapes(
                existing_shapes if existing_shapes else None,
                shape_type="rectangle",
                edge_color=_LAYER_COLORS[i % len(_LAYER_COLORS)],
                face_color="transparent",
                edge_width=2,
                name=cls,
                ndim=2,
                properties=props,
            )
            try:
                layer.current_properties = {
                    "t_end": np.array([_NO_DEATH], dtype=int)
                }
            except Exception:
                pass
            class_layers[cls] = layer

        def _collect() -> list[Annotation]:
            out: list[Annotation] = []
            now = datetime.now().isoformat(timespec="seconds")
            for cls, layer in class_layers.items():
                t_ends = layer.properties.get(
                    "t_end", np.array([], dtype=int)
                )
                for i, rect in enumerate(layer.data):
                    t_end_raw = int(t_ends[i]) if i < len(t_ends) else _NO_DEATH
                    t_end = None if t_end_raw < 0 else t_end_raw
                    out.append(
                        Annotation(
                            bbox=_shape_to_bbox(rect),
                            label=cls,
                            t_start=0,
                            t_end=t_end,
                            created=now,
                        )
                    )
            return out

        def _active_class_layer():
            active = viewer.layers.selection.active
            if active is not None and active.name in class_layers:
                return active
            return None

        @magicgui(call_button="Save annotations")
        def save_widget() -> None:
            state.annotations = _collect()
            save(state, json_out)
            print(f"saved {len(state.annotations)} annotations -> {json_out}")

        @magicgui(call_button="Mark death of selected at current T")
        def mark_death_widget() -> None:
            if t_dim_in_viewer is None:
                print("no time axis in this file; nothing to mark")
                return
            layer = _active_class_layer()
            if layer is None:
                print("select a class shape layer (e.g. 'single') first")
                return
            if not layer.selected_data:
                print("select one or more shapes first")
                return
            current_t = int(viewer.dims.current_step[t_dim_in_viewer])
            t_ends = layer.properties["t_end"].copy()
            for i in layer.selected_data:
                t_ends[i] = current_t
            layer.properties = {"t_end": t_ends}
            print(f"marked {len(layer.selected_data)} shape(s) dead at T={current_t}")

        @magicgui(call_button="Clear death of selected")
        def clear_death_widget() -> None:
            layer = _active_class_layer()
            if layer is None or not layer.selected_data:
                return
            t_ends = layer.properties["t_end"].copy()
            for i in layer.selected_data:
                t_ends[i] = _NO_DEATH
            layer.properties = {"t_end": t_ends}
            print(f"cleared death of {len(layer.selected_data)} shape(s)")

        viewer.window.add_dock_widget(save_widget, name="Save", area="right")
        viewer.window.add_dock_widget(
            mark_death_widget, name="Mark death", area="right"
        )
        viewer.window.add_dock_widget(
            clear_death_widget, name="Clear death", area="right"
        )

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
