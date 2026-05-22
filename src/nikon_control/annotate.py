"""ND2 annotation tool for cell-detection training data.

Opens an ND2 file in napari with one shape layer per class. The user draws
rectangles around cells in the appropriate class layer; bounding boxes are
saved to a JSON file next to the ND2.

CLI:

    nikon-control-annotate path/to/file.nd2
    nikon-control-annotate path/to/file.nd2 -o some/annotations.json
    nikon-control-annotate path/to/file.nd2 --classes single doublet dividing

Lifecycle model: each bbox is **persistent across the whole recording** by
default (cells don't move much, no need to redraw per frame). The bbox is
considered alive on every frame ``t`` with ``t_start <= t <= t_end`` (where
``t_end`` defaults to "end of recording" if not set).

- If a cell **appears** partway through (mitosis, drifts in, etc.), select
  its bbox at the frame of appearance and click "Mark birth of selected at
  current T". Its ``t_start`` is recorded.
- If a cell **dies**, select its bbox at the last alive frame and click
  "Mark death of selected at current T". Its ``t_end`` is recorded.

Shapes outside their alive range are dimmed (25 % opacity) so the user can
still see them spatially while making clear they aren't "alive" at the
current T.

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


def _life_label(t_start: int, t_end: int) -> str:
    """Text shown above bboxes inside napari — birth and/or death markers."""
    parts = []
    if t_start > 0:
        parts.append(f"↑T={t_start}")
    if t_end >= 0:
        parts.append(f"†T={t_end}")
    return " ".join(parts)


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
        from matplotlib.colors import to_rgba
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

        class_colors: dict[str, str] = {
            cls: _LAYER_COLORS[i % len(_LAYER_COLORS)]
            for i, cls in enumerate(state.classes)
        }
        class_layers: dict[str, "napari.layers.Shapes"] = {}
        for cls, base_color in class_colors.items():
            existing_shapes: list[list[list[float]]] = []
            existing_t_starts: list[int] = []
            existing_t_ends: list[int] = []
            for a in state.annotations:
                if a.label == cls:
                    existing_shapes.append(_bbox_to_shape(a.bbox))
                    existing_t_starts.append(a.t_start)
                    existing_t_ends.append(
                        a.t_end if a.t_end is not None else _NO_DEATH
                    )
            props = {
                "t_start": np.array(existing_t_starts or [], dtype=int),
                "t_end": np.array(existing_t_ends or [], dtype=int),
                "life_label": np.array(
                    [
                        _life_label(s, e)
                        for s, e in zip(existing_t_starts, existing_t_ends)
                    ],
                    dtype=object,
                ),
            }
            layer = viewer.add_shapes(
                existing_shapes if existing_shapes else None,
                shape_type="rectangle",
                edge_color=base_color,
                face_color="transparent",
                edge_width=2,
                name=cls,
                ndim=2,
                properties=props,
                text={
                    "string": "{life_label}",
                    "size": 12,
                    "color": base_color,
                    "anchor": "upper_left",
                    "translation": [-12, 0],
                },
            )
            try:
                layer.current_properties = {
                    "t_start": np.array([0], dtype=int),
                    "t_end": np.array([_NO_DEATH], dtype=int),
                    "life_label": np.array([""], dtype=object),
                }
            except Exception:
                pass
            class_layers[cls] = layer

        def _refresh_one_layer(layer) -> None:
            """Dim shapes whose lifecycle excludes the current T."""
            n = len(layer.data)
            if n == 0:
                return
            current_t = (
                int(viewer.dims.current_step[t_dim_in_viewer])
                if t_dim_in_viewer is not None
                else 0
            )
            t_starts = layer.properties.get(
                "t_start", np.zeros(n, dtype=int)
            )
            t_ends = layer.properties.get(
                "t_end", np.full(n, _NO_DEATH, dtype=int)
            )
            base = class_colors.get(layer.name, "red")
            colors = np.zeros((n, 4))
            for i in range(n):
                ts = int(t_starts[i]) if i < len(t_starts) else 0
                te = int(t_ends[i]) if i < len(t_ends) else _NO_DEATH
                alive_end = te if te >= 0 else 10**9
                alpha = 1.0 if ts <= current_t <= alive_end else 0.25
                colors[i] = to_rgba(base, alpha=alpha)
            layer.edge_color = colors

        def _refresh_all_layers(event=None) -> None:
            for layer in class_layers.values():
                _refresh_one_layer(layer)

        viewer.dims.events.current_step.connect(_refresh_all_layers)
        for _layer in class_layers.values():
            _layer.events.data.connect(
                lambda e=None, l=_layer: _refresh_one_layer(l)
            )
        _refresh_all_layers()

        def _collect() -> list[Annotation]:
            out: list[Annotation] = []
            now = datetime.now().isoformat(timespec="seconds")
            for cls, layer in class_layers.items():
                t_starts = layer.properties.get(
                    "t_start", np.array([], dtype=int)
                )
                t_ends = layer.properties.get(
                    "t_end", np.array([], dtype=int)
                )
                for i, rect in enumerate(layer.data):
                    t_start = int(t_starts[i]) if i < len(t_starts) else 0
                    t_end_raw = int(t_ends[i]) if i < len(t_ends) else _NO_DEATH
                    t_end = None if t_end_raw < 0 else t_end_raw
                    out.append(
                        Annotation(
                            bbox=_shape_to_bbox(rect),
                            label=cls,
                            t_start=t_start,
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

        def _report(msg: str) -> None:
            """Show a message in the napari status bar AND stdout."""
            viewer.status = msg
            print(msg)

        def _apply_lifecycle(
            layer, indices: list[int], field: str, value: int
        ) -> bool:
            """Write `value` into the `field` property at `indices`.

            `field` is either ``"t_start"`` or ``"t_end"``. Returns True if the
            write was persisted (verified by read-back).
            """
            arr = layer.properties[field].copy()
            for i in indices:
                arr[i] = value
            new_props = dict(layer.properties)
            new_props[field] = arr
            t_starts = new_props["t_start"]
            t_ends = new_props["t_end"]
            new_props["life_label"] = np.array(
                [
                    _life_label(int(t_starts[i]), int(t_ends[i]))
                    for i in range(len(arr))
                ],
                dtype=object,
            )
            layer.properties = new_props
            layer.refresh()
            _refresh_one_layer(layer)
            read_back = layer.properties[field]
            return all(int(read_back[i]) == value for i in indices)

        def _require_selection(action_name: str):
            layer = _active_class_layer()
            if layer is None:
                _report(
                    "first click a class shape layer "
                    "(single/doublet/...) in the layer list"
                )
                return None, None
            if not layer.selected_data:
                _report(
                    f"no shape selected. To {action_name}: switch to the "
                    "'select shapes' tool (arrow icon in the layer toolbar), "
                    "click a rectangle, then click this button."
                )
                return None, None
            return layer, list(layer.selected_data)

        @magicgui(call_button="Save annotations")
        def save_widget() -> None:
            state.annotations = _collect()
            save(state, json_out)
            n_dead = sum(1 for a in state.annotations if a.t_end is not None)
            n_born_late = sum(1 for a in state.annotations if a.t_start > 0)
            _report(
                f"saved {len(state.annotations)} annotations "
                f"({n_born_late} with birth, {n_dead} with death) -> {json_out}"
            )

        @magicgui(call_button="Mark birth of selected at current T")
        def mark_birth_widget() -> None:
            if t_dim_in_viewer is None:
                _report("this file has no time axis; nothing to mark")
                return
            layer, indices = _require_selection("mark birth")
            if layer is None:
                return
            current_t = int(viewer.dims.current_step[t_dim_in_viewer])
            if not _apply_lifecycle(layer, indices, "t_start", current_t):
                _report("WARNING: birth mark did NOT persist on read-back.")
                return
            _report(
                f"marked {len(indices)} shape(s) in '{layer.name}' "
                f"as born at T={current_t}"
            )

        @magicgui(call_button="Clear birth of selected (reset to T=0)")
        def clear_birth_widget() -> None:
            layer, indices = _require_selection("clear birth")
            if layer is None:
                return
            if not _apply_lifecycle(layer, indices, "t_start", 0):
                _report("WARNING: clear-birth did NOT persist on read-back.")
                return
            _report(
                f"reset birth to T=0 for {len(indices)} shape(s) "
                f"in '{layer.name}'"
            )

        @magicgui(call_button="Mark death of selected at current T")
        def mark_death_widget() -> None:
            if t_dim_in_viewer is None:
                _report("this file has no time axis; nothing to mark")
                return
            layer, indices = _require_selection("mark death")
            if layer is None:
                return
            current_t = int(viewer.dims.current_step[t_dim_in_viewer])
            if not _apply_lifecycle(layer, indices, "t_end", current_t):
                _report("WARNING: death mark did NOT persist on read-back.")
                return
            _report(
                f"marked {len(indices)} shape(s) in '{layer.name}' "
                f"as dead at T={current_t}"
            )

        @magicgui(call_button="Clear death of selected")
        def clear_death_widget() -> None:
            layer, indices = _require_selection("clear death")
            if layer is None:
                return
            if not _apply_lifecycle(layer, indices, "t_end", _NO_DEATH):
                _report("WARNING: clear-death did NOT persist on read-back.")
                return
            _report(
                f"cleared death of {len(indices)} shape(s) in '{layer.name}'"
            )

        viewer.window.add_dock_widget(save_widget, name="Save", area="right")
        viewer.window.add_dock_widget(
            mark_birth_widget, name="Mark birth", area="right"
        )
        viewer.window.add_dock_widget(
            clear_birth_widget, name="Clear birth", area="right"
        )
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
