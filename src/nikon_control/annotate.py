"""ND2 annotation tool for cell-detection training data.

Opens an ND2 file in napari with one shape layer per class. The user draws
rectangles around cells in the appropriate class layer; bounding boxes are
saved to a JSON file next to the ND2.

CLI:

    nikon-control-annotate path/to/file.nd2
    nikon-control-annotate path/to/file.nd2 -o some/annotations.json
    nikon-control-annotate path/to/file.nd2 --classes single doublet dividing

Lifecycle model: each bbox is **persistent across the whole recording** by
default (cells don't move much, no need to redraw per frame). Three
distinct fields describe its lifecycle:

- ``t_start`` — first frame the cell/debris is visible. Defaults to 0.
- ``t_end``   — last frame visible (e.g. cell drifts out of FOV).
  Defaults to ``None`` = visible until end of recording.
- ``t_death`` — frame the cell is marked dead (the cell may still be
  visible as a corpse afterwards). Defaults to ``None``.

In the napari viewer:

- A bbox is **hidden** outside its visibility range ``[t_start, t_end]``.
- A ``↑T=N`` label sits above the bbox whenever it's visible and
  ``t_start > 0``.
- A ``†T=N`` label appears alongside whenever the current frame is at or
  after ``t_death`` (and the bbox is still visible).

Schema (stable, version 0.4):

    {
        "schema_version": "0.4",
        "source": "<path to nd2>",
        "image_shape": [...],
        "axes": ["T", "C", "Y", "X"],
        "channels": ["BF", "GFP", "DAPI"],
        "classes": ["single", "doublet", "debris", "fission_fusion"],
        "annotator": "",
        "annotations": [
            {
                "bbox": [y0, x0, y1, x1],
                "label": "single",
                "t_start": 0,
                "t_end": null,            # null = visible until end of recording
                "t_deaths": [],           # one entry per dying cell; [] = none
                "z": 0,
                "notes": "",
                "created": "2026-06-11T14:00:00"
            },
            ...
        ]
    }

Older files are auto-migrated on load:

- v0.1: the per-annotation ``t`` field is dropped; annotation treated as
  alive for the whole recording.
- v0.2: the old ``t_end`` was the death marker; it's moved into ``t_deaths``
  as a single-element list and the new ``t_end`` (visibility end) is left
  ``None``.
- v0.3: a single ``t_death`` becomes ``t_deaths = [t_death]``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

SCHEMA_VERSION = "0.4"
DEFAULT_CLASSES: tuple[str, ...] = (
    "single",
    "doublet",
    "debris",
    "fission_fusion",
)
_LAYER_COLORS = ("red", "yellow", "cyan", "magenta", "lime")
_UNSET = -1  # sentinel in napari properties for None-valued lifecycle fields


@dataclass
class Annotation:
    bbox: list[float]  # [y0, x0, y1, x1]
    label: str
    t_start: int = 0
    t_end: int | None = None
    t_deaths: list[int] = field(default_factory=list)
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


def _upgrade_annotation(a: dict, from_version: str) -> dict:
    """Migrate per-annotation dict from any older schema to v0.4."""
    # v0.1 had a single per-frame `t`; we no longer track it.
    a.pop("t", None)
    a.setdefault("t_start", 0)
    # v0.2 used `t_end` as the death marker. In v0.3+, t_end means "last
    # visible frame" and death has its own field.
    if from_version == "0.2":
        old_t_end = a.get("t_end")
        if old_t_end is not None and "t_death" not in a:
            a["t_death"] = old_t_end
            a["t_end"] = None
    # v0.3 had a single scalar t_death; v0.4 generalises to a list to
    # cover doublets (and future categories) where each cell dies
    # separately.
    if "t_deaths" not in a:
        single = a.pop("t_death", None)
        a["t_deaths"] = [single] if single is not None else []
    else:
        a.pop("t_death", None)
    a.setdefault("t_end", None)
    return a


def load(path: Path) -> AnnotationFile:
    payload = json.loads(path.read_text())
    raw_anns = payload.pop("annotations", [])
    from_version = payload.get("schema_version", "0.1")
    payload["schema_version"] = SCHEMA_VERSION
    af = AnnotationFile(**payload)
    af.annotations = [
        Annotation(**_upgrade_annotation(a, from_version)) for a in raw_anns
    ]
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


def _compute_label(
    t_start: int, t_deaths: list[int], current_t: int
) -> str:
    """Text shown above a bbox inside napari at the current T frame.

    The birth marker is always shown when ``t_start > 0`` — a static
    reminder of when the cell appeared. Death markers are added one per
    death that's already happened (``d <= current_t``), so scrubbing back
    before each death hides it.
    """
    parts = []
    if t_start > 0:
        parts.append(f"↑T={t_start}")
    past_deaths = sorted(d for d in t_deaths if d <= current_t)
    if past_deaths:
        parts.append("†T=" + ",".join(str(d) for d in past_deaths))
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
        def _encode_deaths(deaths: list[int]) -> str:
            return json.dumps(sorted(set(int(d) for d in deaths)))

        def _decode_deaths(s: str) -> list[int]:
            if not s:
                return []
            try:
                return [int(x) for x in json.loads(s)]
            except (ValueError, json.JSONDecodeError):
                return []

        class_layers: dict[str, "napari.layers.Shapes"] = {}
        for cls, base_color in class_colors.items():
            existing_shapes: list[list[list[float]]] = []
            existing_t_starts: list[int] = []
            existing_t_ends: list[int] = []
            existing_t_deaths: list[str] = []  # JSON-encoded lists
            for a in state.annotations:
                if a.label == cls:
                    existing_shapes.append(_bbox_to_shape(a.bbox))
                    existing_t_starts.append(a.t_start)
                    existing_t_ends.append(
                        a.t_end if a.t_end is not None else _UNSET
                    )
                    existing_t_deaths.append(_encode_deaths(a.t_deaths))
            props = {
                "t_start": np.array(existing_t_starts or [], dtype=int),
                "t_end": np.array(existing_t_ends or [], dtype=int),
                # t_deaths is a per-shape JSON-encoded list of ints — napari
                # properties are 1-D so we serialise the list to a string and
                # decode on read.
                "t_deaths_json": np.array(existing_t_deaths or [], dtype=object),
                # life_label is recomputed on every T change by _refresh_one_layer;
                # initial value is just a placeholder.
                "life_label": np.array(
                    [""] * len(existing_shapes), dtype=object
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
                    "t_end": np.array([_UNSET], dtype=int),
                    "t_deaths_json": np.array(["[]"], dtype=object),
                    "life_label": np.array([""], dtype=object),
                }
            except Exception:
                pass
            class_layers[cls] = layer

        def _refresh_one_layer(layer) -> None:
            """Hide shapes outside [t_start, t_end] and re-compute labels."""
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
                "t_end", np.full(n, _UNSET, dtype=int)
            )
            t_deaths_json = layer.properties.get(
                "t_deaths_json", np.array(["[]"] * n, dtype=object)
            )
            base = class_colors.get(layer.name, "red")
            colors = np.zeros((n, 4))
            labels = np.empty(n, dtype=object)
            for i in range(n):
                ts = int(t_starts[i]) if i < len(t_starts) else 0
                te = int(t_ends[i]) if i < len(t_ends) else _UNSET
                deaths = _decode_deaths(
                    t_deaths_json[i] if i < len(t_deaths_json) else "[]"
                )
                visible_end = te if te >= 0 else 10**9
                in_fov = ts <= current_t <= visible_end
                alpha = 1.0 if in_fov else 0.0
                colors[i] = to_rgba(base, alpha=alpha)
                labels[i] = (
                    _compute_label(ts, deaths, current_t) if in_fov else ""
                )
            layer.edge_color = colors
            new_props = dict(layer.properties)
            new_props["life_label"] = labels
            layer.properties = new_props

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
                t_deaths_json = layer.properties.get(
                    "t_deaths_json", np.array([], dtype=object)
                )
                for i, rect in enumerate(layer.data):
                    t_start = int(t_starts[i]) if i < len(t_starts) else 0
                    t_end_raw = int(t_ends[i]) if i < len(t_ends) else _UNSET
                    deaths = _decode_deaths(
                        t_deaths_json[i] if i < len(t_deaths_json) else "[]"
                    )
                    out.append(
                        Annotation(
                            bbox=_shape_to_bbox(rect),
                            label=cls,
                            t_start=t_start,
                            t_end=None if t_end_raw < 0 else t_end_raw,
                            t_deaths=deaths,
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
            """Write a scalar ``value`` into ``field`` at ``indices``.

            ``field`` is one of ``"t_start"`` or ``"t_end"``. Returns True if
            the write persisted. Per-shape death lists are handled separately
            by ``_modify_deaths``.
            """
            arr = layer.properties[field].copy()
            for i in indices:
                arr[i] = value
            new_props = dict(layer.properties)
            new_props[field] = arr
            layer.properties = new_props
            layer.refresh()
            _refresh_one_layer(layer)
            read_back = layer.properties[field]
            return all(int(read_back[i]) == value for i in indices)

        def _modify_deaths(layer, indices: list[int], op: str, t: int = 0) -> bool:
            """Edit the per-shape JSON-encoded death list.

            ``op`` is one of ``"add"`` (append ``t``), ``"pop"`` (drop the
            most recent), or ``"clear"`` (empty the list).
            """
            arr = layer.properties["t_deaths_json"].copy()
            for i in indices:
                current = _decode_deaths(arr[i])
                if op == "add":
                    current.append(int(t))
                elif op == "pop" and current:
                    current = sorted(current)[:-1]
                elif op == "clear":
                    current = []
                arr[i] = _encode_deaths(current)
            new_props = dict(layer.properties)
            new_props["t_deaths_json"] = arr
            layer.properties = new_props
            layer.refresh()
            _refresh_one_layer(layer)
            return True

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

        def _current_t() -> int:
            return (
                int(viewer.dims.current_step[t_dim_in_viewer])
                if t_dim_in_viewer is not None
                else 0
            )

        def _set_field(field: str, value: int, label: str) -> None:
            if t_dim_in_viewer is None and value >= 0:
                _report("this file has no time axis; nothing to mark")
                return
            layer, indices = _require_selection(label)
            if layer is None:
                return
            if not _apply_lifecycle(layer, indices, field, value):
                _report(
                    f"WARNING: {field} write did NOT persist on read-back."
                )
                return
            if value < 0:
                _report(
                    f"cleared {field} for {len(indices)} shape(s) "
                    f"in '{layer.name}'"
                )
            else:
                _report(
                    f"set {field}={value} for {len(indices)} shape(s) "
                    f"in '{layer.name}'"
                )

        from magicgui.widgets import Container, Label, PushButton

        def _btn(text: str, action) -> "PushButton":
            b = PushButton(text=text)
            b.clicked.connect(action)
            return b

        def _save_now() -> None:
            state.annotations = _collect()
            save(state, json_out)
            _report(
                f"saved {len(state.annotations)} annotations "
                f"({sum(1 for a in state.annotations if a.t_start > 0)} with birth, "
                f"{sum(1 for a in state.annotations if a.t_end is not None)} with end, "
                f"{sum(1 for a in state.annotations if a.t_deaths)} with death) "
                f"-> {json_out}"
            )

        def _death_op(op: str) -> None:
            if op == "add" and t_dim_in_viewer is None:
                _report("this file has no time axis; nothing to mark")
                return
            layer, indices = _require_selection(f"{op} death")
            if layer is None:
                return
            t = _current_t() if op == "add" else 0
            _modify_deaths(layer, indices, op, t)
            if op == "add":
                _report(
                    f"added death T={t} on {len(indices)} shape(s) in '{layer.name}'"
                )
            elif op == "pop":
                _report(
                    f"dropped last death on {len(indices)} shape(s) in '{layer.name}'"
                )
            else:
                _report(
                    f"cleared all deaths on {len(indices)} shape(s) in '{layer.name}'"
                )

        save_btn = _btn("Save annotations", _save_now)

        birth_mark = _btn(
            "Mark @ current T",
            lambda: _set_field("t_start", _current_t(), "mark birth"),
        )
        birth_clear = _btn(
            "Clear (reset to T=0)",
            lambda: _set_field("t_start", 0, "clear birth"),
        )

        end_mark = _btn(
            "Mark @ current T",
            lambda: _set_field("t_end", _current_t(), "mark end"),
        )
        end_clear = _btn(
            "Clear (visible until end)",
            lambda: _set_field("t_end", _UNSET, "clear end"),
        )

        death_add = _btn(
            "Add @ current T",
            lambda: _death_op("add"),
        )
        death_pop = _btn(
            "Drop last",
            lambda: _death_op("pop"),
        )
        death_clear = _btn(
            "Clear all",
            lambda: _death_op("clear"),
        )

        def _row(widgets):
            return Container(widgets=widgets, layout="horizontal", labels=False)

        panel = Container(
            widgets=[
                Label(value="↑ Birth (first visible frame)"),
                _row([birth_mark, birth_clear]),
                Label(value="→ End of visibility (cell leaves FOV)"),
                _row([end_mark, end_clear]),
                Label(
                    value="† Deaths (multi-cell categories — add one entry per dying cell)"
                ),
                _row([death_add, death_pop, death_clear]),
                save_btn,
            ],
            layout="vertical",
            labels=False,
        )
        viewer.window.add_dock_widget(panel, name="Lifecycle", area="right")

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
