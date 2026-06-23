"""ND2 annotation tool for cell-detection training data.

Opens an ND2 file in napari with one shape layer per class. The user draws
rectangles around cells in the appropriate class layer; bounding boxes are
saved to a JSON file next to the ND2.

CLI:

    nikon-control-annotate path/to/file.nd2
    nikon-control-annotate path/to/file.nd2 -o some/annotations.json
    nikon-control-annotate path/to/file.nd2 --classes single doublet dividing

Lifecycle and bbox model:

- ``t_start`` — first frame the cell/debris is visible. Defaults to the T at
  which the bbox was drawn (so a cell drawn at T=5 has ``t_start=5``).
- ``t_end``   — last frame visible (e.g. cell drifts out of FOV). Defaults
  to ``None`` = visible until end of recording.
- ``t_deaths`` — list of frames at which cells die. Single cells have at
  most one entry; doublets up to two; fission/fusion variable.
- ``keyframes`` — ordered list of ``{t, bbox}`` records. A cell that does
  not move has a single keyframe; drifting debris has several. At any T,
  the displayed bbox is linearly interpolated between the surrounding two
  keyframes. T values outside the keyframe range snap to the nearest one.

In the napari viewer:

- A bbox is **hidden** outside its visibility range ``[t_start, t_end]``.
- A ``↑T=N`` label sits above the bbox whenever it's visible and
  ``t_start > 0``.
- Death markers ``†T=N`` appear one per death that has already happened
  at the current frame.

Schema (stable, version 0.5):

    {
        "schema_version": "0.5",
        "source": "<path to nd2>",
        "image_shape": [...],
        "axes": ["T", "C", "Y", "X"],
        "channels": ["BF", "GFP", "DAPI"],
        "classes": ["single", "doublet", "debris", "fission_fusion"],
        "annotator": "",
        "annotations": [
            {
                "label": "single",
                "keyframes": [
                    {"t": 0,  "bbox": [10, 20, 100, 200]},
                    {"t": 20, "bbox": [30, 25, 120, 205]}
                ],
                "t_start": 0,
                "t_end": null,
                "t_deaths": [],
                "z": 0,
                "notes": "",
                "created": "2026-06-23T14:00:00"
            },
            ...
        ]
    }

Older files are auto-migrated on load:

- v0.1: the per-annotation ``t`` field is dropped.
- v0.2: ``t_end`` (then a death marker) moves into ``t_deaths``.
- v0.3: scalar ``t_death`` becomes ``t_deaths = [t_death]``.
- v0.4: ``bbox`` becomes a single keyframe at ``t_start``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

SCHEMA_VERSION = "0.5"
DEFAULT_CLASSES: tuple[str, ...] = (
    "single",
    "doublet",
    "debris",
    "fission_fusion",
)
_LAYER_COLORS = ("red", "yellow", "cyan", "magenta", "lime")
_UNSET = -1  # sentinel in napari properties for None-valued lifecycle fields


@dataclass
class Keyframe:
    t: int
    bbox: list[float]  # [y0, x0, y1, x1]


@dataclass
class Annotation:
    label: str
    keyframes: list[Keyframe] = field(default_factory=list)
    t_start: int = 0
    t_end: int | None = None
    t_deaths: list[int] = field(default_factory=list)
    z: int = 0
    notes: str = ""
    created: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )

    @property
    def bbox(self) -> list[float]:
        """First-keyframe bbox, for callers that want a single representative."""
        if not self.keyframes:
            return [0.0, 0.0, 0.0, 0.0]
        return self.keyframes[0].bbox


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
    payload = {
        "source": ann.source,
        "schema_version": ann.schema_version,
        "image_shape": list(ann.image_shape),
        "axes": list(ann.axes),
        "channels": list(ann.channels),
        "classes": list(ann.classes),
        "annotator": ann.annotator,
        "annotations": [
            {
                "label": a.label,
                "keyframes": [asdict(k) for k in a.keyframes],
                "t_start": a.t_start,
                "t_end": a.t_end,
                "t_deaths": list(a.t_deaths),
                "z": a.z,
                "notes": a.notes,
                "created": a.created,
            }
            for a in ann.annotations
        ],
    }
    path.write_text(json.dumps(payload, indent=2))


def _upgrade_annotation(a: dict, from_version: str) -> dict:
    """Migrate per-annotation dict from any older schema to v0.5."""
    # v0.1: drop the per-frame ``t``.
    a.pop("t", None)
    a.setdefault("t_start", 0)
    # v0.2: old ``t_end`` was the death marker.
    if from_version == "0.2":
        old_t_end = a.get("t_end")
        if old_t_end is not None and "t_death" not in a:
            a["t_death"] = old_t_end
            a["t_end"] = None
    # v0.3: scalar ``t_death`` → list ``t_deaths``.
    if "t_deaths" not in a:
        single = a.pop("t_death", None)
        a["t_deaths"] = [single] if single is not None else []
    else:
        a.pop("t_death", None)
    a.setdefault("t_end", None)
    # v0.4: ``bbox`` becomes a single keyframe at ``t_start``.
    if "keyframes" not in a:
        bbox = a.pop("bbox", None)
        if bbox is not None:
            a["keyframes"] = [{"t": a.get("t_start", 0), "bbox": list(bbox)}]
        else:
            a["keyframes"] = []
    else:
        a.pop("bbox", None)
    return a


def load(path: Path) -> AnnotationFile:
    payload = json.loads(path.read_text())
    raw_anns = payload.pop("annotations", [])
    from_version = payload.get("schema_version", "0.1")
    payload["schema_version"] = SCHEMA_VERSION
    af = AnnotationFile(**payload)
    annotations: list[Annotation] = []
    for raw in raw_anns:
        upgraded = _upgrade_annotation(raw, from_version)
        kfs = [Keyframe(**kf) for kf in upgraded.pop("keyframes", [])]
        annotations.append(Annotation(keyframes=kfs, **upgraded))
    af.annotations = annotations
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


def _interpolate_bbox(keyframes: list[Keyframe], t: int) -> list[float]:
    """Linear interpolation between surrounding keyframes; snap outside the range.

    Single keyframe → constant bbox at every T (matches the "static cell"
    case). Multiple keyframes → linear interpolation between the two flanking
    a given T; for T < first keyframe, snap to first; for T > last, snap to
    last.
    """
    if not keyframes:
        return [0.0, 0.0, 0.0, 0.0]
    if len(keyframes) == 1:
        return list(keyframes[0].bbox)
    sorted_kfs = sorted(keyframes, key=lambda k: k.t)
    if t <= sorted_kfs[0].t:
        return list(sorted_kfs[0].bbox)
    if t >= sorted_kfs[-1].t:
        return list(sorted_kfs[-1].bbox)
    for i in range(len(sorted_kfs) - 1):
        k0, k1 = sorted_kfs[i], sorted_kfs[i + 1]
        if k0.t <= t <= k1.t:
            if k1.t == k0.t:
                return list(k0.bbox)
            alpha = (t - k0.t) / (k1.t - k0.t)
            return [
                k0.bbox[j] + alpha * (k1.bbox[j] - k0.bbox[j]) for j in range(4)
            ]
    return list(sorted_kfs[-1].bbox)


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
        # Always merge in current DEFAULT_CLASSES — that way an existing
        # annotation file written by an older version automatically gains
        # any new built-in categories the next time it's opened. Append, so
        # existing class order (and layer colour assignment) is preserved.
        extras = list(DEFAULT_CLASSES)
        if classes is not None:
            extras.extend(classes)
        for c in extras:
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

        def _encode_keyframes(kfs: list[Keyframe]) -> str:
            return json.dumps(
                sorted(
                    [{"t": int(k.t), "bbox": [float(x) for x in k.bbox]} for k in kfs],
                    key=lambda d: d["t"],
                )
            )

        def _decode_keyframes(s: str) -> list[Keyframe]:
            if not s:
                return []
            try:
                raw = json.loads(s)
            except json.JSONDecodeError:
                return []
            return [Keyframe(t=int(d["t"]), bbox=[float(x) for x in d["bbox"]]) for d in raw]

        class_layers: dict[str, "napari.layers.Shapes"] = {}
        # Guard against reentrancy: _refresh_one_layer sets layer.data which
        # fires events.data, which would normally re-enter _on_data_change.
        refreshing: set[str] = set()

        for cls, base_color in class_colors.items():
            existing_shapes: list[list[list[float]]] = []
            existing_t_starts: list[int] = []
            existing_t_ends: list[int] = []
            existing_t_deaths: list[str] = []
            existing_keyframes: list[str] = []
            for a in state.annotations:
                if a.label == cls:
                    # Start at the first-keyframe bbox; _refresh_one_layer
                    # will interpolate on every T change.
                    existing_shapes.append(_bbox_to_shape(a.bbox))
                    existing_t_starts.append(a.t_start)
                    existing_t_ends.append(
                        a.t_end if a.t_end is not None else _UNSET
                    )
                    existing_t_deaths.append(_encode_deaths(a.t_deaths))
                    existing_keyframes.append(_encode_keyframes(a.keyframes))
            props = {
                "t_start": np.array(existing_t_starts or [], dtype=int),
                "t_end": np.array(existing_t_ends or [], dtype=int),
                "t_deaths_json": np.array(existing_t_deaths or [], dtype=object),
                "keyframes_json": np.array(existing_keyframes or [], dtype=object),
                # life_label recomputed by _refresh_one_layer on every T change.
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
                    # Empty keyframes_json signals to _on_data_change that this
                    # shape was freshly drawn and needs initialisation.
                    "keyframes_json": np.array([""], dtype=object),
                    "life_label": np.array([""], dtype=object),
                }
            except Exception:
                pass
            class_layers[cls] = layer

        def _current_t_or_zero() -> int:
            return (
                int(viewer.dims.current_step[t_dim_in_viewer])
                if t_dim_in_viewer is not None
                else 0
            )

        def _refresh_one_layer(layer) -> None:
            """Interpolate bboxes, hide outside [t_start, t_end], rebuild labels."""
            n = len(layer.data)
            if n == 0:
                return
            current_t = _current_t_or_zero()
            t_starts = layer.properties.get("t_start", np.zeros(n, dtype=int))
            t_ends = layer.properties.get(
                "t_end", np.full(n, _UNSET, dtype=int)
            )
            t_deaths_json = layer.properties.get(
                "t_deaths_json", np.array(["[]"] * n, dtype=object)
            )
            kfs_json = layer.properties.get(
                "keyframes_json", np.array([""] * n, dtype=object)
            )
            base = class_colors.get(layer.name, "red")
            colors = np.zeros((n, 4))
            labels = np.empty(n, dtype=object)
            new_data: list = []
            for i in range(n):
                ts = int(t_starts[i]) if i < len(t_starts) else 0
                te = int(t_ends[i]) if i < len(t_ends) else _UNSET
                deaths = _decode_deaths(
                    t_deaths_json[i] if i < len(t_deaths_json) else "[]"
                )
                kfs = _decode_keyframes(
                    kfs_json[i] if i < len(kfs_json) else ""
                )
                if kfs:
                    bbox = _interpolate_bbox(kfs, current_t)
                    new_data.append(np.array(_bbox_to_shape(bbox)))
                else:
                    # Freshly drawn shape — _on_data_change hasn't yet seeded
                    # its keyframes. Leave the existing vertices untouched.
                    new_data.append(layer.data[i])
                visible_end = te if te >= 0 else 10**9
                in_fov = ts <= current_t <= visible_end
                alpha = 1.0 if in_fov else 0.0
                colors[i] = to_rgba(base, alpha=alpha)
                labels[i] = (
                    _compute_label(ts, deaths, current_t) if in_fov else ""
                )
            refreshing.add(layer.name)
            try:
                # Only reassign data if any vertex changed; saves a redraw and
                # avoids losing the user's selection unnecessarily.
                if any(
                    not np.array_equal(np.asarray(new_data[i]), np.asarray(layer.data[i]))
                    for i in range(n)
                ):
                    selected = set(layer.selected_data)
                    layer.data = new_data
                    try:
                        layer.selected_data = selected
                    except Exception:
                        pass
                layer.edge_color = colors
                new_props = dict(layer.properties)
                new_props["life_label"] = labels
                layer.properties = new_props
            finally:
                refreshing.discard(layer.name)

        def _refresh_all_layers(event=None) -> None:
            for layer in class_layers.values():
                _refresh_one_layer(layer)

        def _on_data_change(event=None, layer=None) -> None:
            """Detect newly drawn shapes and seed their keyframes.

            A new shape arrives with the placeholder ``current_properties``
            (empty ``keyframes_json``). We snapshot the shape's bbox at the
            current T as its first keyframe and set ``t_start`` accordingly.
            """
            if layer.name in refreshing:
                return
            n = len(layer.data)
            if n == 0:
                return
            kfs_json = layer.properties.get(
                "keyframes_json", np.array([], dtype=object)
            )
            starts = layer.properties.get("t_start", np.array([], dtype=int))
            new_kfs = list(kfs_json)
            new_starts = list(int(s) for s in starts)
            # Properties may lag layer.data by one element when napari hasn't
            # yet auto-extended them from current_properties.
            while len(new_kfs) < n:
                new_kfs.append("")
            while len(new_starts) < n:
                new_starts.append(0)
            changed = False
            current_t = _current_t_or_zero()
            for i in range(n):
                if not new_kfs[i]:
                    bbox = _shape_to_bbox(layer.data[i])
                    new_kfs[i] = _encode_keyframes(
                        [Keyframe(t=current_t, bbox=bbox)]
                    )
                    new_starts[i] = current_t
                    changed = True
            if changed:
                new_props = dict(layer.properties)
                new_props["keyframes_json"] = np.array(new_kfs, dtype=object)
                new_props["t_start"] = np.array(new_starts, dtype=int)
                layer.properties = new_props
                _refresh_one_layer(layer)

        viewer.dims.events.current_step.connect(_refresh_all_layers)
        for _layer in class_layers.values():
            _layer.events.data.connect(
                lambda e=None, l=_layer: _on_data_change(e, l)
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
                kfs_json = layer.properties.get(
                    "keyframes_json", np.array([], dtype=object)
                )
                for i, rect in enumerate(layer.data):
                    t_start = int(t_starts[i]) if i < len(t_starts) else 0
                    t_end_raw = int(t_ends[i]) if i < len(t_ends) else _UNSET
                    deaths = _decode_deaths(
                        t_deaths_json[i] if i < len(t_deaths_json) else "[]"
                    )
                    kfs = _decode_keyframes(
                        kfs_json[i] if i < len(kfs_json) else ""
                    )
                    if not kfs:
                        # Shouldn't happen after _on_data_change fires, but be
                        # safe — fall back to the current drawn rectangle.
                        kfs = [
                            Keyframe(
                                t=t_start, bbox=_shape_to_bbox(rect)
                            )
                        ]
                    out.append(
                        Annotation(
                            label=cls,
                            keyframes=kfs,
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

        def _add_keyframe() -> None:
            """Snapshot the selected shape's current bbox as a keyframe at current T."""
            if t_dim_in_viewer is None:
                _report("this file has no time axis; keyframes have no meaning")
                return
            layer, indices = _require_selection("add keyframe")
            if layer is None:
                return
            current_t = _current_t()
            kfs_json = layer.properties["keyframes_json"].copy()
            for i in indices:
                kfs = _decode_keyframes(kfs_json[i])
                bbox = _shape_to_bbox(layer.data[i])
                # Remove any existing keyframe at this exact T to avoid dupes;
                # the new bbox overrides it.
                kfs = [kf for kf in kfs if kf.t != current_t]
                kfs.append(Keyframe(t=current_t, bbox=bbox))
                kfs_json[i] = _encode_keyframes(kfs)
            new_props = dict(layer.properties)
            new_props["keyframes_json"] = kfs_json
            layer.properties = new_props
            _refresh_one_layer(layer)
            _report(
                f"added keyframe @ T={current_t} for {len(indices)} shape(s) in '{layer.name}'"
            )

        def _del_keyframe() -> None:
            """Remove the keyframe at the current T from selected shape(s)."""
            if t_dim_in_viewer is None:
                _report("this file has no time axis")
                return
            layer, indices = _require_selection("delete keyframe")
            if layer is None:
                return
            current_t = _current_t()
            kfs_json = layer.properties["keyframes_json"].copy()
            removed_count = 0
            skipped_last = 0
            for i in indices:
                kfs = _decode_keyframes(kfs_json[i])
                remaining = [kf for kf in kfs if kf.t != current_t]
                if len(remaining) == len(kfs):
                    continue  # nothing at this T for this shape
                if not remaining:
                    skipped_last += 1
                    continue  # refuse to leave the shape with zero keyframes
                kfs_json[i] = _encode_keyframes(remaining)
                removed_count += 1
            new_props = dict(layer.properties)
            new_props["keyframes_json"] = kfs_json
            layer.properties = new_props
            _refresh_one_layer(layer)
            msg = f"removed keyframe @ T={current_t} on {removed_count} shape(s)"
            if skipped_last:
                msg += (
                    f" — refused on {skipped_last} shape(s) "
                    "(would leave zero keyframes)"
                )
            _report(msg)

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

        keyframe_add = _btn("Add @ current T", _add_keyframe)
        keyframe_del = _btn("Drop @ current T", _del_keyframe)

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
                Label(
                    value="⊞ ROI keyframes (for drifting cells/debris — leave alone if static)"
                ),
                _row([keyframe_add, keyframe_del]),
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
