"""Bokeh view for the annotation dashboard.

Thin layer: translate widget events into calls on ``DashboardState`` (the
tested controller) and render the model back into Bokeh glyphs. All
correctness-critical logic lives in ``state.py``; this file only wires
widgets and renders.

Run via the ``nikon-control-dashboard`` launcher (which invokes
``bokeh serve``). ``modify_doc(doc, data_dir)`` has no import-time side
effects so it can be constructed in a test with a bare ``Document``.
"""
from __future__ import annotations

import os
import string
from pathlib import Path

import numpy as np

from ..schema import DEFAULT_CLASSES, AnnotationFile, load, save
from .state import DashboardState

# distinct line colours per class (cycled)
_PALETTE = ["#ff3b30", "#ffcc00", "#34c759", "#00c7be", "#ff9500", "#af52de"]

# Fallback model locations tried when --weights isn't given and no .pth sits
# in the data folder. Add site-specific defaults here.
_DEFAULT_WEIGHTS = [
    r"E:\PROJECTS-01\Clement\cell_detection_model.pth",
]

# Fallback data folders the browser starts in when the launch folder has no
# ND2 files. Add site-specific defaults here.
_DEFAULT_DATA_DIRS = [
    r"G:\PROJECTS-02\Samuel",
]


def _list_drives() -> list[str]:
    """Available volumes to jump between: Windows drive letters, or the root
    and /Volumes mounts on macOS/Linux."""
    if os.name == "nt":
        return [f"{c}:\\" for c in string.ascii_uppercase
                if os.path.exists(f"{c}:\\")]
    vols = ["/"]
    v = Path("/Volumes")
    if v.exists():
        try:
            vols += [str(p) for p in sorted(v.iterdir()) if p.is_dir()]
        except Exception:
            pass
    return vols


def _class_color(classes: list[str]) -> dict[str, str]:
    return {c: _PALETTE[i % len(_PALETTE)] for i, c in enumerate(classes)}


def _plane_extractor(arr, axes: list[str]):
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


def modify_doc(doc, data_dir: str | Path = ".", weights_path: str = "") -> None:
    import threading

    from bokeh.layouts import column, row
    from bokeh.models import (
        BoxEditTool,
        Button,
        ColumnDataSource,
        Div,
        LabelSet,
        LinearColorMapper,
        Range1d,
        RangeSlider,
        Select,
        Slider,
        TextInput,
    )
    from bokeh.plotting import figure

    from ..preannotate import detect_and_track, detect_debris

    data_dir = Path(data_dir)
    # Start the browser at a site-default data folder when the launch folder
    # has no ND2s (e.g. a bare `--show` launch from the cwd).
    try:
        has_nd2 = any(data_dir.glob("*.nd2"))
    except Exception:
        has_nd2 = False
    if not has_nd2:
        for cand in _DEFAULT_DATA_DIRS:
            if Path(cand).is_dir():
                data_dir = Path(cand)
                break
    # Fall back to a site default model if none was provided at launch.
    if not weights_path:
        for cand in _DEFAULT_WEIGHTS:
            if Path(cand).exists():
                weights_path = cand
                break

    # ---- widgets (created empty; populated on load) --------------------
    # In-page file browser so users who can't use a terminal can navigate to
    # the ND2 and the model without --data-dir/--weights. Server-side listing
    # (the browser only shows the lists); safe for multi-user RDP.
    drive_select = Select(title="Drive / volume", value="",
                          options=_list_drives(), width=150)
    dir_input = TextInput(title="Folder", value=str(data_dir))
    up_btn = Button(label="⬆ Up", width=70)
    refresh_btn = Button(label="⟳ Refresh", width=90)
    subdir_select = Select(title="Subfolders (pick to open)", value="",
                           options=[])
    file_select = Select(title="ND2 file", value="", options=[])
    load_btn = Button(label="Load", button_type="primary")
    weights_select = Select(title="Model (.pth) in this folder", value="",
                            options=[])
    # widths kept within the 360px middle column so nothing bleeds into the
    # annotation column on the right
    t_slider = Slider(start=0, end=1, value=0, step=1, title="T (frame)",
                      width=230)
    prev_btn = Button(label="◀ Prev", width=80)
    play_btn = Button(label="▶ Play", width=90)
    next_btn = Button(label="Next ▶", width=80)
    speed_select = Select(title="Speed (fps)", value="5",
                          options=["2", "5", "10", "20"], width=110)
    chan_select = Select(title="Channel", value="", options=[], width=340)
    contrast = RangeSlider(start=0, end=65535, value=(0, 65535),
                           step=1, title="Contrast", width=340)
    label_select = Select(title="Category (applies to selected)",
                          value="", options=[], width=300)
    weights_input = TextInput(title="Detection model (.pth)",
                              value=weights_path)
    score_slider = Slider(start=0.1, end=0.95, value=0.5, step=0.05,
                          title="Detection score threshold")
    detect_btn = Button(label="Detect cells (refresh)", button_type="warning")
    detect_debris_btn = Button(label="Detect debris (refresh)", button_type="warning")
    legend = Div(text="", styles={"font-size": "11px"}, width=460)
    status = Div(text="Pick an ND2 file and click Load.",
                 styles={"font-size": "12px"}, width=460)
    # large progress banner shown UNDER the image during detection
    progress_div = Div(text="", styles={
        "font-size": "22px", "font-weight": "bold", "color": "#0a7",
        "padding": "8px 4px",
    })

    def _btn(label, kind="default"):
        return Button(label=label, button_type=kind, width=150)

    birth_mark, birth_clear = _btn("Mark birth @T"), _btn("Clear birth")
    end_mark, end_clear = _btn("Mark end @T"), _btn("Clear end")
    death_add, death_pop, death_clear = (
        _btn("Add death @T"), _btn("Drop last death"), _btn("Clear deaths")
    )
    kf_add, kf_drop = _btn("Add keyframe @T"), _btn("Drop keyframe @T")
    save_btn = _btn("Save annotations", "success")

    img_src = ColumnDataSource({"image": [np.zeros((2, 2), dtype=np.float32)]})
    box_src = ColumnDataSource(
        {"id": [], "num": [], "label": [], "cx": [], "cy": [], "w": [], "h": [],
         "marker": [], "color": [], "text": []}
    )
    mapper = LinearColorMapper(palette="Greys256", low=0, high=65535)
    fig = figure(width=760, height=760, match_aspect=True,
                 tools="pan,wheel_zoom,reset", title="(no file loaded)")
    img_r = fig.image(image="image", x=0, y=0, dw=1, dh=1, source=img_src,
                      color_mapper=mapper, level="image")
    rect_r = fig.rect(
        x="cx", y="cy", width="w", height="h", source=box_src,
        fill_alpha=0.0, line_color="color", line_width=3,
        # dim the others and make the selected box unmistakable
        nonselection_fill_alpha=0.0, nonselection_line_alpha=0.35,
        selection_fill_color="color", selection_fill_alpha=0.18,
        selection_line_color="white", selection_line_width=5,
    )
    fig.add_layout(
        LabelSet(
            x="cx", y="cy", text="text", source=box_src,
            text_color="white", text_font_size="10pt", text_font_style="bold",
            background_fill_color="black", background_fill_alpha=0.55,
            y_offset=12,
        )
    )
    box_tool = BoxEditTool(renderers=[rect_r], empty_value="")
    fig.add_tools(box_tool)
    fig.toolbar.active_drag = box_tool

    # ---- mutable session context --------------------------------------
    ctx: dict = {"state": None, "plane": None, "n_c": 1, "channels": [],
                 "syncing": False, "default_label": "cell",
                 "selected_ids": [], "json_mtime": None, "play_cb": None,
                 "frame_loading": False, "data_dir": data_dir}

    # ---- in-page directory browser ------------------------------------
    def _rescan(*_) -> None:
        d = Path(dir_input.value).expanduser()
        try:
            entries = sorted(d.iterdir(), key=lambda x: x.name.lower())
        except Exception as exc:
            status.text = f"⚠ cannot read folder '{d}': {exc}"
            return
        subs = [p.name for p in entries if p.is_dir() and not p.name.startswith(".")]
        nd2s = [p.name for p in entries if p.suffix.lower() == ".nd2"]
        pths = [p.name for p in entries if p.suffix.lower() == ".pth"]
        ctx["data_dir"] = d
        subdir_select.options = ["(open a subfolder…)"] + subs
        subdir_select.value = "(open a subfolder…)"
        file_select.options = nd2s
        file_select.value = nd2s[0] if nd2s else ""
        weights_select.options = ["(none)"] + pths
        # auto-fill the model path from this folder if not already valid
        if pths and (not weights_input.value
                     or not Path(weights_input.value).exists()):
            weights_input.value = str(d / pths[0])
            weights_select.value = pths[0]
        else:
            weights_select.value = "(none)"
        status.text = (f"{len(nd2s)} ND2, {len(pths)} model(s) in {d} — "
                       "pick an ND2 and click Load.")

    def _on_subdir(attr, old, new) -> None:
        if new and not new.startswith("("):
            dir_input.value = str(Path(dir_input.value).expanduser() / new)
            _rescan()

    def _on_up(*_) -> None:
        dir_input.value = str(Path(dir_input.value).expanduser().parent)
        _rescan()

    def _on_drive(attr, old, new) -> None:
        if new:
            dir_input.value = new
            _rescan()

    def _on_weights_pick(attr, old, new) -> None:
        if new and new != "(none)":
            weights_input.value = str(ctx["data_dir"] / new)

    def _on_select(attr, old, new) -> None:
        # Track selection by STABLE ID, not row index. box_src.data is
        # replaced on every re-render/scrub, and Bokeh keeps the integer
        # indices — so an index-based selection silently points at a
        # different annotation after the row set changes. Capture ids here.
        data = box_src.data
        ids = data.get("id", [])
        ctx["selected_ids"] = [ids[i] for i in new if i < len(ids)]
        # Textual confirmation of what's selected — but not during a
        # programmatic re-render (syncing), which would clobber other status.
        if not ctx["syncing"]:
            if new:
                i0 = new[0]
                num = data["num"][i0] if i0 < len(data.get("num", [])) else "?"
                lab = data["label"][i0] if i0 < len(data.get("label", [])) else "?"
                extra = f" (+{len(new) - 1} more)" if len(new) > 1 else ""
                status.text = f"Selected track #{num} — {lab}{extra}"
            else:
                status.text = "No box selected."

    box_src.selected.on_change("indices", _on_select)

    def _selected_ids() -> list[str]:
        st = ctx["state"]
        return [i for i in ctx.get("selected_ids", []) if st and st.has(i)]

    def _render_boxes() -> None:
        st = ctx["state"]
        if st is None:
            return
        rows = st.boxes_at(st.current_t)
        colors = _class_color(st.classes)
        def _text(r):
            s = f"#{r['num']} {r['label']}"
            return f"{s}  {r['marker']}" if r["marker"] else s

        ctx["syncing"] = True
        try:
            box_src.data = {
                "id": [r["id"] for r in rows],
                "num": [r["num"] for r in rows],
                "label": [r["label"] for r in rows],
                "cx": [r["cx"] for r in rows],
                "cy": [r["cy"] for r in rows],
                "w": [r["w"] for r in rows],
                "h": [r["h"] for r in rows],
                "marker": [r["marker"] for r in rows],
                "color": [colors.get(r["label"], "#ff3b30") for r in rows],
                "text": [_text(r) for r in rows],
            }
            # Restore selection by id so buttons keep acting on the same
            # annotation across re-renders and T-scrubs.
            sel = set(ctx.get("selected_ids", []))
            box_src.selected.indices = [
                k for k, rid in enumerate(box_src.data["id"]) if rid in sel
            ]
        finally:
            ctx["syncing"] = False
        _render_status()

    def _render_status() -> None:
        st = ctx["state"]
        if st is None:
            return
        counts = st.counts()
        parts = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none"
        status.text = (
            f"<b>{Path(st.af.source).name}</b> &nbsp; T={st.current_t}/"
            f"{st.n_t - 1} &nbsp;|&nbsp; tracks: {parts}"
        )

    def _chan_index() -> int:
        chans = ctx["channels"]
        return chans.index(chan_select.value) if chan_select.value in chans else 0

    def _render_image() -> None:
        st = ctx["state"]
        if st is None:
            return
        # Keep the native dtype (uint16) — half the websocket payload of a
        # float32 cast, and the LinearColorMapper handles the scaling.
        plane = ctx["plane"](st.current_t, _chan_index())
        img_src.data = {"image": [plane]}

    def _autoscale_contrast() -> None:
        st = ctx["state"]
        if st is None:
            return
        plane = ctx["plane"](st.current_t, _chan_index())
        mn, mx = float(plane.min()), float(plane.max())
        lo, hi = (float(v) for v in np.percentile(plane, [0.5, 99.5]))
        # Guard a flat/uniform frame (blank fluo channel, saturation): a
        # RangeSlider or color mapper with start==end breaks in the browser.
        if mx <= mn:
            mx = mn + 1.0
        if hi <= lo:
            hi = lo + 1.0
        # keep the (lo, hi) value inside [start, end] even for saturated
        # frames where a bumped hi could otherwise exceed mx
        mn, mx = min(mn, lo), max(mx, hi)
        contrast.start, contrast.end = mn, mx
        contrast.value = (lo, hi)
        mapper.low, mapper.high = lo, hi

    # ---- load ----------------------------------------------------------
    def do_load(event=None) -> None:
        import nd2

        name = file_select.value
        if not name:
            status.text = "No ND2 file selected."
            return
        _stop_play()  # stop any running playback before switching files
        path = ctx["data_dir"] / name
        # close a previously-open file when switching ND2s
        prev = ctx.get("nd2_file")
        if prev is not None:
            try:
                prev.close()
            except Exception:
                pass
        f = nd2.ND2File(str(path))
        ctx["nd2_file"] = f  # keep the handle alive for lazy dask reads
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
        H = sizes.get("Y", arr.shape[-2])
        W = sizes.get("X", arr.shape[-1])

        json_path = path.with_suffix(".annotations.json")
        if json_path.exists():
            af = load(json_path)
            for c in DEFAULT_CLASSES:
                if c not in af.classes:
                    af.classes.append(c)
        else:
            af = AnnotationFile(
                source=str(path), image_shape=list(arr.shape), axes=axes,
                channels=channels, classes=["cell", *DEFAULT_CLASSES],
            )

        ctx["state"] = DashboardState(af, n_t=n_t)
        ctx["plane"] = _plane_extractor(arr, axes)
        ctx["channels"] = channels
        ctx["json_path"] = json_path
        ctx["json_mtime"] = json_path.stat().st_mtime if json_path.exists() else None
        ctx["selected_ids"] = []
        ctx["default_label"] = af.classes[0] if af.classes else "cell"

        # configure figure extents (flip y so row 0 is at the top) and size
        # the image glyph to fill the frame.
        fig.x_range = Range1d(0, W)
        fig.y_range = Range1d(H, 0)
        img_src.data = {"image": [np.zeros((H, W), dtype=np.float32)]}
        img_r.glyph.dw = W
        img_r.glyph.dh = H
        fig.title.text = name

        t_slider.start = 0
        t_slider.end = max(1, n_t - 1)
        t_slider.value = 0
        chan_select.options = channels
        bf_idx = next((i for i, c in enumerate(channels) if "bf" in c.lower()), 0)
        chan_select.value = channels[bf_idx]
        ctx["bf_index"] = bf_idx
        label_select.options = list(af.classes)
        label_select.value = ctx["default_label"]

        ctx["state"].set_t(0)
        _autoscale_contrast()
        _render_image()
        _render_boxes()
        _render_legend()

    def _render_legend() -> None:
        st = ctx["state"]
        if st is None:
            return
        colors = _class_color(st.classes)
        chips = "".join(
            f'<div style="margin:3px 0"><span style="color:{colors[c]};'
            f'font-size:16px">■</span> {c}</div>'
            for c in st.classes
        )
        legend.text = "<b>Classes</b>" + chips

    # ---- edit reconciliation (BoxEditTool -> controller) ---------------
    def on_box_data_change(attr, old, new) -> None:
        if ctx["syncing"] or ctx["state"] is None:
            return
        d = box_src.data
        n = len(d.get("cx", []))
        rows = [
            {
                "id": d["id"][i] if i < len(d.get("id", [])) else "",
                "label": d["label"][i] if i < len(d.get("label", [])) else "",
                "cx": d["cx"][i], "cy": d["cy"][i],
                "w": d["w"][i], "h": d["h"][i],
            }
            for i in range(n)
        ]
        ctx["state"].apply_cds_edits(rows, ctx["default_label"],
                                     ctx["state"].current_t)
        _render_boxes()

    box_src.on_change("data", on_box_data_change)

    # ---- widget callbacks ---------------------------------------------
    def on_t(attr, old, new) -> None:
        # cheap: update model time + boxes on every tick
        if ctx["state"] is None:
            return
        ctx["state"].set_t(int(new))
        _render_boxes()

    def on_t_image(attr, old, new) -> None:
        # heavy: reload+push the frame only when the slider settles
        if ctx["state"] is None:
            return
        ctx["state"].set_t(int(new))
        _render_image()

    def on_channel(attr, old, new) -> None:
        if ctx["state"] is None:
            return
        _autoscale_contrast()
        _render_image()

    def on_contrast(attr, old, new) -> None:
        mapper.low, mapper.high = float(new[0]), float(new[1])

    def _apply_to_selected(fn, action: str = "") -> None:
        st = ctx["state"]
        if st is None:
            return
        ids = _selected_ids()
        if not ids:
            status.text = "Select a box first (use the tap tool, then a button)."
            return
        refused = sum(1 for i in ids if fn(st, i) is False)
        _render_boxes()
        if refused and action:
            status.text = (
                f"⚠ {action} refused on {refused} box(es): birth must be "
                "≤ end (would make the box invisible)."
            )

    def on_label(attr, old, new) -> None:
        if new:
            _apply_to_selected(lambda st, i: st.set_label(i, new))
            _render_legend()  # a new class may have been introduced

    def _step_t(delta: int) -> None:
        if ctx["state"] is None:
            return
        new_t = max(0, min(ctx["state"].n_t - 1, ctx["state"].current_t + delta))
        # setting the slider fires on_t (boxes); step also refreshes the image
        # since 'value_throttled' doesn't fire on programmatic .value changes
        t_slider.value = new_t
        ctx["state"].set_t(new_t)
        _render_image()

    def _advance_frame() -> None:
        # Playback tick. The frame read (dask/ND2 disk decode) is done in a
        # worker thread so it never blocks the shared Bokeh IOLoop (which
        # would stall other annotators' sessions on the RDP server). An
        # in-flight guard drops ticks while a read is pending, so frames are
        # dropped rather than queued unboundedly at high fps.
        st = ctx["state"]
        if st is None or ctx.get("frame_loading"):
            return
        new_t = st.current_t + 1
        if new_t >= st.n_t:
            new_t = 0  # loop
        ctx["frame_loading"] = True
        c = _chan_index()

        def work(tt=new_t, cc=c):
            try:
                plane = ctx["plane"](tt, cc)
            except Exception:
                plane = None

            def apply():
                ctx["frame_loading"] = False
                if plane is None:
                    return
                st.set_t(tt)
                t_slider.value = tt      # fires on_t -> boxes (cheap)
                img_src.data = {"image": [plane]}
            doc.add_next_tick_callback(apply)

        threading.Thread(target=work, daemon=True).start()

    def _stop_play() -> None:
        cb = ctx.get("play_cb")
        if cb is not None:
            try:
                doc.remove_periodic_callback(cb)
            except Exception:
                pass
            ctx["play_cb"] = None
        ctx["frame_loading"] = False
        play_btn.label = "▶ Play"

    def toggle_play() -> None:
        if ctx["state"] is None:
            return
        if ctx.get("play_cb") is not None:
            _stop_play()
            return
        fps = float(speed_select.value or "5")
        period_ms = max(30, int(1000.0 / fps))
        ctx["play_cb"] = doc.add_periodic_callback(_advance_frame, period_ms)
        play_btn.label = "⏸ Pause"

    def on_speed(attr, old, new) -> None:
        if ctx.get("play_cb") is not None:  # restart at the new rate
            _stop_play()
            toggle_play()

    def do_detect(event=None) -> None:
        st = ctx["state"]
        if st is None:
            status.text = "Load an ND2 first."
            return
        wpath = weights_input.value.strip()
        if not wpath or not Path(wpath).exists():
            status.text = (
                f"⚠ Detection model not found: '{wpath}'. Set the path to "
                "cell_detection_model.pth."
            )
            return
        detect_btn.disabled = True
        status.text = "Loading detection model…"

        def _tick(msg: str) -> None:
            doc.add_next_tick_callback(lambda: setattr(progress_div, "text", msg))

        def work() -> None:
            try:
                det = ctx.get("detector")
                if det is None:
                    from ..detector import CellDetector
                    det = CellDetector(wpath, score_threshold=score_slider.value)
                    ctx["detector"] = det
                else:
                    det.score_threshold = score_slider.value
                dev = det.device.upper()  # CUDA or CPU — surfaced to the user

                def prog(done, total):
                    if done % 10 == 0 or done == total:
                        _tick(f"Detecting on {dev}… frame {done}/{total}")

                anns = detect_and_track(
                    det, ctx["plane"], ctx["bf_index"], st.n_t,
                    provisional_label=ctx["default_label"], progress_cb=prog,
                )

                def finish():
                    n = st.set_detections(anns, ctx["default_label"])
                    ctx["selected_ids"] = []
                    _render_boxes()
                    detect_btn.disabled = False
                    progress_div.text = ""
                    cpu_hint = (" — running on CPU (slow); see docs to install "
                                "the CUDA build of torch for GPU") if dev == "CPU" else ""
                    status.text = (
                        f"Detected {n} '{ctx['default_label']}' track(s) on "
                        f"{dev}. Curated tracks kept. Review & save.{cpu_hint}"
                    )
                doc.add_next_tick_callback(finish)
            except Exception as exc:
                def fail(exc=exc):
                    detect_btn.disabled = False
                    progress_div.text = ""
                    status.text = f"⚠ Detection failed: {exc}"
                doc.add_next_tick_callback(fail)

        threading.Thread(target=work, daemon=True).start()

    def do_detect_debris(event=None) -> None:
        st = ctx["state"]
        if st is None:
            status.text = "Load an ND2 first."
            return
        detect_debris_btn.disabled = True
        status.text = "Detecting debris (motion-based)…"

        def _tick(msg):
            doc.add_next_tick_callback(lambda: setattr(progress_div, "text", msg))

        # Snapshot the current cell boxes so debris whose centre falls inside
        # a cell (the cell's own texture flicker, or a doublet's lobes /
        # mitotic daughters) is dropped. Read now, before the worker starts.
        from ..detector import bbox_center, point_in_bbox
        cell_anns = [a for a in st.annotations() if a.label != "debris"]

        def exclude(t, bbox):
            cy, cx = bbox_center(bbox)
            for a in cell_anns:
                # Use the cell's box regardless of its annotated [t_start,t_end]
                # window: a cell physically occupies that spot even at frames
                # outside its lifecycle (bbox_at clamps to the nearest
                # keyframe). Generous margin covers a splitting doublet.
                if point_in_bbox(cy, cx, a.bbox_at(t), margin=20):
                    return True
            return False

        def work():
            try:
                def prog(done, total):
                    if done % 10 == 0 or done == total:
                        _tick(f"Detecting debris… frame {done}/{total}")

                anns = detect_debris(
                    ctx["plane"], ctx["bf_index"], st.n_t,
                    # large gate (debris moves fast); skip big blobs (cells)
                    # and small specks; pad ROIs so they enclose the debris;
                    # drop debris sitting inside a cell.
                    max_dist=500.0, min_len=3, min_area=150, max_area=1500,
                    bbox_pad=25, exclude_fn=exclude, progress_cb=prog,
                )

                def finish():
                    if "debris" not in st.classes:
                        st.classes.append("debris")
                    n = st.set_detections(anns, "debris")
                    ctx["selected_ids"] = []
                    _render_boxes()
                    _render_legend()
                    detect_debris_btn.disabled = False
                    progress_div.text = ""
                    warn = ""
                    if not cell_anns:
                        warn = (
                            " ⚠ No cell ROIs present, so debris ON cells was "
                            "NOT excluded — run 'Detect cells' first for best "
                            "results."
                        )
                    status.text = (
                        f"Detected {n} debris track(s). Review & save. "
                        f"(fixed bubbles may appear — delete if unwanted).{warn}"
                    )
                doc.add_next_tick_callback(finish)
            except Exception as exc:
                def fail(exc=exc):
                    detect_debris_btn.disabled = False
                    progress_div.text = ""
                    status.text = f"⚠ Debris detection failed: {exc}"
                doc.add_next_tick_callback(fail)

        threading.Thread(target=work, daemon=True).start()

    load_btn.on_click(do_load)
    dir_input.on_change("value", lambda a, o, n: None)  # typing; Refresh applies
    refresh_btn.on_click(_rescan)
    up_btn.on_click(_on_up)
    drive_select.on_change("value", _on_drive)
    subdir_select.on_change("value", _on_subdir)
    weights_select.on_change("value", _on_weights_pick)
    detect_btn.on_click(do_detect)
    detect_debris_btn.on_click(do_detect_debris)
    prev_btn.on_click(lambda: _step_t(-1))
    next_btn.on_click(lambda: _step_t(+1))
    play_btn.on_click(toggle_play)
    speed_select.on_change("value", on_speed)
    # 'value' fires on every drag tick (cheap: move time + boxes); the heavy
    # image reload+push is bound to 'value_throttled' so scrubbing doesn't
    # queue many multi-MB frames over the websocket.
    t_slider.on_change("value", on_t)
    t_slider.on_change("value_throttled", on_t_image)
    chan_select.on_change("value", on_channel)
    contrast.on_change("value", on_contrast)
    label_select.on_change("value", on_label)
    birth_mark.on_click(lambda: _apply_to_selected(lambda st, i: st.mark_birth(i), "Mark birth"))
    birth_clear.on_click(lambda: _apply_to_selected(lambda st, i: st.clear_birth(i)))
    end_mark.on_click(lambda: _apply_to_selected(lambda st, i: st.mark_end(i), "Mark end"))
    end_clear.on_click(lambda: _apply_to_selected(lambda st, i: st.clear_end(i)))
    death_add.on_click(lambda: _apply_to_selected(lambda st, i: st.add_death(i)))
    death_pop.on_click(lambda: _apply_to_selected(lambda st, i: st.pop_death(i)))
    death_clear.on_click(lambda: _apply_to_selected(lambda st, i: st.clear_deaths(i)))
    kf_add.on_click(lambda: _apply_to_selected(lambda st, i: st.add_keyframe(i)))
    kf_drop.on_click(lambda: _apply_to_selected(lambda st, i: st.drop_keyframe(i)))

    def do_save(event=None) -> None:
        st = ctx["state"]
        if st is None:
            return
        json_path = Path(ctx["json_path"])
        # Concurrent-edit guard: if the file exists on disk but its mtime
        # differs from what we loaded — OR it appeared when we had none —
        # another session wrote it. Refuse rather than clobber their work.
        if json_path.exists():
            disk_mtime = json_path.stat().st_mtime
            if ctx.get("json_mtime") is None or disk_mtime != ctx["json_mtime"]:
                status.text = (
                    "⚠ NOT saved: this file changed on disk (another user?). "
                    "Reload to merge, then save again — refusing to overwrite."
                )
                return
        af = st.sync_to_file()
        try:
            save(af, json_path)
        except Exception as exc:  # read-only dir, disconnected share, ...
            status.text = f"⚠ Save FAILED: {exc}"
            return
        try:
            ctx["json_mtime"] = json_path.stat().st_mtime
        except OSError:
            ctx["json_mtime"] = None
        n_dead = sum(1 for a in af.annotations if a.t_deaths)
        status.text = (
            f"Saved {len(af.annotations)} annotations "
            f"({n_dead} with death) → {json_path.name}"
        )

    save_btn.on_click(do_save)

    # ---- layout --------------------------------------------------------
    # Middle column: file / detection / view. Right column: everything to do
    # with annotating (category, lifecycle, save) + the class legend.
    controls = column(
        Div(text="<b>Open a file</b>"),
        drive_select,
        dir_input,
        row(up_btn, refresh_btn),
        subdir_select,
        row(file_select, load_btn),
        Div(text="<b>Detection</b>"),
        weights_select,
        weights_input, score_slider,
        row(detect_btn, detect_debris_btn),
        Div(text="<b>View</b>"),
        chan_select, contrast,
        row(prev_btn, play_btn, next_btn),
        row(t_slider, speed_select),
        width=360,
    )
    _help = dict(styles={"font-size": "11px", "color": "#666"}, width=460)
    annotate_col = column(
        Div(text="<h3 style='margin:2px 0'>Annotation</h3>", width=460),
        legend,
        Div(text="<b>Editing ROIs</b> — pick the <i>Box Edit</i> tool (top-"
                 "right toolbar). <b>Add</b>: click-drag on empty area. "
                 "<b>Move</b>: drag a box. <b>Delete</b>: tap to select then "
                 "press Backspace (or Shift-click). <b>Resize</b>: delete and "
                 "redraw. A new box gets the default category — set it below.",
            **_help),
        Div(text="<b>Selected ROI</b> — tap a box first (it turns white)",
            width=460),
        label_select,
        Div(text="<i>For most cells: just set the category, then mark death. "
                 "Birth / End / Keyframe below are only for the special cases "
                 "described.</i>", **_help),
        Div(text="† <b>Death</b> — frame a cell dies (it stays visible as a "
                 "corpse). A doublet can have two deaths.", **_help),
        row(death_add, death_pop, death_clear),
        Div(text="↑ <b>Birth</b> — first frame the cell exists. Only if it "
                 "appears mid-movie (enters, or born from a division). "
                 "Default: from frame 0.", **_help),
        row(birth_mark, birth_clear),
        Div(text="→ <b>End</b> — last frame visible. Only if the cell leaves "
                 "the FOV (not for death). Default: to the last frame.", **_help),
        row(end_mark, end_clear),
        Div(text="⊞ <b>Keyframe</b> — only for a moving ROI: drag the box to "
                 "follow the object, then Add, and it interpolates between. "
                 "Static cells never need this.", **_help),
        row(kf_add, kf_drop),
        save_btn,
        status,
        width=480,
    )
    doc.add_root(row(column(fig, progress_div), controls, annotate_col,
                     spacing=25))
    doc.title = "nikon-control — cell annotation"

    def _cleanup(session_context) -> None:
        _stop_play()  # cancel the playback periodic callback
        fobj = ctx.get("nd2_file")
        if fobj is not None:
            try:
                fobj.close()
            except Exception:
                pass

    try:
        doc.on_session_destroyed(_cleanup)  # close ND2 when the tab disconnects
    except Exception:
        pass  # bare Document (tests) has no session lifecycle

    _rescan()  # populate the browser from the launch folder
    if file_select.value:
        do_load()  # auto-load the first ND2 if the launch folder has one
