"""Bokeh view for the SIMPLIFIED annotation dashboard.

A deliberately small dashboard for building detector training data. It
shares the viewer and the file browser with the full dashboard (see
``common.py``) but replaces the whole tracking/lifecycle panel with three
buttons.

The task it supports:

- annotate only the first N frames (default 20, settable) — the detector must
  recognise singles/doublets *early*;
- boxes are stored per frame (that is what training consumes), but the boxes
  of one cell share a light ``group`` identity so the annotator picks a class
  **once per cell** instead of once per frame — detections are grouped
  automatically by IoU;
- three classes only: **single**, **doublet**, **debris** (fresh detections
  arrive as *unlabeled* and the annotator assigns one);
- saved to ``<file>.simple.json`` — a different format and filename from the
  tracked dashboard's ``<file>.annotations.json``, so the two can't be mixed
  up.

All correctness logic lives in ``simple_state.SimpleState``; this file wires
widgets and renders.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..schema_simple import (
    DEFAULT_N_FRAMES,
    PROVISIONAL_LABEL,
    TRAINING_CLASSES,
    SimpleAnnotationFile,
    SimpleBox,
    boxes_from_annotations,
    load_simple,
    save_simple,
    simple_path_for,
)
from ..overlap import PREFER_AREA, PREFER_SCORE
from . import common
from .simple_state import SimpleState

# Short on-image tags so a crowded frame stays readable.
_SHORT = {"single": "S", "doublet": "D", "debris": "deb",
          PROVISIONAL_LABEL: "?"}


def modify_doc(doc, data_dir: str | Path = ".", weights_path: str = "") -> None:
    import threading

    from bokeh.layouts import column, row
    from bokeh.events import Tap
    from bokeh.models import (
        Button,
        Div,
        RadioButtonGroup,
        Range1d,
        RangeSlider,
        Select,
        Slider,
        Spinner,
        TextInput,
        Toggle,
    )

    from ..preannotate import detect_debris

    data_dir = common.resolve_data_dir(data_dir)
    weights_path = common.resolve_weights(weights_path)

    # ---- shared viewer -------------------------------------------------
    # box_edit=False: no drag-to-draw tool at all. Its gesture can get stuck
    # following the cursor when a mouseup is lost over RDP; ROIs are placed
    # with the buttons below instead, which cannot get stuck.
    fig, img_src, img_r, box_src, rect_r, mapper = common.build_image_figure(
        box_edit=False)

    # ---- file browser (same as the full dashboard) ---------------------
    drive_select = Select(title="Drive / volume", value="",
                          options=common.list_drives(), width=150)
    dir_input = TextInput(title="Folder", value=str(data_dir))
    up_btn = Button(label="⬆ Up", width=70)
    refresh_btn = Button(label="⟳ Refresh", width=90)
    subdir_select = Select(title="Subfolders (pick to open)", value="",
                           options=[])
    file_select = Select(title="ND2 file", value="", options=[])
    load_btn = Button(label="Load", button_type="primary")
    weights_select = Select(title="Model (.pth) in this folder", value="",
                            options=[])

    # ---- view ----------------------------------------------------------
    t_slider = Slider(start=0, end=1, value=0, step=1, title="T (frame)",
                      width=230)
    prev_btn = Button(label="◀ Prev", width=80)
    next_btn = Button(label="Next ▶", width=80)
    chan_select = Select(title="Channel", value="", options=[], width=340)
    contrast = RangeSlider(start=0, end=65535, value=(0, 65535), step=1,
                           title="Contrast", width=340)

    # ---- detection -----------------------------------------------------
    n_frames_spin = Spinner(title="Annotate first N frames", low=1, high=10000,
                            step=1, value=DEFAULT_N_FRAMES, width=170)
    weights_input = TextInput(title="Detection model (.pth)", value=weights_path)
    score_slider = Slider(start=0.1, end=0.95, value=0.5, step=0.05,
                          title="Detection score threshold")
    detect_btn = Button(label="Detect cells (first N)", button_type="warning")
    detect_debris_btn = Button(label="Detect debris (first N)",
                               button_type="warning")
    overlap_spin = Spinner(title="Max overlap (%)", low=10, high=100, step=5,
                           value=70, width=145)
    prefer_select = Select(title="On overlap keep the", value="bigger",
                           options=["bigger", "higher-scoring"], width=165)
    dedupe_btn = Button(label="⧈ Remove overlapping ROIs", width=300)

    # ---- annotation (the whole simplified panel) -----------------------
    add_roi_btn = Button(label="➕ Add ROI", button_type="primary", width=150)
    del_frame_btn = Button(label="🗑 This frame", button_type="danger",
                           width=145)
    del_cell_btn = Button(label="🗑 Whole cell", button_type="danger",
                          width=145)
    propagate_btn = Button(label="⤳ Copy to later frames", width=300)
    confirm_frame_btn = Button(label="✓ Accept predictions on this frame",
                               width=300)
    confirm_all_btn = Button(label="✓ Accept ALL predictions in this file",
                             width=300)
    place_toggle = Toggle(label="🎯 Click on image to place", width=300)
    nudge_spin = Spinner(title="Nudge step (px)", low=1, high=500, step=1,
                         value=10, width=145)
    nudge_up = Button(label="↑", width=45)
    nudge_down = Button(label="↓", width=45)
    nudge_left = Button(label="←", width=45)
    nudge_right = Button(label="→", width=45)
    _SCOPES = ["cell", "frame", "forward"]
    scope_radio = RadioButtonGroup(
        labels=["whole cell", "this frame", "from here on"], active=0,
        width=300)
    cls_btns = {c: Button(label=c, width=140) for c in TRAINING_CLASSES}
    width_spin = Spinner(title="ROI width (px)", low=4, high=4000, step=2,
                         value=110, width=145)
    height_spin = Spinner(title="ROI height (px)", low=4, high=4000, step=2,
                          value=110, width=145)
    shrink_btn = Button(label="− 10%", width=95)
    grow_btn = Button(label="＋ 10%", width=95)
    drop_far_btn = Button(label="🗑 Drop boxes beyond frame N", width=300)
    save_btn = Button(label="Save simplified annotations",
                      button_type="success", width=300)
    legend = Div(text="", styles={"font-size": "11px"}, width=420)
    progress_summary = Div(text="", styles={"font-size": "12px"}, width=420)
    status = Div(text="Pick an ND2 file and click Load.",
                 styles={"font-size": "12px"}, width=420)
    progress_div = Div(text="", styles={
        "font-size": "22px", "font-weight": "bold", "color": "#0a7",
        "padding": "8px 4px",
    })

    ctx: dict = {"state": None, "plane": None, "channels": [],
                 "syncing": False, "selected_ids": [], "json_mtime": None,
                 "data_dir": data_dir, "syncing_size": False,
                 "bf_index": 0, "detector": None}

    # ---- file browser wiring -------------------------------------------
    def _rescan(*_) -> None:
        d = Path(dir_input.value).expanduser()
        try:
            subs, nd2s, pths = common.scan_folder(d)
        except Exception as exc:
            status.text = f"⚠ cannot read folder '{d}': {exc}"
            return
        ctx["data_dir"] = d
        subdir_select.options = ["(open a subfolder…)"] + subs
        subdir_select.value = "(open a subfolder…)"
        file_select.options = nd2s
        file_select.value = nd2s[0] if nd2s else ""
        weights_select.options = ["(none)"] + pths
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

    drive_select.on_change("value", _on_drive)
    subdir_select.on_change("value", _on_subdir)
    up_btn.on_click(_on_up)
    refresh_btn.on_click(_rescan)
    weights_select.on_change("value", _on_weights_pick)

    # ---- rendering ------------------------------------------------------
    def _selected_ids() -> list[str]:
        st = ctx["state"]
        return [i for i in ctx["selected_ids"] if st is not None and st.has(i)]

    def _render_boxes() -> None:
        st = ctx["state"]
        if st is None:
            return
        rows = st.boxes_at()
        colors = common.class_color(st.classes)
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
                "color": [colors.get(r["label"], "#ffffff") for r in rows],
                # cell number + short class tag, so the annotator can see
                # the same cell keep its identity while scrubbing
                "text": [
                    f"{r['num']} " + _SHORT.get(r["label"], r["label"])
                    + (f" {r['marker']}" if r["marker"] else "")
                    for r in rows
                ],
            }
            sel = set(_selected_ids())
            box_src.selected.indices = [
                n for n, r in enumerate(rows) if r["id"] in sel
            ]
        finally:
            ctx["syncing"] = False
        _render_progress()

    def _render_progress() -> None:
        st = ctx["state"]
        if st is None:
            return
        cells = st.cell_counts()
        boxes = st.counts()
        here = st.counts_at()
        todo_cells = st.unlabeled_cells()
        # cells is the honest measure of independent data; boxes is what the
        # exporter emits, so show both
        parts = " · ".join(
            f"<b>{c}</b> {cells.get(c, 0)} cell(s)/{boxes.get(c, 0)} box"
            for c in TRAINING_CLASSES
        )
        far = st.out_of_range_count()
        far_txt = (f" <span style='color:#c60'>· {far} box(es) beyond frame "
                   f"{st.max_t}</span>") if far else ""
        unconf = st.unconfirmed_count()
        unconf_txt = (
            f"<br><span style='color:#c60'>{unconf} box(es) unverified "
            "(model predictions not yet accepted)</span>") if unconf else ""
        progress_summary.text = (
            f"Frames 0–{st.max_t} · this frame: {sum(here.values())} box(es)"
            f"<br>Labelled: {parts}"
            f"<br><span style='color:{'#c60' if todo_cells else '#0a7'}'>"
            f"{todo_cells} cell(s) still <i>unlabeled</i></span>"
            f"{unconf_txt}{far_txt}"
        )

    def _render_legend() -> None:
        st = ctx["state"]
        if st is None:
            return
        colors = common.class_color(st.classes)
        chips = "".join(
            f'<div style="margin:3px 0"><span style="color:{colors[c]};'
            f'font-size:16px">■</span> {c} '
            f'<span style="color:#888">({_SHORT.get(c, "")})</span></div>'
            for c in st.classes
        )
        legend.text = "<b>Classes</b>" + chips

    def _chan_index() -> int:
        chans = ctx["channels"]
        return chans.index(chan_select.value) if chan_select.value in chans else 0

    def _render_image() -> None:
        st = ctx["state"]
        if st is None:
            return
        img_src.data = {"image": [ctx["plane"](st.current_t, _chan_index())]}

    def _autoscale_contrast() -> None:
        st = ctx["state"]
        if st is None:
            return
        plane = ctx["plane"](st.current_t, _chan_index())
        mn, mx, lo, hi, step = common.contrast_bounds(plane)
        contrast.step = step
        contrast.start, contrast.end = mn, mx
        contrast.value = (lo, hi)
        mapper.low, mapper.high = lo, hi

    # ---- load ------------------------------------------------------------
    def do_load(event=None) -> None:
        name = file_select.value
        if not name:
            status.text = "No ND2 file selected."
            return
        path = ctx["data_dir"] / name
        prev = ctx.get("nd2_file")
        if prev is not None:
            try:
                prev.close()
            except Exception:
                pass
        nd = common.open_nd2(path)
        ctx["nd2_file"] = nd["file"]

        json_path = simple_path_for(path)
        if json_path.exists():
            try:
                af = load_simple(json_path)
            except ValueError as exc:  # pointed at a tracked file
                status.text = f"⚠ {exc}"
                return
        else:
            af = SimpleAnnotationFile(
                source=str(path), image_shape=list(nd["arr"].shape),
                axes=nd["axes"], channels=nd["channels"],
                bf_channel=nd["bf_index"],
                n_frames=min(int(n_frames_spin.value), nd["n_t"]),
            )

        st = SimpleState(af, n_t=nd["n_t"])
        ctx["state"] = st
        ctx["plane"] = nd["plane"]
        ctx["channels"] = nd["channels"]
        ctx["json_path"] = json_path
        ctx["json_mtime"] = (json_path.stat().st_mtime
                             if json_path.exists() else None)
        ctx["selected_ids"] = []
        ctx["bf_index"] = nd["bf_index"]

        H, W = nd["H"], nd["W"]
        fig.x_range = Range1d(0, W)
        fig.y_range = Range1d(H, 0)
        img_src.data = {"image": [np.zeros((H, W), dtype=np.float32)]}
        img_r.glyph.dw = W
        img_r.glyph.dh = H
        fig.title.text = f"{name}  (simplified / per-frame)"

        n_frames_spin.high = nd["n_t"]
        n_frames_spin.value = st.n_frames
        t_slider.start = 0
        t_slider.end = max(1, st.max_t)
        t_slider.value = 0
        chan_select.options = nd["channels"]
        chan_select.value = nd["channels"][nd["bf_index"]]

        st.set_t(0)
        _autoscale_contrast()
        _render_image()
        _render_boxes()
        _render_legend()
        status.text = (
            f"Loaded {name} — {nd['n_t']} frames, annotating the first "
            f"{st.n_frames}. {len(af.boxes)} box(es) in {json_path.name}."
        )

    load_btn.on_click(do_load)

    # ---- edit reconciliation --------------------------------------------
    def on_box_data_change(attr, old, new) -> None:
        if ctx["syncing"] or ctx["state"] is None:
            return
        st = ctx["state"]
        n = len(new.get("cx", []))
        rows = [{
            "id": (new.get("id") or [None] * n)[k],
            "cx": new["cx"][k], "cy": new["cy"][k],
            "w": new["w"][k], "h": new["h"][k],
            "label": (new.get("label") or [None] * n)[k],
        } for k in range(n)]
        st.apply_cds_edits(rows, PROVISIONAL_LABEL)
        _render_boxes()

    box_src.on_change("data", on_box_data_change)

    def _on_select(attr, old, new) -> None:
        ids = box_src.data.get("id", [])
        ctx["selected_ids"] = [ids[i] for i in new if i < len(ids)]
        st = ctx["state"]
        sel = _selected_ids()
        if sel and st is not None:
            w, h = st.size_of(sel[0])
            ctx["syncing_size"] = True
            try:
                width_spin.value = round(w)
                height_spin.value = round(h)
            finally:
                ctx["syncing_size"] = False

    box_src.selected.on_change("indices", _on_select)

    # ---- annotation actions ----------------------------------------------
    def _apply_to_selected(fn) -> None:
        st = ctx["state"]
        if st is None:
            return
        ids = _selected_ids()
        if not ids:
            status.text = "Select a box first (tap it), then use the buttons."
            return
        for i in ids:
            fn(st, i)
        _render_boxes()

    def _scope() -> str:
        return _SCOPES[scope_radio.active]

    def _set_class(label: str) -> None:
        st = ctx["state"]
        if st is None:
            return
        ids = _selected_ids()
        if not ids:
            status.text = "Select a box first (tap it), then pick a class."
            return
        scope = _scope()
        touched = sum(st.set_label(i, label, scope=scope, t=st.current_t)
                      for i in ids)
        _render_boxes()
        where = {"cell": "on every frame of the cell",
                 "frame": "on this frame only",
                 "forward": f"from frame {st.current_t} on"}[scope]
        status.text = f"Set → <b>{label}</b> {where} ({touched} box(es))."

    for cls, btn in cls_btns.items():
        btn.on_click(lambda cls=cls: _set_class(cls))

    def _add_roi() -> None:
        st = ctx["state"]
        if st is None:
            status.text = "Load an ND2 first."
            return
        img = img_src.data["image"][0]
        H, W = img.shape[-2], img.shape[-1]
        new_id = st.add_box(W / 2.0, H / 2.0, float(width_spin.value),
                            float(height_spin.value), PROVISIONAL_LABEL)
        ctx["selected_ids"] = [new_id]
        _render_boxes()
        status.text = ("Added an ROI at the centre. Use <b>🎯 Click on image "
                       "to place</b> or the arrows to position it, the "
                       "spinners to size it, then pick a class.")

    def _delete_roi(scope: str) -> None:
        st = ctx["state"]
        if st is None:
            return
        ids = _selected_ids()
        if not ids:
            status.text = "Select a box first (tap it), then Delete."
            return
        n = sum(st.delete(i, scope=scope) for i in ids if st.has(i))
        ctx["selected_ids"] = []
        _render_boxes()
        status.text = (f"Deleted {n} box(es)"
                       + (" — the whole cell." if scope == "cell"
                          else " on this frame."))

    def _propagate() -> None:
        st = ctx["state"]
        if st is None:
            return
        ids = _selected_ids()
        if not ids:
            status.text = "Select a box first (tap it), then copy it forward."
            return
        n = sum(st.propagate_forward(i) for i in ids)
        _render_boxes()
        status.text = (
            f"Copied to {n} later frame(s) as the same cell — one class "
            "change now updates all of them. Nudge individual frames if the "
            "cell drifts."
        )

    add_roi_btn.on_click(_add_roi)
    del_frame_btn.on_click(lambda: _delete_roi("frame"))
    del_cell_btn.on_click(lambda: _delete_roi("cell"))
    propagate_btn.on_click(_propagate)

    def _arm_place(active: bool) -> None:
        """Remember which box to move BEFORE the tap happens.

        Tapping empty space clears the tap-selection, so the live selection
        can't be used as the target — capture it when the button is armed.
        """
        st = ctx["state"]
        if not active:
            ctx["place_target"] = None
            return
        ids = _selected_ids()
        if st is None or not ids:
            place_toggle.active = False
            status.text = "Select a box first (tap it), then arm placing."
            return
        ctx["place_target"] = ids[0]
        status.text = ("Now click on the image where this ROI should go — "
                       "its size is kept. No dragging needed.")

    place_toggle.on_click(_arm_place)

    def _on_tap(event) -> None:
        """Move the armed box's centre to the clicked point (one-shot)."""
        if not place_toggle.active:
            return
        st = ctx["state"]
        target = ctx.get("place_target")
        if st is None or target is None or not st.has(target):
            place_toggle.active = False
            return
        st.move_to(target, float(event.x), float(event.y))
        ctx["selected_ids"] = [target]
        ctx["place_target"] = None
        place_toggle.active = False  # one-shot: no surprise moves later
        _render_boxes()
        status.text = (f"Moved ROI to ({event.x:.0f}, {event.y:.0f}). "
                       "Arm again to move another.")

    fig.on_event(Tap, _on_tap)

    def _nudge(dx: float, dy: float) -> None:
        step = float(nudge_spin.value or 10)
        _apply_to_selected(lambda st, i: st.nudge(i, dx * step, dy * step))

    nudge_up.on_click(lambda: _nudge(0, -1))     # y grows downward (flipped)
    nudge_down.on_click(lambda: _nudge(0, 1))
    nudge_left.on_click(lambda: _nudge(-1, 0))
    nudge_right.on_click(lambda: _nudge(1, 0))

    def _overlap_args() -> tuple[float, str]:
        thresh = float(overlap_spin.value or 70) / 100.0
        prefer = (PREFER_AREA if prefer_select.value == "bigger"
                  else PREFER_SCORE)
        return thresh, prefer

    def _dedupe(quiet: bool = False) -> int:
        """Drop duplicate overlapping boxes across every frame in range."""
        st = ctx["state"]
        if st is None:
            return 0
        thresh, prefer = _overlap_args()
        n = st.suppress_overlaps(thresh, prefer)
        if n:
            ctx["selected_ids"] = [i for i in ctx["selected_ids"]
                                   if st.has(i)]
            _render_boxes()
        if not quiet:
            pct = int(thresh * 100)
            status.text = (
                f"Removed {n} box(es) overlapping another by more than "
                f"{pct}% (kept the {prefer_select.value} one; boxes you "
                "made or classified are never removed for a detection)."
                if n else
                f"No box overlaps another by more than {pct}%."
            )
        return n

    dedupe_btn.on_click(lambda: _dedupe())

    def _confirm(all_frames: bool) -> None:
        st = ctx["state"]
        if st is None:
            return
        n = st.confirm_all() if all_frames else st.confirm_frame()
        _render_boxes()
        where = "the whole file" if all_frames else f"frame {st.current_t}"
        status.text = (f"Accepted {n} prediction(s) on {where} — they now "
                       "count as human-verified and survive a re-detect.")

    confirm_frame_btn.on_click(lambda: _confirm(False))
    confirm_all_btn.on_click(lambda: _confirm(True))

    def _on_size_change(attr, old, new) -> None:
        if ctx["syncing_size"]:
            return
        _apply_to_selected(
            lambda st, i: st.resize(i, float(width_spin.value),
                                    float(height_spin.value))
        )

    width_spin.on_change("value", _on_size_change)
    height_spin.on_change("value", _on_size_change)
    shrink_btn.on_click(lambda: _apply_to_selected(
        lambda st, i: st.scale(i, 0.9)))
    grow_btn.on_click(lambda: _apply_to_selected(
        lambda st, i: st.scale(i, 1.1)))

    # ---- frame navigation / range ----------------------------------------
    def on_t(attr, old, new) -> None:
        st = ctx["state"]
        if st is None:
            return
        st.set_t(int(new))
        _render_image()
        _render_boxes()

    t_slider.on_change("value", on_t)

    def _step_t(delta: int) -> None:
        st = ctx["state"]
        if st is None:
            return
        t_slider.value = max(0, min(st.max_t, st.current_t + delta))

    prev_btn.on_click(lambda: _step_t(-1))
    next_btn.on_click(lambda: _step_t(1))

    def on_n_frames(attr, old, new) -> None:
        st = ctx["state"]
        if st is None:
            return
        far = st.set_n_frames(int(new))
        t_slider.end = max(1, st.max_t)
        if st.current_t > st.max_t:
            t_slider.value = st.max_t
        _render_boxes()
        status.text = (
            f"Annotating frames 0–{st.max_t}."
            + (f" ⚠ {far} existing box(es) are beyond this range — they are "
               "KEPT (not shown). Use 'Drop boxes beyond frame N' to remove "
               "them." if far else "")
        )

    n_frames_spin.on_change("value", on_n_frames)

    def _drop_far() -> None:
        st = ctx["state"]
        if st is None:
            return
        n = st.drop_out_of_range()
        _render_boxes()
        status.text = f"Dropped {n} box(es) beyond frame {st.max_t}."

    drop_far_btn.on_click(_drop_far)

    def on_channel(attr, old, new) -> None:
        if ctx["state"] is None:
            return
        _autoscale_contrast()
        _render_image()

    chan_select.on_change("value", on_channel)

    def on_contrast(attr, old, new) -> None:
        mapper.low, mapper.high = float(new[0]), float(new[1])

    contrast.on_change("value", on_contrast)

    # ---- detection --------------------------------------------------------
    def _tick(msg: str) -> None:
        doc.add_next_tick_callback(lambda: setattr(progress_div, "text", msg))

    def do_detect(event=None) -> None:
        """Per-frame cell detection over the first N frames — NO tracking."""
        st = ctx["state"]
        if st is None:
            status.text = "Load an ND2 first."
            return
        wpath = weights_input.value.strip()
        if not wpath or not Path(wpath).exists():
            status.text = (f"⚠ Detection model not found: '{wpath}'. Set the "
                           "path to cell_detection_model.pth.")
            return
        detect_btn.disabled = True
        status.text = "Loading detection model…"
        n_frames = st.n_frames
        bf = ctx["bf_index"]

        def work() -> None:
            from ..detector import CellDetector

            def _make(device=None):
                d = CellDetector(wpath, device=device,
                                 score_threshold=score_slider.value)
                ctx["detector"] = d
                return d

            def _run(det):
                """Detect per frame, then link frame-to-frame by IoU purely to
                GROUP the boxes of one cell — so the annotator classifies each
                cell once instead of once per frame. Geometry stays exactly as
                detected on each frame (nothing is interpolated), which is what
                training consumes.

                With a MULTI-CLASS model the predicted class is used as the
                pre-selected label (still fully changeable); with the original
                single-class model every box arrives as ``unlabeled``.
                """
                import uuid as _uuid

                from ..tracking import IoUTracker

                per_frame = []
                # predicted class per (frame, detection), keyed by rounded
                # bbox so it can be recovered after tracking reorders things
                pred: dict[tuple, str] = {}
                for t in range(n_frames):
                    if t % 5 == 0 or t == n_frames - 1:
                        _tick(f"Detecting on {det.device.upper()}… "
                              f"frame {t + 1}/{n_frames}")
                    dets = det.detect_frame(ctx["plane"](t, bf))
                    per_frame.append(dets)
                    for d in dets:
                        if d.label:
                            pred[(t, *(round(v, 1) for v in d.bbox))] = d.label
                _tick("Linking detections into cells…")

                def _label_for(t, bb):
                    """Predicted class if the model gives one AND it is a
                    class we train on; otherwise leave it unlabeled for a
                    human to set."""
                    name = pred.get((t, *(round(float(v), 1) for v in bb)))
                    return name if name in TRAINING_CLASSES else PROVISIONAL_LABEL

                out: list[SimpleBox] = []
                for tr in IoUTracker(iou_threshold=0.3, max_age=2).track(
                        per_frame):
                    gid = _uuid.uuid4().hex
                    # one class per cell: majority vote over its frames, so a
                    # single odd frame doesn't split the cell's identity
                    votes: dict[str, int] = {}
                    for t, bb in zip(tr.frames, tr.bboxes):
                        lb = _label_for(t, bb)
                        if lb != PROVISIONAL_LABEL:
                            votes[lb] = votes.get(lb, 0) + 1
                    cell_label = (max(votes, key=votes.get) if votes
                                  else PROVISIONAL_LABEL)
                    for t, bb, sc in zip(tr.frames, tr.bboxes, tr.scores):
                        out.append(SimpleBox(
                            t=t, bbox=[float(v) for v in bb],
                            label=cell_label, score=float(sc),
                            auto=True, group=gid, origin="cells",
                        ))
                return out

            try:
                det = ctx.get("detector")
                if det is None:
                    det = _make()
                else:
                    det.score_threshold = score_slider.value
                try:
                    boxes = _run(det)
                except Exception as exc:
                    if "cuda" in str(exc).lower() and det.device != "cpu":
                        _tick("⚠ GPU error — retrying on CPU (slower)…")
                        det = _make(device="cpu")
                        boxes = _run(det)
                    else:
                        raise
                dev = det.device.upper()

                def finish():
                    n = st.set_detections(boxes, t_range=(0, n_frames),
                                           origin="cells")
                    # a multi-class model runs NMS per class, so the same
                    # object can arrive as both single and doublet — and the
                    # debris pass may already cover some of these
                    dropped = _dedupe(quiet=True)
                    ctx["selected_ids"] = []
                    _render_boxes()
                    detect_btn.disabled = False
                    progress_div.text = ""
                    ncells = len({b.group for b in boxes})
                    pre = len({b.group for b in boxes
                               if b.label != PROVISIONAL_LABEL})
                    dup = (f" {dropped} overlapping duplicate(s) removed."
                           if dropped else "")
                    if pre:
                        status.text = (
                            f"Detected {n} box(es) = <b>{ncells} cell(s)</b> "
                            f"across frames 0–{n_frames - 1} on {dev}.{dup} "
                            f"<b>{pre} cell(s) were pre-classified by the "
                            "model</b> — check them and fix any that are "
                            "wrong (tap + a class button), then <b>✓ Accept "
                            "predictions</b> to mark them reviewed. "
                            "Unaccepted predictions count as unverified."
                        )
                    else:
                        status.text = (
                            f"Detected {n} box(es) = <b>{ncells} cell(s)</b> "
                            f"across frames 0–{n_frames - 1} on {dev}.{dup} "
                            "All <i>unlabeled</i> (this model detects but does "
                            "not identify). Tap a box and pick a class — "
                            "scope <b>whole cell</b> labels every frame at "
                            "once. Already-classified cells were kept."
                        )
                doc.add_next_tick_callback(finish)
            except Exception as exc:
                msg = str(exc)
                low = msg.lower()
                hint = ""
                if ("dll" in low or "c10" in low or "winerror 1114" in low
                        or "initialization routine" in low):
                    hint = (" — torch failed to load. Check the env, install "
                            "the VC++ Redistributable, reinstall torch. See "
                            "docs/windows-deploy.md.")

                def fail(exc=exc, hint=hint):
                    detect_btn.disabled = False
                    progress_div.text = ""
                    status.text = f"⚠ Detection failed: {exc}{hint}"
                doc.add_next_tick_callback(fail)

        threading.Thread(target=work, daemon=True).start()

    detect_btn.on_click(do_detect)

    def do_detect_debris(event=None) -> None:
        """Debris detection over the first N frames, flattened per frame.

        Debris detection is inherently temporal (it needs a median
        background), so the tracked detector is reused and its tracks are
        then flattened into independent per-frame boxes.
        """
        st = ctx["state"]
        if st is None:
            status.text = "Load an ND2 first."
            return
        detect_debris_btn.disabled = True
        status.text = "Detecting debris…"
        n_frames = st.n_frames
        bf = ctx["bf_index"]

        def work() -> None:
            try:
                def prog(done, total):
                    if done % 5 == 0 or done == total:
                        _tick(f"Debris… frame {done}/{total}")

                anns = detect_debris(
                    ctx["plane"], bf, n_frames, t_range=(0, n_frames),
                    label="debris", progress_cb=prog,
                )
                boxes = boxes_from_annotations(anns, n_frames, label="debris")
                for b in boxes:
                    b.auto = True       # refreshable; human edits pin them
                    b.origin = "debris"  # so the cell pass won't wipe these

                def finish():
                    n = st.set_detections(boxes, t_range=(0, n_frames),
                                           origin="debris")
                    dropped = _dedupe(quiet=True)
                    ctx["selected_ids"] = []
                    _render_boxes()
                    detect_debris_btn.disabled = False
                    progress_div.text = ""
                    dup = (f" {dropped} overlapping duplicate(s) removed."
                           if dropped else "")
                    status.text = (
                        f"Found {n} debris box(es) across frames "
                        f"0–{n_frames - 1}.{dup} Review and delete false "
                        "positives; they are already labelled 'debris'."
                    )
                doc.add_next_tick_callback(finish)
            except Exception as exc:
                def fail(exc=exc):
                    detect_debris_btn.disabled = False
                    progress_div.text = ""
                    status.text = f"⚠ Debris detection failed: {exc}"
                doc.add_next_tick_callback(fail)

        threading.Thread(target=work, daemon=True).start()

    detect_debris_btn.on_click(do_detect_debris)

    # ---- save --------------------------------------------------------------
    def do_save(event=None) -> None:
        st = ctx["state"]
        if st is None:
            return
        json_path = Path(ctx["json_path"])
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
            save_simple(af, json_path)
        except Exception as exc:
            status.text = f"⚠ Save FAILED: {exc}"
            return
        try:
            ctx["json_mtime"] = json_path.stat().st_mtime
        except OSError:
            ctx["json_mtime"] = None
        n_train = len(af.training_boxes())
        todo = st.unlabeled_count()
        status.text = (
            f"Saved {len(af.boxes)} box(es) → {json_path.name} "
            f"({n_train} ready for training"
            + (f", {todo} still unlabeled)" if todo else ")")
        )

    save_btn.on_click(do_save)

    # ---- layout ------------------------------------------------------------
    _help = dict(styles={"font-size": "11px", "color": "#666"}, width=420)
    controls = column(
        Div(text="<b>Open a file</b>"),
        drive_select,
        dir_input,
        row(up_btn, refresh_btn),
        subdir_select,
        row(file_select, load_btn),
        Div(text="<b>Detection</b>"),
        n_frames_spin,
        weights_select,
        weights_input, score_slider,
        detect_btn,
        detect_debris_btn,
        row(overlap_spin, prefer_select),
        dedupe_btn,
        Div(text="<b>View</b>"),
        chan_select, contrast,
        row(prev_btn, next_btn),
        t_slider,
        width=360,
    )
    annotate_col = column(
        Div(text="<h3 style='margin:2px 0'>Simple annotation "
                 "<span style='font-size:12px;color:#888'>(per frame, no "
                 "tracking)</span></h3>", width=420),
        Div(text="Label cells and debris over the first N frames. Boxes are "
                 "stored <b>per frame</b> (that's what training needs), but "
                 "the boxes of one cell are linked, so you pick a class "
                 "<b>once per cell</b> — not once per frame. Saved to "
                 "<code>&lt;file&gt;.simple.json</code>.", **_help),
        legend,
        progress_summary,
        Div(text="<b>1 · Add or pick a box</b> — <b>tap</b> a box to select "
                 "it (turns white). The number on each box is its <b>cell "
                 "number</b>, kept as you scrub. <b>➕ Add ROI</b> puts a new "
                 "one on this frame.", **_help),
        row(add_roi_btn, del_frame_btn, del_cell_btn),
        Div(text="<b>Position it — no dragging.</b> Arm <b>🎯 Click on image "
                 "to place</b> then click where the ROI should go (one move "
                 "per arming), or nudge it with the arrows. Size comes from "
                 "the spinners below, so a box can never get stuck resizing "
                 "under the cursor.", **_help),
        place_toggle,
        row(nudge_spin, nudge_left, nudge_right, nudge_up, nudge_down),
        propagate_btn,
        Div(text="<b>2 · Set its class</b>", width=420),
        Div(text="Applies to — <b>whole cell</b>: every frame of this cell "
                 "(the normal choice: classify once). <b>this frame</b>: a "
                 "one-off fix. <b>from here on</b>: for a cell that changes "
                 "part-way, e.g. a single that divides into a doublet.",
            **_help),
        scope_radio,
        row(*(cls_btns[c] for c in TRAINING_CLASSES)),
        Div(text="If a trained multi-class model pre-selected the classes, "
                 "fix any that are wrong then <b>accept</b> the rest. "
                 "Unaccepted predictions are exported as <i>unverified</i> "
                 "(and <code>--verified-only</code> drops them), so the model "
                 "never silently becomes its own ground truth.", **_help),
        confirm_frame_btn,
        confirm_all_btn,
        Div(text="<i>Resize the selected box:</i>", **_help),
        row(width_spin, height_spin),
        row(shrink_btn, grow_btn),
        Div(text="<b>3 · Save</b>", width=420),
        save_btn,
        drop_far_btn,
        status,
        width=440,
    )
    doc.add_root(row(column(fig, progress_div), controls, annotate_col,
                     spacing=25))
    doc.title = "nikon-control — simple annotation"

    def _cleanup(session_context) -> None:
        fobj = ctx.get("nd2_file")
        if fobj is not None:
            try:
                fobj.close()
            except Exception:
                pass

    try:
        doc.on_session_destroyed(_cleanup)
    except Exception:
        pass

    _rescan()
    if file_select.value:
        do_load()
