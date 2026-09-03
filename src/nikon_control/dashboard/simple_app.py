"""Bokeh view for the SIMPLIFIED annotation dashboard.

A deliberately small dashboard for building detector training data. It
shares the viewer and the file browser with the full dashboard (see
``common.py``) but replaces the whole tracking/lifecycle panel with three
buttons.

The task it supports:

- annotate only the first N frames (default 20, settable) — the detector must
  recognise singles/doublets *early*;
- every ROI is independent: a cell on frame 0 and the same cell on frame 5
  are two separate labels, and nothing is tracked;
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
from . import common
from .simple_state import SimpleState

# Short on-image tags so a crowded frame stays readable.
_SHORT = {"single": "S", "doublet": "D", "debris": "deb",
          PROVISIONAL_LABEL: "?"}


def modify_doc(doc, data_dir: str | Path = ".", weights_path: str = "") -> None:
    import threading

    from bokeh.layouts import column, row
    from bokeh.models import (
        Button,
        Div,
        Range1d,
        RangeSlider,
        Select,
        Slider,
        Spinner,
        TextInput,
    )

    from ..preannotate import detect_debris

    data_dir = common.resolve_data_dir(data_dir)
    weights_path = common.resolve_weights(weights_path)

    # ---- shared viewer -------------------------------------------------
    fig, img_src, img_r, box_src, rect_r, mapper = common.build_image_figure()

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

    # ---- annotation (the whole simplified panel) -----------------------
    add_roi_btn = Button(label="➕ Add ROI", button_type="primary", width=150)
    del_roi_btn = Button(label="🗑 Delete ROI", button_type="danger", width=150)
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
                "text": [
                    _SHORT.get(r["label"], r["label"])
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
        counts = st.counts()
        here = st.counts_at()
        todo = st.unlabeled_count()
        parts = " · ".join(
            f"<b>{c}</b> {counts.get(c, 0)}" for c in TRAINING_CLASSES
        )
        far = st.out_of_range_count()
        far_txt = (f" <span style='color:#c60'>· {far} box(es) beyond frame "
                   f"{st.max_t}</span>") if far else ""
        progress_summary.text = (
            f"Frames 0–{st.max_t} · this frame: {sum(here.values())} box(es)"
            f"<br>Total labelled: {parts}"
            f"<br><span style='color:{'#c60' if todo else '#0a7'}'>"
            f"{todo} still <i>unlabeled</i></span>{far_txt}"
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

    def _set_class(label: str) -> None:
        _apply_to_selected(lambda st, i: st.set_label(i, label))
        n = len(_selected_ids())
        if n:
            status.text = f"Set {n} box(es) → {label}."

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
        status.text = ("Added an ROI on this frame — now pick single / "
                       "doublet / debris.")

    def _delete_roi() -> None:
        st = ctx["state"]
        if st is None:
            return
        ids = _selected_ids()
        if not ids:
            status.text = "Select a box first (tap it), then Delete."
            return
        for i in ids:
            st.delete(i)
        ctx["selected_ids"] = []
        _render_boxes()
        status.text = f"Deleted {len(ids)} ROI(s)."

    add_roi_btn.on_click(_add_roi)
    del_roi_btn.on_click(_delete_roi)

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
                out: list[SimpleBox] = []
                for t in range(n_frames):
                    if t % 5 == 0 or t == n_frames - 1:
                        _tick(f"Detecting on {det.device.upper()}… "
                              f"frame {t + 1}/{n_frames}")
                    for d in det.detect_frame(ctx["plane"](t, bf)):
                        out.append(SimpleBox(
                            t=t, bbox=[float(v) for v in d.bbox],
                            label=PROVISIONAL_LABEL, score=d.score,
                            auto=True,
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
                    n = st.set_detections(boxes, t_range=(0, n_frames))
                    ctx["selected_ids"] = []
                    _render_boxes()
                    detect_btn.disabled = False
                    progress_div.text = ""
                    status.text = (
                        f"Detected {n} box(es) across frames 0–{n_frames - 1} "
                        f"on {dev} — all <i>unlabeled</i>. Tap each and set "
                        "single / doublet / debris. Already-classified boxes "
                        "were kept."
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
                    b.auto = True  # refreshable; human edits pin them

                def finish():
                    n = st.set_detections(boxes, t_range=(0, n_frames))
                    ctx["selected_ids"] = []
                    _render_boxes()
                    detect_debris_btn.disabled = False
                    progress_div.text = ""
                    status.text = (
                        f"Found {n} debris box(es) across frames "
                        f"0–{n_frames - 1}. Review and delete false "
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
        Div(text="Label cells and debris on each of the first N frames. "
                 "Every box is <b>independent</b> — the same cell on two "
                 "frames is two labels, and nothing is tracked. Saved to "
                 "<code>&lt;file&gt;.simple.json</code>.", **_help),
        legend,
        progress_summary,
        Div(text="<b>1 · Add or pick a box</b> — <b>tap</b> a box to select "
                 "it (turns white). <b>➕ Add ROI</b> puts a new one at the "
                 "centre of this frame. Move it with the <i>Box Edit</i> tool "
                 "in the toolbar (Esc cancels a half-drawn box).", **_help),
        row(add_roi_btn, del_roi_btn),
        Div(text="<b>2 · Set its class</b>", width=420),
        row(*(cls_btns[c] for c in TRAINING_CLASSES)),
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
