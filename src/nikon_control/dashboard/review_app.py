"""Bokeh view for the annotation REVIEW dashboard.

A small third dashboard that opens a ``nikon-control-export`` dataset (COCO
+ TIFFs) and lets anyone fix the labels before training. Reviewing the
exported dataset — rather than the sidecars — means what you check is exactly
what the model will be fed.

It shares the viewer with the other dashboards (``common.py``) and uses the
same drag-free ROI positioning as the simplified one (click-to-place +
nudge), so a box can never get stuck under the cursor.

The highest-value workflow is the **"with unverified predictions"** filter:
it walks only the images where a model pre-classified boxes that nobody has
accepted yet.

All logic lives in ``review_state.ReviewState``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..overlap import PREFER_AREA, PREFER_SCORE
from . import common
from .review_state import (
    FILTER_ALL,
    FILTER_DISAGREE,
    FILTER_EMPTY,
    FILTER_OVERLAP,
    FILTER_REVIEWED,
    FILTER_UNREVIEWED,
    FILTER_UNVERIFIED,
    ReviewState,
    load_splits,
)

_SHORT = {"single": "S", "doublet": "D", "debris": "deb"}

# The exported TIFF holds only the channel the model trains on. Other
# channels are read live from the source ND2 (its path is recorded per image
# at export time), so this option always shows the exact training pixels.
_EXPORTED_CHANNEL = "exported (as trained)"


def modify_doc(doc, dataset_dir: str | Path = ".",
               weights_path: str = "") -> None:
    import threading

    from bokeh.events import Tap
    from bokeh.layouts import column, row
    from bokeh.models import (
        Button,
        ColumnDataSource,
        Div,
        LabelSet,
        Range1d,
        RangeSlider,
        Select,
        Slider,
        Spinner,
        TextInput,
        Toggle,
    )

    fig, img_src, img_r, box_src, rect_r, mapper = common.build_image_figure(
        box_edit=False)

    # Model predictions as a SECOND overlay: dashed, and deliberately NOT in
    # the tap tool's renderer list, so they can be compared side by side with
    # the annotations but never selected or edited — they aren't annotations.
    pred_src = ColumnDataSource({"cx": [], "cy": [], "w": [], "h": [],
                                 "text": [], "color": []})
    fig.rect(x="cx", y="cy", width="w", height="h", source=pred_src,
             fill_alpha=0.0, line_color="color", line_width=2,
             line_dash="dashed")
    fig.add_layout(LabelSet(
        x="cx", y="cy", text="text", source=pred_src, text_color="#7fdbff",
        text_font_size="9pt", background_fill_color="#000033",
        background_fill_alpha=0.6, y_offset=-14))

    # ---- widgets --------------------------------------------------------
    drive_select = Select(title="Drive / volume", value="",
                          options=common.list_drives(), width=150)
    dir_input = TextInput(title="Dataset folder (from nikon-control-export)",
                          value=str(dataset_dir), width=340)
    up_btn = Button(label="⬆ Up", width=70)
    subdir_select = Select(title="Subfolders (pick to open)", value="",
                           options=[], width=340)
    load_btn = Button(label="Load dataset", button_type="primary", width=140)

    filter_select = Select(title="Show", value=FILTER_ALL,
                           options=[FILTER_ALL], width=340)
    img_slider = Slider(start=0, end=1, value=0, step=1, title="Image",
                        width=230)
    chan_select = Select(title="Channel", value=_EXPORTED_CHANNEL,
                         options=[_EXPORTED_CHANNEL], width=340)
    prev_btn = Button(label="◀ Prev", width=80)
    next_btn = Button(label="Next ▶", width=80)
    contrast = RangeSlider(start=0, end=65535, value=(0, 65535), step=1,
                           title="Contrast", width=340)

    add_roi_btn = Button(label="➕ Add ROI", button_type="primary", width=145)
    del_roi_btn = Button(label="🗑 Delete ROI", button_type="danger", width=145)
    place_toggle = Toggle(label="🎯 Click on image to place", width=300)
    nudge_spin = Spinner(title="Nudge step (px)", low=1, high=500, step=1,
                         value=10, width=145)
    nudge_up = Button(label="↑", width=45)
    nudge_down = Button(label="↓", width=45)
    nudge_left = Button(label="←", width=45)
    nudge_right = Button(label="→", width=45)
    width_spin = Spinner(title="ROI width (px)", low=4, high=4000, step=2,
                         value=110, width=145)
    height_spin = Spinner(title="ROI height (px)", low=4, high=4000, step=2,
                          value=110, width=145)
    shrink_btn = Button(label="− 10%", width=95)
    grow_btn = Button(label="＋ 10%", width=95)
    cls_row: dict[str, Button] = {}
    cls_holder = column(width=420)
    weights_input = TextInput(title="Model (.pth) to compare against",
                              value=weights_path, width=340)
    score_slider = Slider(start=0.1, end=0.95, value=0.5, step=0.05,
                          title="Prediction score threshold", width=340)
    predict_btn = Button(label="🤖 Run model on all images",
                         button_type="warning", width=300)
    show_pred_toggle = Toggle(label="👁 Show model predictions", active=True,
                              width=300)
    use_model_btn = Button(label="⇦ Use the model's class for this box",
                           width=300)
    agree_div = Div(text="", styles={"font-size": "12px"}, width=420)
    progress_div = Div(text="", styles={
        "font-size": "18px", "font-weight": "bold", "color": "#0a7",
        "padding": "6px 4px"})
    overlap_spin = Spinner(title="Max overlap (%)", low=10, high=100, step=5,
                           value=70, width=145)
    prefer_select = Select(title="On overlap keep the", value="bigger",
                           options=["bigger", "higher-scoring"], width=165)
    dedupe_btn = Button(label="⧈ Remove overlaps (this image)", width=300)
    dedupe_all_btn = Button(label="⧈ Remove overlaps (whole dataset)",
                            width=300)
    reviewed_btn = Button(label="✓ Mark reviewed & next", button_type="success",
                          width=300)
    save_btn = Button(label="💾 Save dataset", button_type="success", width=300)
    legend = Div(text="", styles={"font-size": "11px"}, width=420)
    info = Div(text="", styles={"font-size": "12px"}, width=420)
    status = Div(text="Point at a dataset folder and click Load dataset.",
                 styles={"font-size": "12px"}, width=420)

    ctx: dict = {"state": None, "syncing": False, "selected_ids": [],
                 "place_target": None, "syncing_size": False,
                 "dataset_dir": Path(dataset_dir),
                 "nd2": None, "nd2_source": None}

    # ---- folder browsing -------------------------------------------------
    def _rescan(*_) -> None:
        d = Path(dir_input.value).expanduser()
        try:
            subs, _, _ = common.scan_folder(d)
        except Exception as exc:
            status.text = f"⚠ cannot read folder '{d}': {exc}"
            return
        ctx["dataset_dir"] = d
        subdir_select.options = ["(open a subfolder…)"] + subs
        subdir_select.value = "(open a subfolder…)"
        looks_right = (d / "annotations").is_dir() and (d / "images").is_dir()
        status.text = (
            f"{d} — looks like an export dataset, click Load dataset."
            if looks_right else
            f"{d} — no annotations/ + images/ here; open the dataset folder "
            "produced by nikon-control-export."
        )

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

    drive_select.on_change("value", _on_drive)
    subdir_select.on_change("value", _on_subdir)
    up_btn.on_click(_on_up)

    # ---- rendering --------------------------------------------------------
    def _selected_ids() -> list[str]:
        st = ctx["state"]
        return [i for i in ctx["selected_ids"] if st is not None and st.has(i)]

    def _render_boxes() -> None:
        st = ctx["state"]
        if st is None:
            return
        rows = st.boxes()
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
            box_src.selected.indices = [n for n, r in enumerate(rows)
                                        if r["id"] in sel]
        finally:
            ctx["syncing"] = False
        _render_predictions()
        _render_agreement()
        _render_info()

    def _render_predictions() -> None:
        st = ctx["state"]
        if st is None:
            return
        if not show_pred_toggle.active or not st.has_predictions:
            pred_src.data = {"cx": [], "cy": [], "w": [], "h": [],
                             "text": [], "color": []}
            return
        preds = st.predicted()
        ag = st.agreement()
        extra = set(ag.extra) if ag else set()
        mismatch = {pi for _, pi, _, _ in (ag.class_mismatch if ag else [])}
        cx, cy, w, h, text, color = [], [], [], [], [], []
        for i, pd in enumerate(preds):
            y0, x0, y1, x1 = (float(v) for v in pd["bbox"])
            cx.append((x0 + x1) / 2)
            cy.append((y0 + y1) / 2)
            w.append(x1 - x0)
            h.append(y1 - y0)
            sc = pd.get("score")
            tag = _SHORT.get(str(pd["label"]), str(pd["label"]))
            text.append(f"model {tag}"
                        + (f" {float(sc):.2f}" if sc is not None else ""))
            # red = the model found something nobody annotated; yellow = same
            # object, different class; blue = agreement
            color.append("#ff3b30" if i in extra
                         else "#ffcc00" if i in mismatch else "#7fdbff")
        pred_src.data = {"cx": cx, "cy": cy, "w": w, "h": h,
                         "text": text, "color": color}

    def _render_agreement() -> None:
        st = ctx["state"]
        if st is None:
            return
        if not st.has_predictions:
            agree_div.text = ("<i>No model predictions yet — set a model and "
                              "click <b>Run model on all images</b> to find "
                              "where it disagrees with the annotations.</i>")
            return
        ag = st.agreement()
        t = st.agreement_totals()
        head = (f"<b>Model vs annotation</b> — dataset: "
                f"{t['images_disagreeing']}/{t['images']} image(s) disagree "
                f"({t['class_mismatch']} class, {t['missed']} missed, "
                f"{t['extra']} extra)")
        if ag is None:
            agree_div.text = head
            return
        if ag.agrees:
            body = ("<br><span style='color:#0a7'>This image: model agrees "
                    f"({ag.n_agree} box(es) matched).</span>")
        else:
            reasons = "".join(f"<li>{r}</li>" for r in ag.reasons())
            body = (f"<br><span style='color:#c60'>This image: model found "
                    f"{ag.n_pred} box(es) vs {ag.n_gt} annotated</span>"
                    f"<ul style='margin:2px 0 0 16px'>{reasons}</ul>")
        conf = st.agreement_confusion()
        if conf:
            worst = sorted(conf.items(), key=lambda kv: -kv[1])[:3]
            body += ("<br><span style='color:#888'>most common confusions: "
                     + ", ".join(f"{a}→{b} ×{n}" for (a, b), n in worst)
                     + "</span>")
        agree_div.text = head + body

    def _render_info() -> None:
        st = ctx["state"]
        if st is None:
            return
        img = st.image()
        done, total = st.review_progress()
        vis = len(st.visible())
        counts = st.counts()
        parts = " · ".join(f"<b>{c}</b> {counts.get(c, 0)}" for c in st.classes)
        unver = st.unverified_count()
        # built outside the f-string: nested quotes/backslashes in an
        # f-string expression are a syntax error before Python 3.12
        reviewed_tag = ("  <b style='color:#0a7'>reviewed</b>"
                        if st.is_reviewed() else "")
        unver_tag = ("<br><span style='color:#c60'>"
                     f"{unver} unverified model prediction(s) left</span>"
                     if unver else "")
        n_ov = st.overlap_count(float(overlap_spin.value or 70) / 100.0)
        ov_tag = ("<br><span style='color:#c60'>"
                  f"{n_ov} box(es) here overlap another</span>"
                  if n_ov else "")
        src_name = Path(str(img.get("source", "?"))).name
        info.text = (
            f"<b>{img['file_name']}</b> "
            f"<span style='color:#888'>[{st.split_of()}]</span><br>"
            f"from {src_name} frame {img.get('frame', '?')} · "
            f"{len(st.boxes())} box(es){reviewed_tag}"
            f"<br>image {st.current + 1}/{len(st.images)} "
            f"({vis} match the filter) · reviewed {done}/{total}"
            f"<br>Dataset totals: {parts}{unver_tag}{ov_tag}"
        )

    def _render_legend() -> None:
        st = ctx["state"]
        if st is None:
            return
        colors = common.class_color(st.classes)
        legend.text = "<b>Classes</b>" + "".join(
            f'<div style="margin:3px 0"><span style="color:{colors[c]};'
            f'font-size:16px">■</span> {c} '
            f'<span style="color:#888">({_SHORT.get(c, "")})</span></div>'
            for c in st.classes
        )

    def _nd2_for(source: str):
        """Open (and cache) the source ND2 of the current image.

        Only one handle is kept: switching image closes the previous one, so
        a long review session doesn't accumulate open files. Returns None
        when the ND2 isn't reachable from this machine (e.g. reviewing a
        dataset copied off the microscope server) — the caller then falls
        back to the exported plane.
        """
        if ctx.get("nd2_source") == source and ctx.get("nd2") is not None:
            return ctx["nd2"]
        prev = ctx.get("nd2")
        if prev is not None:
            try:
                prev["file"].close()
            except Exception:
                pass
        ctx["nd2"], ctx["nd2_source"] = None, None
        if not source or not Path(source).exists():
            return None
        try:
            ctx["nd2"] = common.open_nd2(source)
            ctx["nd2_source"] = source
        except Exception:
            return None
        return ctx["nd2"]

    def _channel_names(img: dict) -> list[str]:
        """Channel names for this image: from the COCO entry when the export
        recorded them, else from the ND2, else none."""
        names = [str(c) for c in (img.get("channels") or [])]
        if names:
            return names
        nd = _nd2_for(str(img.get("source", "")))
        return list(nd["channels"]) if nd else []

    def _populate_channels() -> None:
        st = ctx["state"]
        if st is None:
            return
        img = st.image()
        names = _channel_names(img)
        exported_idx = int(img.get("channel", 0) or 0)
        opts = [_EXPORTED_CHANNEL]
        for i, name in enumerate(names):
            # mark which one the exported TIFF actually is
            opts.append(f"{name} (exported)" if i == exported_idx else name)
        keep = chan_select.value if chan_select.value in opts else _EXPORTED_CHANNEL
        ctx["syncing"] = True
        try:
            chan_select.options = opts
            chan_select.value = keep
        finally:
            ctx["syncing"] = False

    def _channel_index(img: dict) -> int | None:
        """Index of the selected channel, or None to use the exported TIFF."""
        choice = chan_select.value
        if not choice or choice == _EXPORTED_CHANNEL:
            return None
        names = _channel_names(img)
        bare = choice[:-len(" (exported)")] if choice.endswith(" (exported)") \
            else choice
        if bare in names:
            idx = names.index(bare)
            # the exported channel is already on disk as a TIFF — cheaper and
            # guaranteed available, so use that rather than re-reading the ND2
            return None if idx == int(img.get("channel", 0) or 0) else idx
        return None

    def _render_image() -> None:
        st = ctx["state"]
        if st is None:
            return
        import tifffile

        img = st.image()
        path = ctx["dataset_dir"] / "images" / str(img["file_name"])
        cidx = _channel_index(img)
        plane = None
        if cidx is not None:
            nd = _nd2_for(str(img.get("source", "")))
            if nd is None:
                status.text = (
                    f"⚠ source ND2 not reachable ({img.get('source')}) — only "
                    "the exported channel can be shown. Review on a machine "
                    "that can see the ND2 to use the other channels."
                )
                chan_select.value = _EXPORTED_CHANNEL
            else:
                try:
                    plane = np.asarray(nd["plane"](int(img.get("frame", 0)),
                                                   cidx))
                except Exception as exc:
                    status.text = f"⚠ cannot read channel from ND2: {exc}"
                    plane = None
        if plane is None:
            try:
                plane = np.asarray(tifffile.imread(path))
            except Exception as exc:
                status.text = f"⚠ cannot read {path.name}: {exc}"
                return
        H, W = plane.shape[-2], plane.shape[-1]
        fig.x_range = Range1d(0, W)
        fig.y_range = Range1d(H, 0)
        img_r.glyph.dw = W
        img_r.glyph.dh = H
        img_src.data = {"image": [plane]}
        chan_txt = ("" if _channel_index(img) is None
                    else f"  —  {chan_select.value}")
        fig.title.text = f"{img['file_name']}  [{st.split_of()}]{chan_txt}"
        mn, mx, lo, hi, step = common.contrast_bounds(plane)
        contrast.step = step
        contrast.start, contrast.end = mn, mx
        contrast.value = (lo, hi)
        mapper.low, mapper.high = lo, hi

    def _show(idx: int | None = None) -> None:
        st = ctx["state"]
        if st is None:
            return
        if idx is not None:
            st.goto(idx)
        ctx["selected_ids"] = []
        ctx["place_target"] = None
        place_toggle.active = False
        ctx["syncing"] = True
        try:
            img_slider.value = st.current
        finally:
            ctx["syncing"] = False
        _populate_channels()
        _render_image()
        _render_boxes()

    def on_channel(attr, old, new) -> None:
        if ctx["syncing"] or ctx["state"] is None:
            return
        _render_image()   # re-autoscales contrast for the new channel

    chan_select.on_change("value", on_channel)

    # ---- load --------------------------------------------------------------
    def do_load(event=None) -> None:
        d = Path(dir_input.value).expanduser()
        try:
            st = ReviewState(load_splits(d), d)
        except Exception as exc:
            status.text = f"⚠ {exc}"
            return
        ctx["state"] = st
        ctx["dataset_dir"] = d
        # class buttons come from the dataset's own categories
        cls_row.clear()
        for c in st.classes:
            cls_row[c] = Button(label=c, width=130)
        for c, b in cls_row.items():
            b.on_click(lambda c=c: _set_class(c))
        cls_holder.children = [row(*cls_row.values())]
        filter_select.options = [
            FILTER_ALL, FILTER_DISAGREE, FILTER_UNVERIFIED, FILTER_OVERLAP,
            FILTER_UNREVIEWED, FILTER_REVIEWED, FILTER_EMPTY, *st.classes,
        ]
        filter_select.value = FILTER_ALL
        img_slider.start = 0
        img_slider.end = max(1, len(st.images) - 1)
        if st.load_predictions():
            meta = st.prediction_meta or {}
            status.text = ("Loaded cached predictions from "
                           f"{meta.get('model', '?')}.")
        _render_legend()
        _show(0)
        counts = st.split_counts()
        status.text = (
            f"Loaded {len(st.images)} image(s) "
            + " + ".join(f"{v} {k}" for k, v in counts.items())
            + f", {sum(st.counts().values())} box(es). "
            "Fix any wrong class, then Save."
        )

    load_btn.on_click(do_load)

    # ---- navigation --------------------------------------------------------
    def on_slider(attr, old, new) -> None:
        if ctx["syncing"] or ctx["state"] is None:
            return
        _show(int(new))

    img_slider.on_change("value", on_slider)

    def _step(delta: int) -> None:
        st = ctx["state"]
        if st is None:
            return
        before = st.current
        st.step(delta)
        if st.current == before:
            status.text = "No further image matches the current filter."
        _show()

    prev_btn.on_click(lambda: _step(-1))
    next_btn.on_click(lambda: _step(1))

    def on_filter(attr, old, new) -> None:
        st = ctx["state"]
        if st is None:
            return
        n = st.set_filter(new)
        if not n:
            status.text = f"No image matches '{new}'."
            return
        status.text = f"{n} image(s) match '{new}'."
        _show()

    filter_select.on_change("value", on_filter)

    def on_contrast(attr, old, new) -> None:
        mapper.low, mapper.high = float(new[0]), float(new[1])

    contrast.on_change("value", on_contrast)

    # ---- selection / edits --------------------------------------------------
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

    def _apply(fn) -> int:
        st = ctx["state"]
        if st is None:
            return 0
        ids = _selected_ids()
        if not ids:
            status.text = "Tap a box first, then use the buttons."
            return 0
        for i in ids:
            fn(st, i)
        _render_boxes()
        return len(ids)

    def _set_class(label: str) -> None:
        n = _apply(lambda st, i: st.set_class(i, label))
        if n:
            status.text = (f"Set {n} box(es) → <b>{label}</b> "
                           "(also marks them verified). Remember to Save.")

    def _delete() -> None:
        n = _apply(lambda st, i: st.delete(i))
        if n:
            ctx["selected_ids"] = []
            _render_boxes()
            status.text = f"Deleted {n} box(es). Remember to Save."

    del_roi_btn.on_click(_delete)

    def _add_roi() -> None:
        st = ctx["state"]
        if st is None:
            return
        plane = img_src.data["image"][0]
        H, W = plane.shape[-2], plane.shape[-1]
        label = st.classes[0]
        key = st.add_box(W / 2.0, H / 2.0, float(width_spin.value),
                         float(height_spin.value), label)
        ctx["selected_ids"] = [key]
        _render_boxes()
        status.text = (f"Added a '{label}' box at the centre — place it, size "
                       "it, then set the right class.")

    add_roi_btn.on_click(_add_roi)

    def _arm_place(active: bool) -> None:
        st = ctx["state"]
        if not active:
            ctx["place_target"] = None
            return
        ids = _selected_ids()
        if st is None or not ids:
            place_toggle.active = False
            status.text = "Tap a box first, then arm placing."
            return
        ctx["place_target"] = ids[0]
        status.text = "Now click on the image where the box should go."

    place_toggle.on_click(_arm_place)

    def _on_tap(event) -> None:
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
        place_toggle.active = False
        _render_boxes()
        status.text = f"Moved box to ({event.x:.0f}, {event.y:.0f})."

    fig.on_event(Tap, _on_tap)

    def _nudge(dx: float, dy: float) -> None:
        step = float(nudge_spin.value or 10)
        _apply(lambda st, i: st.nudge(i, dx * step, dy * step))

    nudge_up.on_click(lambda: _nudge(0, -1))
    nudge_down.on_click(lambda: _nudge(0, 1))
    nudge_left.on_click(lambda: _nudge(-1, 0))
    nudge_right.on_click(lambda: _nudge(1, 0))

    def _on_size(attr, old, new) -> None:
        if ctx["syncing_size"]:
            return
        _apply(lambda st, i: st.resize(i, float(width_spin.value),
                                       float(height_spin.value)))

    width_spin.on_change("value", _on_size)
    height_spin.on_change("value", _on_size)
    shrink_btn.on_click(lambda: _apply(lambda st, i: st.scale(i, 0.9)))
    grow_btn.on_click(lambda: _apply(lambda st, i: st.scale(i, 1.1)))

    # ---- review + save -------------------------------------------------------
    def do_predict(event=None) -> None:
        """Run a model over every image in the dataset, in a worker thread.

        The exported TIFFs are exactly what training consumed, so this is an
        apples-to-apples comparison.
        """
        st = ctx["state"]
        if st is None:
            status.text = "Load a dataset first."
            return
        wpath = weights_input.value.strip()
        if not wpath or not Path(wpath).exists():
            status.text = (f"⚠ Model not found: '{wpath}'. Point it at a "
                           ".pth checkpoint.")
            return
        predict_btn.disabled = True
        status.text = "Loading model…"
        images = [(str(img["file_name"]),
                   ctx["dataset_dir"] / "images" / str(img["file_name"]))
                  for _, img in st.images]
        thr = float(score_slider.value)

        def work() -> None:
            import tifffile

            from ..detector import CellDetector

            def _tick(msg):
                doc.add_next_tick_callback(
                    lambda: setattr(progress_div, "text", msg))

            def _make(device=None):
                return CellDetector(wpath, device=device, score_threshold=thr)

            def _run(det):
                out: dict[str, list[dict]] = {}
                total = len(images)
                for i, (fname, path) in enumerate(images, start=1):
                    if i % 5 == 0 or i == total:
                        _tick(f"Predicting on {det.device.upper()}… "
                              f"{i}/{total}")
                    try:
                        plane = np.asarray(tifffile.imread(path))
                    except Exception:
                        out[fname] = []
                        continue
                    out[fname] = [
                        {"bbox": [float(v) for v in d.bbox],
                         "label": d.label or "single",
                         "score": float(d.score)}
                        for d in det.detect_frame(plane)
                    ]
                return out

            try:
                det = _make()
                try:
                    per_image = _run(det)
                except Exception as exc:
                    if "cuda" in str(exc).lower() and det.device != "cpu":
                        _tick("⚠ GPU error — retrying on CPU…")
                        det = _make(device="cpu")
                        per_image = _run(det)
                    else:
                        raise
                multiclass = det.is_multiclass
                dev = det.device.upper()
                classes = list(det.classes)

                def finish():
                    st.set_predictions(per_image, {
                        "model": wpath, "score_threshold": thr,
                        "device": dev, "multiclass": multiclass,
                        "classes": classes,
                    })
                    st.save_predictions()
                    predict_btn.disabled = False
                    progress_div.text = ""
                    bad = st.disagreeing_indices()
                    _render_boxes()
                    note = ("" if multiclass else
                            " ⚠ This model has a single foreground class, so "
                            "only box COUNTS and positions are compared, not "
                            "types.")
                    status.text = (
                        f"Ran the model on {len(images)} image(s) on {dev}. "
                        f"<b>{len(bad)} image(s) disagree</b> — choose "
                        "<i>where the model disagrees</i> in Show to walk "
                        f"them worst-first.{note}"
                    )
                doc.add_next_tick_callback(finish)
            except Exception as exc:
                def fail(exc=exc):
                    predict_btn.disabled = False
                    progress_div.text = ""
                    status.text = f"⚠ Prediction failed: {exc}"
                doc.add_next_tick_callback(fail)

        threading.Thread(target=work, daemon=True).start()

    predict_btn.on_click(do_predict)

    def _toggle_pred(active: bool) -> None:
        _render_predictions()
        _render_agreement()

    show_pred_toggle.on_click(_toggle_pred)

    def _use_model_class() -> None:
        """Adopt the model's class for the selected annotation."""
        st = ctx["state"]
        if st is None:
            return
        ids = _selected_ids()
        if not ids:
            status.text = "Tap a box first."
            return
        applied = 0
        for i in ids:
            m = st.matched_prediction_for(i)
            if m is not None:
                st.set_class(i, str(m["label"]))
                applied += 1
        _render_boxes()
        status.text = (
            f"Adopted the model's class for {applied} box(es). Remember to "
            "Save." if applied else
            "No model prediction matches the selected box — the model may "
            "have missed it, so the annotation may well be right."
        )

    use_model_btn.on_click(_use_model_class)

    def _dedupe(all_images: bool) -> None:
        st = ctx["state"]
        if st is None:
            return
        thresh = float(overlap_spin.value or 70) / 100.0
        prefer = (PREFER_AREA if prefer_select.value == "bigger"
                  else PREFER_SCORE)
        n = st.suppress_overlaps(thresh, prefer, all_images=all_images)
        ctx["selected_ids"] = [i for i in ctx["selected_ids"] if st.has(i)]
        _render_boxes()
        where = "the whole dataset" if all_images else "this image"
        pct = int(thresh * 100)
        status.text = (
            f"Removed {n} box(es) from {where} overlapping another by more "
            f"than {pct}% (kept the {prefer_select.value} one; boxes a human "
            "made are never removed for a detection). Remember to Save."
            if n else f"Nothing on {where} overlaps by more than {pct}%."
        )

    dedupe_btn.on_click(lambda: _dedupe(False))
    dedupe_all_btn.on_click(lambda: _dedupe(True))

    def _mark_reviewed() -> None:
        st = ctx["state"]
        if st is None:
            return
        st.mark_reviewed(True)
        done, total = st.review_progress()
        _step(1)
        status.text = f"Marked reviewed ({done}/{total}). Remember to Save."

    reviewed_btn.on_click(_mark_reviewed)

    def do_save(event=None) -> None:
        st = ctx["state"]
        if st is None:
            return
        try:
            written = st.save()
        except Exception as exc:
            status.text = f"⚠ Save FAILED: {exc}"
            return
        status.text = ("Saved " + ", ".join(p.name for p in written)
                       + " (originals kept as .json.bak).")

    save_btn.on_click(do_save)

    # ---- layout ---------------------------------------------------------------
    _help = dict(styles={"font-size": "11px", "color": "#666"}, width=420)
    controls = column(
        Div(text="<b>Open an exported dataset</b>"),
        drive_select,
        dir_input,
        row(up_btn, load_btn),
        subdir_select,
        Div(text="<b>Browse</b>"),
        filter_select,
        row(prev_btn, next_btn),
        img_slider,
        Div(text="<b>View</b>"),
        chan_select,
        contrast,
        Div(text="<b>Compare with a model</b>"),
        weights_input,
        score_slider,
        predict_btn,
        show_pred_toggle,
        width=360,
    )
    review_col = column(
        Div(text="<h3 style='margin:2px 0'>Review annotations</h3>", width=420),
        Div(text="Checks the <b>exported dataset</b> — exactly what training "
                 "will see. The <b>“with unverified predictions”</b> filter "
                 "walks only images where a model guessed the class and "
                 "nobody has accepted it yet; a box marked <b>?</b> is one of "
                 "those.", **_help),
        Div(text="Only the training channel is exported as a TIFF; the "
                 "<b>Channel</b> dropdown reads the other channels of the "
                 "same frame live from the source ND2 (so it needs the ND2 "
                 "to be reachable). Boxes are unchanged — the coordinates are "
                 "the same in every channel.", **_help),
        legend,
        info,
        agree_div,
        Div(text="Dashed boxes are the <b>model's</b> predictions: "
                 "<span style='color:#0aa'>blue</span> matched an annotation, "
                 "<span style='color:#c60'>yellow</span> matched but with a "
                 "<b>different class</b>, <span style='color:#c00'>red</span> "
                 "matched nothing annotated. They can't be selected or "
                 "edited. A disagreement may be the model's fault or the "
                 "annotation's — that's why it's worth a look.", **_help),
        use_model_btn,
        Div(text="<b>Fix the class</b> — tap a box, then:", width=420),
        cls_holder,
        Div(text="<b>Fix the box</b> — no dragging: arm 🎯 and click where it "
                 "belongs, or nudge with the arrows.", **_help),
        row(add_roi_btn, del_roi_btn),
        place_toggle,
        row(nudge_spin, nudge_left, nudge_right, nudge_up, nudge_down),
        row(width_spin, height_spin),
        row(shrink_btn, grow_btn),
        Div(text="<b>Duplicates</b> — a multi-class model can return the same "
                 "object as two classes. Overlap is measured against the "
                 "<i>smaller</i> box, so a single inside a doublet counts as "
                 "100%.", **_help),
        row(overlap_spin, prefer_select),
        dedupe_btn,
        dedupe_all_btn,
        reviewed_btn,
        save_btn,
        Div(text="⚠ Re-running <code>nikon-control-export</code> rebuilds the "
                 "dataset from the <code>*.simple.json</code> sidecars and "
                 "<b>discards edits made here</b>. Review is the last QC step "
                 "before training; fixes you want to keep permanently belong "
                 "in the <a href='/simple'>simple annotation</a> dashboard.",
            **_help),
        status,
        width=440,
    )
    doc.add_root(row(column(fig, progress_div), controls, review_col,
                     spacing=25))
    doc.title = "nikon-control — review annotations"

    def _cleanup(session_context) -> None:
        nd = ctx.get("nd2")
        if nd is not None:
            try:
                nd["file"].close()
            except Exception:
                pass

    try:
        doc.on_session_destroyed(_cleanup)
    except Exception:
        pass  # bare Document (tests) has no session lifecycle

    _rescan()
    if (Path(dir_input.value) / "annotations").is_dir():
        do_load()
