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
    FILTER_EMPTY,
    FILTER_OVERLAP,
    FILTER_REVIEWED,
    FILTER_UNREVIEWED,
    FILTER_UNVERIFIED,
    ReviewState,
    load_splits,
)

_SHORT = {"single": "S", "doublet": "D", "debris": "deb"}


def modify_doc(doc, dataset_dir: str | Path = ".") -> None:
    from bokeh.events import Tap
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
        Toggle,
    )

    fig, img_src, img_r, box_src, rect_r, mapper = common.build_image_figure(
        box_edit=False)

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
                 "dataset_dir": Path(dataset_dir)}

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
        _render_info()

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

    def _render_image() -> None:
        st = ctx["state"]
        if st is None:
            return
        import tifffile

        img = st.image()
        path = ctx["dataset_dir"] / "images" / str(img["file_name"])
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
        fig.title.text = f"{img['file_name']}  [{st.split_of()}]"
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
        _render_image()
        _render_boxes()

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
            FILTER_ALL, FILTER_UNVERIFIED, FILTER_OVERLAP, FILTER_UNREVIEWED,
            FILTER_REVIEWED, FILTER_EMPTY, *st.classes,
        ]
        filter_select.value = FILTER_ALL
        img_slider.start = 0
        img_slider.end = max(1, len(st.images) - 1)
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
        contrast,
        width=360,
    )
    review_col = column(
        Div(text="<h3 style='margin:2px 0'>Review annotations</h3>", width=420),
        Div(text="Checks the <b>exported dataset</b> — exactly what training "
                 "will see. The <b>“with unverified predictions”</b> filter "
                 "walks only images where a model guessed the class and "
                 "nobody has accepted it yet; a box marked <b>?</b> is one of "
                 "those.", **_help),
        legend,
        info,
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
    doc.add_root(row(column(fig), controls, review_col, spacing=25))
    doc.title = "nikon-control — review annotations"

    _rescan()
    if (Path(dir_input.value) / "annotations").is_dir():
        do_load()
