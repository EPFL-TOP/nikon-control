"""Route ``/scope`` — drive the microscope and see where it is.

A view onto :class:`nikon_control.scope.control.Scope`; all the hardware
behaviour lives there, and this module only turns it into widgets. That split
is deliberate — the same operations have to be scriptable for the unattended
40× scan, so none of them may depend on a browser being open.

Three panels, in the order a session actually goes:

**Connect** — load a Micro-Manager ``.cfg``, or the demo devices to try the
dashboard with no microscope attached. The status line then names which
device filled which role, so a missing PFS is visible before it matters.

**Drive** — live/snapped image, stage jog, focus, PFS, objective, channel.
Focus deserves a note: the ``Focus ±`` buttons call ``focus_by()``, which
moves the **PFS offset** while PFS is locked and the **Z drive** when it is
not, because moving Z under a lock does nothing except fight the lock.

**Plate** — register the plate by driving to three wells and capturing each,
then a map of every well in stage coordinates with the objective's position
on it. Arm "click a well to drive there" and the map becomes the navigator.
That map is the check that the registration is right: if the marker does not
sit in the well you are looking at, the fit is wrong, and it is far easier to
see that here than to infer it from a scan that missed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..scope import plate as plate_mod
from ..scope.control import Scope, ScopeError
from .common import build_image_figure, contrast_bounds

# Poll the hardware this often while connected. Fast enough that a jog feels
# immediate, slow enough that it does not saturate a serial stand.
POLL_MS = 400
# Cap the live-view rate regardless of exposure — the browser is the
# bottleneck long before the camera is.
LIVE_MIN_MS = 200

PLATE_TYPES = ["96-well", "384-well", "24-well", "12-well", "6-well",
               "48-well", "1536-well"]

_OK = "#34c759"
_WARN = "#ff9500"
_BAD = "#ff3b30"
_DIM = "#8e8e93"


def _badge(text: str, color: str) -> str:
    return (f'<span style="display:inline-block;padding:2px 8px;border-radius:'
            f'10px;background:{color};color:white;font-weight:600;'
            f'font-size:11px">{text}</span>')


def modify_doc(doc, config_path: str = "", plate_path: str = "") -> None:
    from bokeh.layouts import column, gridplot, row
    from bokeh.models import (
        Button,
        CheckboxGroup,
        ColumnDataSource,
        Div,
        RangeSlider,
        Select,
        Spinner,
        TapTool,
        TextInput,
        Toggle,
    )
    from bokeh.plotting import figure

    # ----------------------------------------------------------- app state
    state: dict = {
        "scope": None,          # Scope | None
        "cal": None,            # PlateCalibration | None
        "refs": [],             # list[plate_mod.WellRef]
        "live_cb": None,        # periodic callback handle
        "auto_contrast": True,
        # Set while the poll writes hardware readings into widgets. Without
        # it, showing the current objective in the Select looks exactly like
        # a user asking to rotate the turret — and the poll would fight every
        # value the user set, four times a second.
        "syncing": False,
    }

    # ------------------------------------------------------------- connect
    cfg_input = TextInput(title="Micro-Manager configuration (.cfg)",
                          value=config_path, width=460)
    connect_btn = Button(label="Connect", button_type="primary", width=110,
                         name="connect")
    demo_btn = Button(label="Demo devices", width=130, name="demo")
    conn_div = Div(text=_badge("not connected", _DIM), width=460,
                   name="status")
    roles_div = Div(text="", width=460,
                    styles={"font-size": "11px", "color": "#666"})

    # --------------------------------------------------------------- image
    fig, img_src, img_r, _box_src, _rect_r, mapper = build_image_figure(
        width=620, height=620, box_edit=False)
    fig.title.text = "no image yet"
    contrast = RangeSlider(start=0, end=65535, value=(0, 65535), step=1,
                           title="Contrast", width=600)
    snap_btn = Button(label="📷 Snap", button_type="primary", width=110,
                      name="snap")
    live_tog = Toggle(label="▶ Live", width=110)
    auto_ct = CheckboxGroup(labels=["auto contrast"], active=[0], width=130)
    exposure = Spinner(title="Exposure (ms)", low=1, high=10000, step=1,
                       value=10, width=120)
    channel_sel = Select(title="Channel", options=[], value="", width=150)

    # --------------------------------------------------------------- drive
    pos_div = Div(text="—", width=380, name="position",
                  styles={"font-family": "monospace", "font-size": "13px"})
    step_spin = Spinner(title="Stage step (µm)", low=0.1, high=5000, step=10,
                        value=100, width=120)
    up_btn = Button(label="↑", width=52, name="up")
    down_btn = Button(label="↓", width=52)
    left_btn = Button(label="←", width=52)
    right_btn = Button(label="→", width=52, name="right")
    goto_x = Spinner(title="X (µm)", value=0, step=100, width=110)
    goto_y = Spinner(title="Y (µm)", value=0, step=100, width=110)
    goto_btn = Button(label="Go to XY", width=100)

    zstep_spin = Spinner(title="Focus step (µm)", low=0.01, high=500,
                         step=0.5, value=1.0, width=120)
    zup_btn = Button(label="Focus +", width=90)
    zdn_btn = Button(label="Focus −", width=90)
    focus_note = Div(text="", width=380,
                     styles={"font-size": "11px", "color": "#666"})

    pfs_div = Div(text=_badge("PFS —", _DIM), width=200, name="pfs")
    pfs_on_btn = Button(label="Engage PFS", button_type="success", width=110,
                        name="pfs_on")
    pfs_off_btn = Button(label="Disengage", width=100)

    obj_sel = Select(title="Objective", options=[], value="", width=230)
    obj_ok = CheckboxGroup(labels=["allow turret move"], active=[], width=230)

    # --------------------------------------------------------------- plate
    plate_sel = Select(title="Plate", options=PLATE_TYPES, value="96-well",
                       width=130)
    well_input = TextInput(title="Well", value="A1", width=80, name="well")
    capture_btn = Button(label="Capture current XY", button_type="primary",
                         width=160, name="capture")
    clear_btn = Button(label="Clear", width=70)
    calib_btn = Button(label="Calibrate", button_type="success", width=110,
                       name="calibrate")
    suggest_div = Div(text="", width=520,
                      styles={"font-size": "12px", "color": "#666"})
    refs_div = Div(text="", width=520, styles={"font-size": "12px"})
    cal_div = Div(text="", width=520, name="calibration")
    plate_file = TextInput(title="Calibration file", value=plate_path or
                           "plate.json", width=380)
    save_btn = Button(label="Save", width=80)
    load_btn = Button(label="Load", width=80)

    well_src = ColumnDataSource({"name": [], "x": [], "y": [], "w": [],
                                 "h": [], "color": []}, name="wells")
    here_src = ColumnDataSource({"x": [], "y": []}, name="here")
    pmap = figure(width=520, height=380, match_aspect=True,
                  tools="pan,wheel_zoom,reset", title="plate not registered",
                  x_axis_label="stage x (µm)", y_axis_label="stage y (µm)")
    well_r = pmap.ellipse(x="x", y="y", width="w", height="h", source=well_src,
                          fill_color="color", fill_alpha=0.25,
                          line_color="#555", line_width=1,
                          selection_fill_alpha=0.6,
                          nonselection_fill_alpha=0.25)
    pmap.scatter(x="x", y="y", source=here_src, marker="cross", size=18,
                 line_color=_BAD, line_width=3)
    pmap.add_tools(TapTool(renderers=[well_r]))
    goto_well_tog = Toggle(label="🔒 click a well to drive there", width=250)

    # ------------------------------------------------------------ helpers

    def say(msg: str, color: str = _DIM) -> None:
        conn_div.text = _badge(msg, color)

    def scope() -> Scope | None:
        return state["scope"]

    def run(fn) -> None:
        """Run a hardware action, reporting a failure instead of raising.

        Every control goes through this: a Bokeh callback that raises leaves
        the user staring at an unchanged page with the reason only in the
        server log, which over RDP is invisible.
        """
        s = scope()
        if s is None:
            say("not connected", _WARN)
            return
        try:
            fn(s)
        except ScopeError as exc:
            say(str(exc), _WARN)
        except Exception as exc:                       # noqa: BLE001
            say(f"{type(exc).__name__}: {exc}", _BAD)
        else:
            refresh()

    def guard(fn):
        """Wrap a hardware action as a Button.on_click handler."""
        def on_click():
            run(fn)
        return on_click

    def guard_change(fn):
        """Wrap ``fn(scope, new_value)`` as a widget on_change handler.

        Bokeh validates handler signatures, so the two shapes cannot share
        one wrapper. Ignores changes the poll made itself.
        """
        def on_change(attr, old, new):
            if state["syncing"] or new == old:
                return
            run(lambda s: fn(s, new))
        return on_change

    def sync(widget, value, attr: str = "value") -> None:
        """Show a hardware reading in a widget without it reading as an edit."""
        if getattr(widget, attr) == value:
            return
        state["syncing"] = True
        try:
            setattr(widget, attr, value)
        finally:
            state["syncing"] = False

    # ------------------------------------------------------------- connect

    def _after_connect(s: Scope, what: str) -> None:
        state["scope"] = s
        named = ", ".join(f"<b>{r}</b>={n}" for r, n in s.roles.items())
        roles_div.text = f"roles: {named or 'none resolved'}"
        missing = [r for r in ("xystage", "focus", "camera")
                   if not s.has(r)]
        if missing:
            say(f"{what} — missing: {', '.join(missing)}", _WARN)
        else:
            say(f"connected: {what}", _OK)
        sync(obj_sel, s.objectives(), "options")
        sync(channel_sel, s.channels(), "options")
        refresh()

    def do_connect() -> None:
        path = cfg_input.value.strip()
        if not path:
            say("give a .cfg path, or use Demo devices", _WARN)
            return
        if not Path(path).exists():
            say(f"no such file: {path}", _BAD)
            return
        try:
            _after_connect(Scope.from_config(path), Path(path).name)
        except Exception as exc:                        # noqa: BLE001
            say(f"could not load configuration: {exc}", _BAD)

    def do_demo() -> None:
        try:
            _after_connect(Scope.demo(), "demo devices")
        except Exception as exc:                        # noqa: BLE001
            say(f"demo devices unavailable: {exc}", _BAD)

    connect_btn.on_click(do_connect)
    demo_btn.on_click(do_demo)

    # --------------------------------------------------------------- image

    def show(frame) -> None:
        arr = np.asarray(frame)
        if arr.ndim > 2:
            arr = arr[..., 0] if arr.shape[-1] <= 4 else arr[0]
        h, w = arr.shape[:2]
        img_src.data = {"image": [arr.astype(np.float32)]}
        img_r.glyph.update(dw=w, dh=h)
        if state["auto_contrast"]:
            mn, mx, lo, hi, step = contrast_bounds(arr)
            contrast.start, contrast.end, contrast.step = mn, mx, step
            contrast.value = (lo, hi)
            mapper.low, mapper.high = lo, hi
        fig.title.text = f"{w}×{h}  {arr.dtype}"

    def do_snap(_s=None) -> None:
        s = scope()
        if s is None:
            say("not connected", _WARN)
            return
        try:
            show(s.snap())
        except Exception as exc:                        # noqa: BLE001
            say(f"snap failed: {exc}", _BAD)
            live_tog.active = False

    snap_btn.on_click(lambda: do_snap())

    def on_live(attr, old, new) -> None:
        if new and state["live_cb"] is None:
            s = scope()
            period = LIVE_MIN_MS
            if s is not None:
                try:
                    period = max(LIVE_MIN_MS, int(s.exposure_ms()) + 50)
                except Exception:
                    pass
            state["live_cb"] = doc.add_periodic_callback(do_snap, period)
            live_tog.label = "⏸ Stop"
        elif not new and state["live_cb"] is not None:
            doc.remove_periodic_callback(state["live_cb"])
            state["live_cb"] = None
            live_tog.label = "▶ Live"

    live_tog.on_change("active", on_live)
    contrast.on_change("value", lambda a, o, n: mapper.update(low=n[0],
                                                              high=n[1]))
    auto_ct.on_change("active",
                      lambda a, o, n: state.update(auto_contrast=bool(n)))
    exposure.on_change("value", guard_change(
        lambda s, new: s.set_exposure_ms(float(new))))
    channel_sel.on_change("value", guard_change(
        lambda s, new: s.set_channel(new) if new else None))

    # --------------------------------------------------------------- drive
    up_btn.on_click(guard(lambda s: s.move_xy_by(0, float(step_spin.value))))
    down_btn.on_click(guard(lambda s: s.move_xy_by(0, -float(step_spin.value))))
    left_btn.on_click(guard(lambda s: s.move_xy_by(-float(step_spin.value), 0)))
    right_btn.on_click(guard(lambda s: s.move_xy_by(float(step_spin.value), 0)))
    goto_btn.on_click(guard(lambda s: s.move_xy(float(goto_x.value),
                                                float(goto_y.value))))
    zup_btn.on_click(guard(lambda s: s.focus_by(float(zstep_spin.value))))
    zdn_btn.on_click(guard(lambda s: s.focus_by(-float(zstep_spin.value))))

    def do_engage(s: Scope) -> None:
        if not s.engage_pfs():
            say("PFS engaged but did not lock — out of range?", _WARN)

    pfs_on_btn.on_click(guard(do_engage))
    pfs_off_btn.on_click(guard(lambda s: s.disengage_pfs()))

    def on_objective(attr, old, new) -> None:
        s = scope()
        if s is None or not new or new == old or state["syncing"]:
            return
        if 0 not in obj_ok.active:
            say("tick 'allow turret move' first — rotating the turret under "
                "a plate can crash the objective", _WARN)
            # Revert through sync(), or this assignment re-enters the handler
            # and reverts again, for ever.
            sync(obj_sel, old)
            return
        run(lambda sc: sc.set_objective(new, confirm=True))

    obj_sel.on_change("value", on_objective)

    # --------------------------------------------------------------- plate

    def redraw_plate() -> None:
        cal = state["cal"]
        if cal is None:
            well_src.data = {k: [] for k in well_src.data}
            pmap.title.text = "plate not registered"
            return
        lay = plate_mod.layout(cal)
        captured = {r.name.upper() for r in state["refs"]}
        well_src.data = {
            "name": lay.names, "x": lay.x, "y": lay.y,
            "w": [lay.well_width_um] * len(lay.names),
            "h": [lay.well_height_um] * len(lay.names),
            # the wells that were actually measured, marked
            "color": ["#ffcc00" if n.upper() in captured else "#4a90d9"
                      for n in lay.names],
        }
        pmap.title.text = (f"{cal.plate} · rotation {cal.rotation:+.3f}° · "
                           f"residual {cal.residual_um:.0f} µm")

    def show_refs() -> None:
        if not state["refs"]:
            refs_div.text = "<i>no wells captured yet</i>"
            return
        refs_div.text = " &nbsp; ".join(
            f"<b>{r.name}</b> ({r.x:.0f}, {r.y:.0f})" for r in state["refs"])

    def do_suggest() -> None:
        try:
            a, b, c = plate_mod.suggested_refs(plate_sel.value)
        except Exception as exc:                        # noqa: BLE001
            suggest_div.text = f"unknown plate: {exc}"
            return
        suggest_div.text = (
            f"Drive to each of <b>{a}</b>, <b>{b}</b>, <b>{c}</b>, centre it, "
            f"and capture. Corners pin down the rotation — adjacent wells "
            f"barely constrain it.")
        well_input.value = a

    plate_sel.on_change("value", lambda attr, old, new: do_suggest())

    def do_capture() -> None:
        s = scope()
        if s is None:
            say("not connected", _WARN)
            return
        name = well_input.value.strip().upper()
        if not name:
            say("name the well first", _WARN)
            return
        try:
            here = s.xy()
        except Exception as exc:                        # noqa: BLE001
            say(f"could not read the stage: {exc}", _BAD)
            return
        state["refs"] = [r for r in state["refs"] if r.name.upper() != name]
        state["refs"].append(plate_mod.WellRef(name, here.x, here.y))
        show_refs()
        say(f"captured {name} at ({here.x:.0f}, {here.y:.0f}) µm", _OK)

    def do_clear() -> None:
        state["refs"] = []
        show_refs()

    def do_calibrate() -> None:
        if not state["refs"]:
            cal_div.text = _badge("capture at least one well first", _WARN)
            return
        try:
            cal = plate_mod.calibrate(plate_sel.value, state["refs"])
        except ValueError as exc:
            # A refused fit is the useful answer, not a failure to report away.
            state["cal"] = None
            redraw_plate()
            cal_div.text = _badge(f"refused: {exc}", _BAD)
            return
        state["cal"] = cal
        notes = []
        if not cal.rotation_estimated:
            notes.append("rotation assumed 0 — capture a far corner to "
                         "measure it")
        if cal.residual_um > 50:
            notes.append(f"residual {cal.residual_um:.0f} µm is large for "
                         f"hand-centred wells; re-check the well names")
        cal_div.text = (_badge("registered", _OK) + " " + cal.describe() +
                        ("<br><span style='color:#b36b00'>" +
                         "; ".join(notes) + "</span>" if notes else ""))
        redraw_plate()
        refresh()

    capture_btn.on_click(do_capture)
    clear_btn.on_click(do_clear)
    calib_btn.on_click(do_calibrate)

    def do_save() -> None:
        if state["cal"] is None:
            cal_div.text = _badge("nothing to save — calibrate first", _WARN)
            return
        try:
            plate_mod.save(state["cal"], plate_file.value)
            cal_div.text = _badge(f"saved to {plate_file.value}", _OK)
        except Exception as exc:                        # noqa: BLE001
            cal_div.text = _badge(f"could not save: {exc}", _BAD)

    def do_load() -> None:
        try:
            cal = plate_mod.load(plate_file.value)
        except Exception as exc:                        # noqa: BLE001
            cal_div.text = _badge(f"could not load: {exc}", _BAD)
            return
        state["cal"] = cal
        if cal.plate in PLATE_TYPES:
            plate_sel.value = cal.plate
        cal_div.text = _badge("loaded", _OK) + " " + cal.describe()
        redraw_plate()
        refresh()

    save_btn.on_click(do_save)
    load_btn.on_click(do_load)

    def on_well_tap(attr, old, new) -> None:
        if not new:
            return
        idx = new[0]
        name = well_src.data["name"][idx]
        wx = well_src.data["x"][idx]
        wy = well_src.data["y"][idx]
        if not goto_well_tog.active:
            say(f"{name} at ({wx:.0f}, {wy:.0f}) µm — arm the toggle to drive "
                f"there", _DIM)
            well_src.selected.indices = []
            return
        run(lambda s: s.move_xy(wx, wy))
        say(f"moved to {name}", _OK)
        well_src.selected.indices = []

    well_src.selected.on_change("indices", on_well_tap)

    # -------------------------------------------------------------- refresh

    def refresh() -> None:
        """Poll the hardware and repaint every readout."""
        s = scope()
        if s is None:
            return
        st = s.state()

        if st.xy is not None:
            pos_div.text = (f"X {st.xy.x:10.1f}   Y {st.xy.y:10.1f} µm"
                            + (f"<br>Z {st.z:10.2f} µm" if st.z is not None
                               else ""))
            here_src.data = {"x": [st.xy.x], "y": [st.xy.y]}
            cal = state["cal"]
            if cal is not None:
                name, _wx, _wy, dist = plate_mod.nearest_well(cal, st.xy.x,
                                                              st.xy.y)
                pos_div.text += (f"<br><b>{name}</b> "
                                 f"<span style='color:#888'>"
                                 f"({dist:.0f} µm from centre)</span>")
        elif st.z is not None:
            pos_div.text = f"Z {st.z:10.2f} µm"

        if not st.pfs_available:
            pfs_div.text = _badge("no PFS in this configuration", _DIM)
        elif st.pfs_locked:
            pfs_div.text = _badge("PFS locked", _OK)
        elif st.pfs_engaged:
            pfs_div.text = _badge("PFS on — not locked", _WARN)
        else:
            pfs_div.text = _badge("PFS off", _DIM)
        if st.pfs_offset is not None:
            pfs_div.text += f" <span style='font-family:monospace'>" \
                            f"offset {st.pfs_offset:.1f}</span>"

        focus_note.text = (
            "Focus ± moves the <b>PFS offset</b> (the only control that "
            "changes focus while locked)." if st.pfs_locked and
            st.pfs_offset is not None else
            "Focus ± moves the <b>Z drive</b>.")

        if st.objectives:
            sync(obj_sel, st.objectives, "options")
        if st.objective:
            sync(obj_sel, st.objective)
        if st.channels:
            sync(channel_sel, st.channels, "options")
        if st.channel:
            sync(channel_sel, st.channel)
        if st.exposure_ms is not None:
            sync(exposure, st.exposure_ms)
        if st.error:
            say(st.error, _WARN)

    doc.add_periodic_callback(refresh, POLL_MS)

    # --------------------------------------------------------------- layout
    jog = gridplot(
        [[None, up_btn, None],
         [left_btn, None, right_btn],
         [None, down_btn, None]],
        toolbar_location=None, merge_tools=False)

    connect_panel = column(
        Div(text="<h3 style='margin:0'>1 · Connect</h3>"),
        row(cfg_input, column(Div(text="<br>"), row(connect_btn, demo_btn))),
        conn_div, roles_div,
    )

    drive_panel = column(
        Div(text="<h3 style='margin:0'>2 · Drive</h3>"),
        pos_div,
        row(jog, column(step_spin, row(goto_x, goto_y), goto_btn)),
        row(zdn_btn, zup_btn, zstep_spin),
        focus_note,
        row(pfs_div),
        row(pfs_on_btn, pfs_off_btn),
        obj_sel, obj_ok,
    )

    image_panel = column(
        row(snap_btn, live_tog, auto_ct),
        row(exposure, channel_sel),
        fig, contrast,
    )

    plate_panel = column(
        Div(text="<h3 style='margin:0'>3 · Plate</h3>"),
        row(plate_sel, well_input, capture_btn, clear_btn, calib_btn),
        suggest_div, refs_div, cal_div,
        row(plate_file, column(Div(text="<br>"), row(save_btn, load_btn))),
        goto_well_tog, pmap,
    )

    doc.add_root(column(
        connect_panel,
        row(image_panel, column(drive_panel, plate_panel)),
    ))
    doc.title = "Nikon scope control"

    do_suggest()
    show_refs()
    if plate_path and Path(plate_path).exists():
        plate_file.value = plate_path
        do_load()
