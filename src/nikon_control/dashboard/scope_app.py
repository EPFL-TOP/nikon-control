"""Route ``/scope`` — drive the microscope and see where it is.

A view onto :class:`nikon_control.scope.control.Scope`; all the hardware
behaviour lives there, and this module only turns it into widgets. That split
is deliberate — the same operations have to be scriptable for the unattended
40× scan, so none of them may depend on a browser being open.

Three panels, in the order a session actually goes:

**Connect** — load a Micro-Manager ``.cfg``; or **build one** by connecting
every attached device in turn and writing out what answered; or the demo
devices, to try the dashboard with no microscope. The status line then names
which device filled which role, so a missing PFS is visible before it
matters.

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

from ..scope import (channels as channels_mod, config_build,
                     plate as plate_mod, scan as scan_mod,
                     timing as timing_mod, wells as wells_mod)
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
        MultiChoice,
        RangeSlider,
        RadioButtonGroup,
        Select,
        Slider,
        Spinner,
        TabPanel,
        Tabs,
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
        "timings": None,        # timing_mod.Timings once measured
        "sel": None,            # wells_mod.Selection
        "edges": {},            # well -> {"left": x, "right": x, ...}
        "scan_plan": None,      # scan_mod.ScanPlan
        "scan_iter": None,      # the running generator
        "scan_cb": None,        # its periodic callback
        "scan_stop": False,
    }

    # ------------------------------------------------------------- connect
    cfg_input = TextInput(title="Micro-Manager configuration (.cfg)",
                          value=config_path, width=460)
    connect_btn = Button(label="Connect", button_type="primary", width=110,
                         name="connect")
    demo_btn = Button(label="Demo devices", width=130, name="demo")
    build_btn = Button(label="Build from hardware", width=160, name="build")
    overwrite_ck = CheckboxGroup(labels=["replace the existing file"],
                                 active=[], width=210, name="overwrite")
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

    zstep_spin = Spinner(title="Z step (µm)", low=0.01, high=500,
                         step=0.5, value=1.0, width=110)
    zup_btn = Button(label="Z +", width=70, name="z_up")
    zdn_btn = Button(label="Z −", width=70)
    offstep_spin = Spinner(title="Offset step", low=0.01, high=10000,
                           step=1, value=10, width=110)
    offup_btn = Button(label="Offset +", width=90, name="offset_up")
    offdn_btn = Button(label="Offset −", width=90)
    offset_spin = Spinner(title="PFS offset", step=1, value=0, width=130,
                          name="pfs_offset")
    focus_note = Div(text="", width=400,
                     styles={"font-size": "11px", "color": "#666"})

    light_div = Div(text=_badge("light —", _DIM), width=380, name="light")
    shutter_tog = Toggle(label="💡 Open shutter", width=140, name="shutter")
    autoshut_ck = CheckboxGroup(labels=["auto-shutter"], active=[0], width=140)
    intensity = Slider(start=0, end=100, value=0, step=1, width=300,
                       title="Intensity", name="intensity", disabled=True)
    intensity_note = Div(text="", width=380,
                         styles={"font-size": "11px", "color": "#666"})

    pfs_div = Div(text=_badge("PFS —", _DIM), width=200, name="pfs")
    pfs_on_btn = Button(label="Engage PFS", button_type="success", width=110,
                        name="pfs_on")
    pfs_off_btn = Button(label="Disengage", width=100)

    obj_sel = Select(title="Objective", options=[], value="", width=230)
    obj_ok = CheckboxGroup(labels=["allow turret move"], active=[], width=230)

    # ------------------------------------------------------------ channels
    chan_name = TextInput(title="New channel name", value="BF", width=150,
                          name="channel_name")
    chan_capture = Button(label="Capture current settings", width=200,
                          button_type="primary", name="capture_channel")
    chan_exposure_ck = CheckboxGroup(labels=["include exposure"], active=[0],
                                     width=160)
    chan_div = Div(text="", width=520, name="channel_status")
    chan_list = Div(text="", width=520, styles={"font-size": "12px"})

    # ---------------------------------------------------------- throughput
    acq_channels = MultiChoice(title="Channels per position", value=[],
                               options=[], width=340, name="acq_channels")
    interval_spin = Spinner(title="Interval (min)", low=0.1, high=600,
                            step=1, value=5, width=110)
    duration_spin = Spinner(title="Movie length (h)", low=0.1, high=200,
                            step=1, value=10, width=120)
    npos_spin = Spinner(title="Positions wanted", low=1, high=10000, step=10,
                        value=100, width=130)
    movedist_spin = Spinner(title="Typical hop (µm)", low=0, high=100000,
                            step=100, value=2000, width=130)
    measure_btn = Button(label="⏱ Measure on the microscope", width=230,
                         button_type="primary", name="measure")
    timing_div = Div(text="<i>press Measure — the numbers that matter "
                          "(stage settling, readout, PFS lock) cannot be "
                          "guessed.</i>", width=520, name="timing")
    budget_div = Div(text="", width=520, name="budget",
                     styles={"font-size": "14px"})

    # ---------------------------------------------------------------- scan
    px_spin = Spinner(title="Pixel size (µm/px)", low=0.0001, high=100,
                      step=0.001, value=0.1625, width=150, name="pixel_size")
    px_set = Button(label="Apply", width=80, name="pixel_set")
    px_div = Div(text="", width=520, styles={"font-size": "12px"},
                 name="pixel_status")
    cover_spin = Spinner(title="Cover this fraction of each well", low=0.01,
                         high=1.5, step=0.05, value=0.33, width=220)
    scan_rows = Spinner(title="Rows", low=1, high=200, step=1, value=7,
                        width=90, name="scan_rows")
    scan_cols = Spinner(title="Columns", low=1, high=200, step=1, value=7,
                        width=100, name="scan_cols")
    fit_btn = Button(label="Fit grid to coverage", width=180, name="fit_grid")
    overlap_spin = Spinner(title="Overlap (%)", low=0, high=90, step=5,
                           value=10, width=110)
    refocus_spin = Spinner(title="Refocus every N fields (0 = never)", low=0,
                           high=1000, step=1, value=1, width=250)
    scan_chan = Select(title="Channel (blank = leave as is)", options=[],
                       value="", width=230, name="scan_channel")
    scan_dir = TextInput(title="Write frames to", value="scan", width=330,
                         name="scan_dir")
    plan_btn = Button(label="Plan", width=90, name="scan_plan")
    scan_btn = Button(label="▶ Run scan", button_type="primary", width=130,
                      name="scan_run")
    stop_btn = Button(label="■ Stop", width=90, name="scan_stop")
    scan_div = Div(text="", width=520, name="scan_status")
    scan_progress = Div(text="", width=520, name="scan_progress",
                        styles={"font-family": "monospace",
                                "font-size": "12px"})

    # --------------------------------------------------------------- plate
    plate_sel = Select(title="Plate", options=PLATE_TYPES, value="96-well",
                       width=130, name="plate_type")
    well_input = TextInput(title="Well", value="A1", width=80, name="well")
    capture_btn = Button(label="Centre is here", button_type="primary",
                         width=140, name="capture")
    edge_x0 = Button(label="◀ −X wall", width=95, name="edge_x0")
    edge_x1 = Button(label="+X wall ▶", width=95, name="edge_x1")
    edge_y0 = Button(label="▼ −Y wall", width=95, name="edge_y0")
    edge_y1 = Button(label="▲ +Y wall", width=95, name="edge_y1")
    edge_use = Button(label="Centre from walls", button_type="success",
                      width=160, name="edge_use")
    edge_div = Div(text="", width=520, name="edge_status",
                   styles={"font-size": "12px"})
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

    # Selection and reference are independent states, so they use
    # independent channels: fill says selected, outline says measured.
    well_src = ColumnDataSource({"name": [], "x": [], "y": [], "w": [],
                                 "h": [], "color": [], "line": [],
                                 "lw": []}, name="wells")
    here_src = ColumnDataSource({"x": [], "y": []}, name="here")
    pmap = figure(width=520, height=380, match_aspect=True,
                  tools="pan,wheel_zoom,reset", title="plate not registered",
                  x_axis_label="stage x (µm)", y_axis_label="stage y (µm)")
    well_r = pmap.ellipse(x="x", y="y", width="w", height="h", source=well_src,
                          fill_color="color", fill_alpha=0.55,
                          line_color="line", line_width="lw",
                          selection_fill_alpha=0.75,
                          nonselection_fill_alpha=0.55)
    pmap.scatter(x="x", y="y", source=here_src, marker="cross", size=18,
                 line_color=_BAD, line_width=3)
    pmap.add_tools(TapTool(renderers=[well_r]))
    click_mode = RadioButtonGroup(labels=["click = select", "click = drive"],
                                  active=0, width=260, name="click_mode")
    wells_text = TextInput(title="Wells (A1, B2-B5, C*, *3, A1:D6, all)",
                           value="", width=330, name="wells_text")
    wells_add = Button(label="Select", width=80, name="wells_add")
    wells_sub = Button(label="Deselect", width=90)
    wells_all = Button(label="All", width=60)
    wells_none = Button(label="None", width=70, name="wells_none")
    wells_div = Div(text="", width=520, name="wells_status")
    serp_ck = CheckboxGroup(labels=["serpentine order (halves travel)"],
                            active=[0], width=280)

    # ------------------------------------------------------------ helpers

    def say(msg: str, color: str = _DIM) -> None:
        conn_div.text = _badge(msg, color)

    def scope() -> Scope | None:
        return state["scope"]

    def run(fn) -> bool:
        """Run a hardware action, reporting a failure instead of raising.

        Returns whether it actually ran, so a caller cannot announce success
        over the top of the failure this just displayed.

        Every control goes through this: a Bokeh callback that raises leaves
        the user staring at an unchanged page with the reason only in the
        server log, which over RDP is invisible.
        """
        s = scope()
        if s is None:
            say("not connected", _WARN)
            return False
        try:
            fn(s)
        except ScopeError as exc:
            say(str(exc), _WARN)
            return False
        except Exception as exc:                       # noqa: BLE001
            say(f"{type(exc).__name__}: {exc}", _BAD)
            return False
        refresh()
        return True

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

    def _release() -> None:
        """Drop the current connection before making another.

        Only one connection to the stand exists at a time, so replacing a
        Scope without closing the old one makes the new one fail — and the
        failure looks like broken hardware rather than a held handle.
        """
        old = state.get("scope")
        state["scope"] = None
        if old is not None:
            try:
                old.close()
            except Exception:
                pass

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
        show_channels()
        show_pixel_size()
        for warning in s.role_warnings:
            roles_div.text += f"<br><b style='color:#b36b00'>{warning}</b>"
        refresh()

    def do_connect() -> None:
        path = cfg_input.value.strip()
        if not path:
            say("give a .cfg path, or use Demo devices", _WARN)
            return
        if not Path(path).exists():
            say(f"no such file: {path}", _BAD)
            return
        _release()
        try:
            _after_connect(Scope.from_config(path), Path(path).name)
        except Exception as exc:                        # noqa: BLE001
            say(f"could not load configuration: {exc}", _BAD)

    def do_demo() -> None:
        _release()
        try:
            _after_connect(Scope.demo(), "demo devices")
        except Exception as exc:                        # noqa: BLE001
            say(f"demo devices unavailable: {exc}", _BAD)

    def do_build() -> None:
        """Write a .cfg for whatever is attached, then connect to it.

        Connecting every device takes seconds to minutes, and a Bokeh
        callback blocks the session while it runs — so paint the notice
        first and do the work on the next tick, or the user watches an
        unchanged page and assumes the button is broken.
        """
        out = cfg_input.value.strip() or "MMConfig.cfg"
        say("connecting every attached device — this can take a minute…",
            _WARN)
        doc.add_next_tick_callback(lambda: _build_now(out))

    def _build_now(out: str) -> None:
        # The build opens its own core and initialises the stand. If this
        # session is still connected, that second connection is refused and
        # the build reports a hub that would not initialise — so let go first.
        _release()
        core = None
        try:
            core = config_build.new_core()
            result = config_build.build(core)
        except Exception as exc:                        # noqa: BLE001
            say(f"build failed: {exc}", _BAD)
            return
        if not result.devices:
            say("; ".join(result.notes) or "nothing connected", _BAD)
            _unload(core)
            return
        # The target is pre-filled with the file this session connected to —
        # the one the Channels tab appends presets into, and the one someone
        # may have hand-edited. A degraded build (stand powered on after the
        # controller, so only the camera connects) still reaches here, so an
        # unconditional write can replace a working configuration with a
        # camera-only one and lose every channel preset with it.
        target = Path(out)
        if target.exists() and 0 not in overwrite_ck.active:
            say(f"{target.name} already exists — tick 'replace the existing "
                f"file' to overwrite it, or change the name. "
                f"({len(result.devices)} device(s) were found.)", _WARN)
            _unload(core)
            return
        backup = None
        try:
            if target.exists():
                backup = target.with_suffix(target.suffix + ".bak")
                backup.write_text(target.read_text())
            target.write_text(config_build.to_text(result, core,
                                                   preserve_from=out))
        except Exception as exc:                        # noqa: BLE001
            say(f"built, but could not write {out}: {exc}", _BAD)
            _unload(core)
            return
        cfg_input.value = out
        failed = (f" — {len(result.failures)} device(s) did not connect"
                  if result.failures else "")
        kept = len(config_build.preserved_lines(backup)) if backup else 0
        roles_div.text = (
            f"wrote <b>{out}</b>: {len(result.devices)} device(s){failed}"
            + (f"<br>previous file saved as {backup.name}" if backup else "")
            + (f"; carried {kept} preset/pixel-size line(s) over" if kept else "")
            + ("<br>did not connect: " +
               ", ".join(f"{n} ({w})" for n, w in result.failures[:6])
               if result.failures else ""))
        # The build's core still holds the hardware. Release it before
        # loading the file, or the reload hits the same one-connection limit
        # the build just escaped.
        _unload(core)
        try:
            _after_connect(Scope.from_config(out), Path(out).name)
        except Exception as exc:                        # noqa: BLE001
            say(f"wrote {out} but could not load it: {exc}", _BAD)

    def _unload(core) -> None:
        if core is None:
            return
        try:
            core.unloadAllDevices()
        except Exception:
            pass

    connect_btn.on_click(do_connect)
    demo_btn.on_click(do_demo)
    build_btn.on_click(do_build)

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
    # Z and the PFS offset are separate controls on purpose. A single
    # "Focus ±" that silently switched between them would be labelled in µm
    # while moving a device whose units are not µm — the offset is in the
    # offset device's own units, and one unit is not one micron.
    zup_btn.on_click(guard(lambda s: s.move_z_by(float(zstep_spin.value))))
    zdn_btn.on_click(guard(lambda s: s.move_z_by(-float(zstep_spin.value))))
    offup_btn.on_click(guard(
        lambda s: s.set_pfs_offset(s.pfs_offset() + float(offstep_spin.value))))
    offdn_btn.on_click(guard(
        lambda s: s.set_pfs_offset(s.pfs_offset() - float(offstep_spin.value))))
    offset_spin.on_change("value", guard_change(
        lambda s, new: s.set_pfs_offset(float(new))))

    def on_intensity(attr, old, new) -> None:
        if state["syncing"] or new == old:
            return
        run(lambda s: s.set_intensity(float(new)))

    intensity.on_change("value_throttled", on_intensity)

    def on_shutter(attr, old, new) -> None:
        if state["syncing"]:
            return
        run(lambda s: s.set_shutter(bool(new)))

    shutter_tog.on_change("active", on_shutter)
    autoshut_ck.on_change("active", lambda a, o, n:
                          None if state["syncing"] else
                          run(lambda s: s.set_auto_shutter(bool(n))))

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

    # ------------------------------------------------------------ channels

    def show_channels() -> None:
        s = scope()
        if s is None:
            return
        names = s.channels()
        sync(channel_sel, names, "options")
        sync(acq_channels, names, "options")
        sync(scan_chan, [""] + names, "options")
        chan_list.text = ("channels: " + ", ".join(f"<b>{n}</b>" for n in names)
                          if names else
                          "<i>no channels defined yet — set the light path, "
                          "filters and intensity by eye, then capture</i>")

    def do_capture_channel() -> None:
        s = scope()
        if s is None:
            say("not connected", _WARN)
            return
        name = chan_name.value.strip()
        if not channels_mod.valid_name(name):
            chan_div.text = _badge("give a name without commas", _WARN)
            return
        try:
            preset = channels_mod.capture(
                s, name, with_exposure=bool(chan_exposure_ck.active))
        except Exception as exc:                        # noqa: BLE001
            chan_div.text = _badge(f"capture failed: {exc}", _BAD)
            return
        if not preset.settings:
            chan_div.text = _badge(
                "nothing to capture — no shutters, filter turrets or light "
                "path in this configuration", _WARN)
            return
        # Define it in the running core so it works now, and write it to the
        # .cfg so it survives a restart.
        try:
            channels_mod.apply_to_core(s, preset)
        except Exception as exc:                        # noqa: BLE001
            chan_div.text = _badge(f"could not define it live: {exc}", _BAD)
            return
        target = cfg_input.value.strip()
        written = ""
        if target and Path(target).exists():
            try:
                channels_mod.append_to_config(target, preset)
                written = f" and saved to {Path(target).name}"
            except Exception as exc:                    # noqa: BLE001
                written = f" (but could not write the .cfg: {exc})"
        chan_div.text = (_badge(f"channel {name!r} defined{written}", _OK)
                         + f"<br><span style='font-size:11px'>"
                         f"{preset.describe()}</span>")
        show_channels()

    chan_capture.on_click(do_capture_channel)

    # ---------------------------------------------------------- throughput

    def do_measure() -> None:
        s = scope()
        if s is None:
            say("not connected", _WARN)
            return
        wanted = list(acq_channels.value) or None
        timing_div.text = ("<i>measuring — the stage will move and the camera "
                           "will snap a few times…</i>")
        doc.add_next_tick_callback(lambda: _measure_now(wanted))

    def _measure_now(wanted) -> None:
        s = scope()
        if s is None:
            return
        try:
            t = timing_mod.measure(s, channels=wanted)
        except Exception as exc:                        # noqa: BLE001
            timing_div.text = _badge(f"measurement failed: {exc}", _BAD)
            return
        state["timings"] = t
        timing_div.text = ("<b>measured</b><br><pre style='margin:4px 0;"
                           "font-size:12px'>"
                           + "\n".join(t.describe()) + "</pre>")
        recompute_budget()

    def recompute_budget() -> None:
        t = state.get("timings")
        if t is None:
            return
        chans = list(acq_channels.value) or [""]
        budget = timing_mod.plan(
            t, chans,
            interval_s=float(interval_spin.value) * 60.0,
            requested=int(npos_spin.value),
            move_um=float(movedist_spin.value),
            refocus=True,
            duration_h=float(duration_spin.value),
        )
        colour = _OK if budget.fits else _BAD
        lines = budget.describe()
        budget_div.text = (
            _badge("fits" if budget.fits else "does not fit", colour)
            + "<br>" + "<br>".join(lines)
            + "<br><span style='font-size:11px;color:#666'>20% of the "
              "interval is held back as slack — a timelapse scheduled to the "
              "full interval drifts later at every timepoint.</span>")

    measure_btn.on_click(do_measure)
    for w in (interval_spin, duration_spin, npos_spin, movedist_spin):
        w.on_change("value", lambda attr, old, new: recompute_budget())
    acq_channels.on_change("value", lambda attr, old, new: recompute_budget())

    # ---------------------------------------------------------------- scan

    def show_pixel_size() -> None:
        s = scope()
        if s is None:
            return
        px = s.pixel_size_um()
        if px > 0:
            try:
                w, h = s.fov_um()
                px_div.text = (f"<b>{px:g} µm/px</b> → field "
                               f"{w:.0f} × {h:.0f} µm")
            except ScopeError as exc:
                px_div.text = str(exc)
            sync(px_spin, px)
            return
        guess, how = s.suggest_pixel_size_um()
        if guess > 0:
            sync(px_spin, round(guess, 5))
        px_div.text = (
            _badge("no pixel size configured", _WARN)
            + f" — a scan grid needs one. Suggestion: <b>{guess:g}</b> µm/px "
              f"({how})" if guess > 0 else
            _badge("no pixel size configured", _WARN) + f" — {how}. "
            f"Measure it with a stage micrometer and enter it.")

    def do_set_pixel_size() -> None:
        if run(lambda s: s.set_pixel_size_um(float(px_spin.value))):
            show_pixel_size()

    px_set.on_click(do_set_pixel_size)

    def _well_um():
        try:
            return plate_mod.well_size_um(plate_sel.value)
        except Exception:
            return None

    def do_fit_grid() -> None:
        s = scope()
        if s is None:
            say("not connected", _WARN)
            return
        try:
            fov = s.fov_um()
        except ScopeError as exc:
            scan_div.text = _badge(str(exc), _WARN)
            return
        well_um = _well_um()
        if not well_um:
            scan_div.text = _badge("unknown plate type", _WARN)
            return
        rows, cols = scan_mod.fields_for_coverage(
            fov, well_um, float(cover_spin.value),
            float(overlap_spin.value) / 100.0)
        sync(scan_rows, rows)
        sync(scan_cols, cols)
        do_plan_scan()

    fit_btn.on_click(do_fit_grid)

    def do_plan_scan() -> None:
        s = scope()
        if s is None:
            say("not connected", _WARN)
            return
        cal = state["cal"]
        if cal is None:
            scan_div.text = _badge(
                "register the plate first — a scan needs to know where the "
                "wells are", _WARN)
            return
        sel = selection()
        if not sel.wells:
            scan_div.text = _badge(
                "select wells first, on the Plate tab", _WARN)
            return
        try:
            fov = s.fov_um()
        except ScopeError as exc:
            scan_div.text = _badge(str(exc), _WARN)
            return
        try:
            sp = scan_mod.plan(cal, sel, fov_um=fov,
                               rows=int(scan_rows.value),
                               columns=int(scan_cols.value),
                               overlap=float(overlap_spin.value) / 100.0)
        except ValueError as exc:
            scan_div.text = _badge(str(exc), _WARN)
            return
        state["scan_plan"] = sp
        lines = sp.describe(_well_um())
        t = state.get("timings")
        if t is not None:
            secs = scan_mod.estimate_seconds(
                sp, t, channels=[scan_chan.value or ""],
                refocus_every=int(refocus_spin.value))
            lines.append(f"<b>≈ {secs / 60:.1f} min</b> from the measured "
                         f"timings")
        else:
            lines.append("<i>measure the timings on the Throughput tab for a "
                         "time estimate</i>")
        scan_div.text = _badge("planned", _OK) + "<br>" + "<br>".join(lines)

    plan_btn.on_click(do_plan_scan)
    for w in (scan_rows, scan_cols, overlap_spin):
        w.on_change("value", lambda a, o, n: None)

    def do_run_scan() -> None:
        if state["scan_cb"] is not None:
            scan_div.text = _badge("a scan is already running", _WARN)
            return
        do_plan_scan()
        sp = state["scan_plan"]
        s = scope()
        if sp is None or s is None:
            return
        state["scan_stop"] = False
        try:
            state["scan_iter"] = scan_mod.run_iter(
                s, sp, scan_dir.value.strip() or "scan",
                channel=scan_chan.value or "",
                refocus_every=int(refocus_spin.value),
                should_stop=lambda: state["scan_stop"])
        except Exception as exc:                        # noqa: BLE001
            scan_div.text = _badge(f"could not start: {exc}", _BAD)
            return
        # Stepped from a callback rather than run in a loop: a 361-field scan
        # takes minutes, and a blocking callback would freeze the page for all
        # of it — including the Stop button.
        state["scan_cb"] = doc.add_periodic_callback(_scan_step, 20)
        scan_btn.disabled = True

    def _scan_step() -> None:
        it = state.get("scan_iter")
        if it is None:
            _scan_finished()
            return
        try:
            result = next(it)
        except StopIteration:
            _scan_finished()
            return
        except Exception as exc:                        # noqa: BLE001
            scan_div.text = _badge(f"scan failed: {exc}", _BAD)
            _scan_finished()
            return
        sp = state["scan_plan"]
        done = len(result.frames) + len(result.failures)
        total = sp.n_fields if sp else 0
        bar = "█" * int(24 * done / total) if total else ""
        scan_progress.text = (
            f"{bar:<24} {done}/{total} — {len(result.frames)} frame(s)"
            + (f", {len(result.failures)} failed" if result.failures else ""))

    def _scan_finished() -> None:
        if state["scan_cb"] is not None:
            try:
                doc.remove_periodic_callback(state["scan_cb"])
            except Exception:
                pass
        state["scan_cb"] = None
        state["scan_iter"] = None
        scan_btn.disabled = False
        out = Path(scan_dir.value.strip() or "scan")
        scan_div.text = (_badge("scan finished", _OK)
                         + f" manifest: {out / scan_mod.MANIFEST}")

    def do_stop_scan() -> None:
        state["scan_stop"] = True
        scan_div.text = _badge("stopping after the current field…", _WARN)

    scan_btn.on_click(do_run_scan)
    stop_btn.on_click(do_stop_scan)

    # --------------------------------------------------------------- plate

    def redraw_plate() -> None:
        cal = state["cal"]
        if cal is None:
            well_src.data = {k: [] for k in well_src.data}
            pmap.title.text = "plate not registered"
            # Wells can be chosen before the plate is registered — the names
            # come from the plate type, not from the calibration — so the
            # count must still update even with no map to draw.
            show_wells()
            return
        lay = plate_mod.layout(cal)
        captured = {r.name.upper() for r in state["refs"]}
        sel = selection()
        chosen = set(sel.wells) if sel else set()
        well_src.data = {
            "name": lay.names, "x": lay.x, "y": lay.y,
            "w": [lay.well_width_um] * len(lay.names),
            "h": [lay.well_height_um] * len(lay.names),
            # fill = selected for imaging; outline = measured for calibration
            "color": ["#2f7ed8" if n.upper() in chosen else "#e8e8e8"
                      for n in lay.names],
            "line": ["#ffb400" if n.upper() in captured else "#9a9a9a"
                     for n in lay.names],
            "lw": [3 if n.upper() in captured else 1 for n in lay.names],
        }
        pmap.title.text = (f"{cal.plate} · rotation {cal.rotation:+.3f}° · "
                           f"residual {cal.residual_um:.0f} µm · "
                           f"{len(chosen)} well(s) selected")
        show_wells()

    def selection():
        """The live selection, created (and re-created) for the current plate."""
        sel = state.get("sel")
        want = plate_sel.value
        if sel is None or sel.plate != want:
            # A plate-type change invalidates the well names entirely, so
            # carrying the old set over would silently keep wells that do not
            # exist on the new plate.
            sel = wells_mod.Selection(want)
            state["sel"] = sel
        return sel

    def show_wells() -> None:
        sel = selection()
        n = len(sel.wells)
        if not n:
            wells_div.text = ("<i>no wells selected — click wells on the map, "
                              "or type a range above</i>")
            return
        order = (sel.serpentine() if serp_ck.active else sel.ordered())
        shown = ", ".join(order[:16]) + (" …" if len(order) > 16 else "")
        wells_div.text = (f"<b>{n}</b> well(s), visiting order: {shown}")

    def _apply_wells(add: bool) -> None:
        sel = selection()
        text = wells_text.value.strip()
        if not text:
            wells_div.text = _badge("type a well range first", _WARN)
            return
        names = wells_mod.parse(sel.plate, text)
        if not names:
            wells_div.text = _badge(
                f"nothing in {text!r} matched a well on a {sel.plate} plate",
                _WARN)
            return
        sel.add(names) if add else sel.remove(names)
        redraw_plate()

    wells_add.on_click(lambda: _apply_wells(True))
    wells_sub.on_click(lambda: _apply_wells(False))
    wells_all.on_click(lambda: (selection().select_all(), redraw_plate()))
    wells_none.on_click(lambda: (selection().clear(), redraw_plate()))
    serp_ck.on_change("active", lambda a, o, n: show_wells())

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
        state["edges"] = {}
        show_edges()
        show_refs()

    # ---- wall touches, which is how a centre is really found -------------

    def show_edges() -> None:
        name = well_input.value.strip().upper()
        got = state["edges"].get(name, {})
        if not got:
            edge_div.text = (
                "<i>At 40× a well is not even visible — the field is ~333 µm "
                "and a 96-well well is 6400 µm. So do not aim for the centre: "
                "drive until the <b>wall</b> sits mid-image on one side, "
                "capture, then the opposite side. The centre is the midpoint, "
                "and the implied diameter checks itself.</i>")
            return
        edge_div.text = (f"<b>{name}</b> walls: "
                         + ", ".join(f"{k} {v:.0f}" for k, v in
                                     sorted(got.items())))

    def _capture_edge(which: str):
        def handler() -> None:
            s = scope()
            if s is None:
                say("not connected", _WARN)
                return
            name = well_input.value.strip().upper()
            if not name:
                edge_div.text = _badge("name the well first", _WARN)
                return
            try:
                here = s.xy()
            except Exception as exc:                    # noqa: BLE001
                say(f"could not read the stage: {exc}", _BAD)
                return
            value = here.x if which in ("left", "right") else here.y
            state["edges"].setdefault(name, {})[which] = value
            show_edges()
        return handler

    edge_x0.on_click(_capture_edge("left"))
    edge_x1.on_click(_capture_edge("right"))
    edge_y0.on_click(_capture_edge("bottom"))
    edge_y1.on_click(_capture_edge("top"))

    def do_centre_from_edges() -> None:
        name = well_input.value.strip().upper()
        got = state["edges"].get(name, {})
        if not got:
            edge_div.text = _badge("capture at least one wall in each axis "
                                   "first", _WARN)
            return
        try:
            centre = plate_mod.centre_from_edges(plate_sel.value, name, **got)
        except ValueError as exc:
            edge_div.text = _badge(str(exc), _WARN)
            return
        state["refs"] = [r for r in state["refs"]
                         if r.name.upper() != centre.name]
        state["refs"].append(centre.to_ref())
        colour = _OK if centre.trustworthy else _WARN
        edge_div.text = _badge("centre from walls", colour) + " " + \
            centre.describe()
        if not centre.trustworthy:
            edge_div.text += ("<br><span style='color:#b36b00'>that is far "
                              "from the plate's well size — a wall of the "
                              "wrong well, or the wrong plate type</span>")
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
        if cal.residual_um > plate_mod.GOOD_RESIDUAL_UM:
            notes.append(f"residual {cal.residual_um:.0f} µm exceeds the "
                         f"~{plate_mod.GOOD_RESIDUAL_UM:.0f} µm a well scan "
                         f"can absorb — re-check the well names")
        cal_div.text = (_badge("registered", _OK) + " " + cal.describe() +
                        ("<br><span style='color:#b36b00'>" +
                         "; ".join(notes) + "</span>" if notes else ""))
        redraw_plate()
        refresh()

    capture_btn.on_click(do_capture)
    clear_btn.on_click(do_clear)
    calib_btn.on_click(do_calibrate)
    edge_use.on_click(do_centre_from_edges)

    def do_save() -> None:
        if state["cal"] is None:
            cal_div.text = _badge("nothing to save — calibrate first", _WARN)
            return
        try:
            plate_mod.save(state["cal"], plate_file.value)
            sel = selection()
            if sel.wells:
                wells_mod.save_selection(sel, plate_file.value)
            cal_div.text = _badge(
                f"saved to {plate_file.value}"
                + (f" with {len(sel.wells)} well(s)" if sel.wells else ""), _OK)
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
            sync(plate_sel, cal.plate)
        state["sel"] = (wells_mod.load_selection(plate_file.value)
                        or wells_mod.Selection(cal.plate))
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
        well_src.selected.indices = []
        if click_mode.active == 0:
            selection().toggle(name)
            redraw_plate()
            return
        if run(lambda s: s.move_xy(wx, wy)):
            say(f"moved to {name}", _OK)

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
        if st.pfs_in_range is False:
            pfs_div.text += " " + _badge("out of range", _BAD)
        elif st.pfs_in_range is True:
            pfs_div.text += " " + _badge("in range", _DIM)
        if st.pfs_offset is not None:
            pfs_div.text += (f" <span style='font-family:monospace'>"
                             f"offset {st.pfs_offset:.1f}</span>")
            sync(offset_spin, st.pfs_offset)

        # "Why is my image black?" is answered here rather than left to be
        # guessed at: a config with no shutter cannot turn a light on at all.
        if not st.roles.get("shutter"):
            light_div.text = _badge("no light control in this config", _WARN) \
                + " <span style='font-size:11px'>a dark frame is expected</span>"
        elif st.auto_shutter:
            light_div.text = _badge("auto-shutter", _OK) + \
                " <span style='font-size:11px'>opened for each acquisition" \
                "</span>"
        elif st.shutter_open:
            light_div.text = _badge("shutter open", _OK)
        else:
            light_div.text = _badge("shutter CLOSED", _WARN)
        state["syncing"] = True
        try:
            shutter_tog.active = bool(st.shutter_open)
            autoshut_ck.active = [0] if st.auto_shutter else []
            # There is no MMCore API for brightness — it is a device property
            # whose name differs per vendor, so name the one being driven
            # rather than leave an unlabelled slider to be trusted blindly.
            info = st.intensity
            if info is None:
                intensity.disabled = True
                intensity_note.text = (
                    "no intensity property found on the lamp — list them with "
                    "<code>nikon-control-scope config &lt;cfg&gt; "
                    "--properties</code>")
            else:
                intensity.disabled = False
                if info.numeric:
                    intensity.start = info.lower
                    intensity.end = info.upper
                    intensity.step = max((info.upper - info.lower) / 100.0,
                                         0.01)
                if (n := info.number) is not None:
                    intensity.value = n
                intensity.title = f"Intensity — {info.device}.{info.name}"
                intensity_note.text = ""
        finally:
            state["syncing"] = False

        if st.pfs_locked:
            focus_note.text = (
                "PFS is holding, so <b>Z is not the focus control</b> — it "
                "moves and PFS pulls straight back. Use <b>Offset ±</b>: it "
                "is what sets where the focal plane sits relative to the "
                "coverslip. <i>A lock at the wrong offset focuses on the "
                "glass, not the cells — which looks blurry while PFS reports "
                "everything is fine.</i>")
        elif st.pfs_available and not st.pfs_engaged:
            focus_note.text = ("PFS is off: <b>Z ±</b> moves focus. Engage "
                               "PFS, then use <b>Offset ±</b> to bring the "
                               "cells sharp, and reuse that offset.")
        else:
            focus_note.text = "<b>Z ±</b> moves the focus drive."

        if st.objectives:
            sync(obj_sel, st.objectives, "options")
        if st.objective:
            sync(obj_sel, st.objective)
        if st.channels and list(channel_sel.options) != list(st.channels):
            sync(channel_sel, st.channels, "options")
            sync(acq_channels, st.channels, "options")
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
        Div(text="<h3 style='margin:0'>Connect</h3>"),
        row(cfg_input,
            column(Div(text="<br>"),
                   row(connect_btn, build_btn, demo_btn))),
        overwrite_ck,
        conn_div, roles_div,
    )

    drive_panel = column(
        pos_div,
        row(jog, column(step_spin, row(goto_x, goto_y), goto_btn)),
        row(zdn_btn, zup_btn, zstep_spin),
        row(offdn_btn, offup_btn, offstep_spin, offset_spin),
        focus_note,
        light_div,
        row(shutter_tog, autoshut_ck),
        intensity, intensity_note,
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
        row(plate_sel, well_input, capture_btn, clear_btn, calib_btn),
        suggest_div,
        row(edge_x0, edge_x1, edge_y0, edge_y1, edge_use),
        edge_div, refs_div, cal_div,
        row(plate_file, column(Div(text="<br>"), row(save_btn, load_btn))),
        Div(text="<b>Wells to image</b>", styles={"margin-top": "6px"}),
        row(wells_text, column(Div(text="<br>"),
                               row(wells_add, wells_sub, wells_all,
                                   wells_none))),
        row(click_mode, serp_ck),
        wells_div, pmap,
    )

    channel_panel = column(
        Div(text="<b>Define a channel</b><br>"
                 "<span style='font-size:12px;color:#666'>Set the light path, "
                 "filter turret, shutter and intensity by eye — then capture "
                 "what the microscope is doing. The objective, stage and "
                 "focus are never captured.</span>"),
        row(chan_name, chan_exposure_ck),
        chan_capture, chan_div, chan_list,
    )

    throughput_panel = column(
        Div(text="<b>How many positions fit in one timepoint?</b><br>"
                 "<span style='font-size:12px;color:#666'>Measured on this "
                 "microscope: stage settling, camera readout, PFS lock and "
                 "channel switching.</span>"),
        acq_channels,
        row(interval_spin, duration_spin),
        row(npos_spin, movedist_spin),
        measure_btn, timing_div, budget_div,
    )

    scan_panel = column(
        Div(text="<b>Scan the selected wells, field by field</b><br>"
                 "<span style='font-size:12px;color:#666'>Frames plus a "
                 "manifest recording every frame's stage position — which is "
                 "what lets a cell found offline be driven back to.</span>"),
        row(px_spin, column(Div(text="<br>"), px_set)),
        px_div,
        row(cover_spin, fit_btn),
        row(scan_rows, scan_cols, overlap_spin),
        refocus_spin,
        row(scan_chan, scan_dir),
        row(plan_btn, scan_btn, stop_btn),
        scan_div, scan_progress,
    )

    tabs = Tabs(tabs=[
        TabPanel(child=drive_panel, title="Drive"),
        TabPanel(child=plate_panel, title="Plate"),
        TabPanel(child=channel_panel, title="Channels"),
        TabPanel(child=scan_panel, title="Scan"),
        TabPanel(child=throughput_panel, title="Throughput"),
    ], width=560)

    def _on_session_end(session_context):
        """A closed tab (or a dropped RDP session) must not leave the lamp
        burning the specimen, nor keep the stand claimed against the next
        session's Connect."""
        state["scan_stop"] = True
        if state.get("scan_cb") is not None:
            try:
                doc.remove_periodic_callback(state["scan_cb"])
            except Exception:
                pass
            state["scan_cb"] = None
        if state.get("live_cb") is not None:
            try:
                doc.remove_periodic_callback(state["live_cb"])
            except Exception:
                pass
            state["live_cb"] = None
        _release()          # closes the shutter, then unloads the devices

    doc.on_session_destroyed(_on_session_end)

    doc.add_root(column(connect_panel, row(image_panel, tabs)))
    doc.title = "Nikon scope control"

    do_suggest()
    show_refs()
    show_edges()
    if plate_path and Path(plate_path).exists():
        plate_file.value = plate_path
        do_load()
