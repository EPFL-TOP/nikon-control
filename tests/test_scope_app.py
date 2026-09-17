"""Integration tests for the ``/scope`` dashboard.

These build the real Bokeh document and fire the real handlers against
Micro-Manager's demo devices — no browser, no microscope. The point is to
catch the class of bug that unit tests miss: a widget wired to nothing, a
handler with the wrong signature, a readout that never updates.
"""
import pytest

pytest.importorskip("bokeh")
pytest.importorskip("pymmcore_plus")

from bokeh.document import Document  # noqa: E402

from nikon_control.scope import discover  # noqa: E402
from nikon_control.dashboard.scope_app import modify_doc  # noqa: E402

needs_demo = pytest.mark.skipif(
    "DemoCamera" not in discover.available_adapters(),
    reason="Micro-Manager demo adapters not installed (run: mmcore install)",
)


def widget(doc, name):
    found = [m for m in doc.select({}) if getattr(m, "name", None) == name]
    assert found, f"no widget named {name!r} in the document"
    return found[0]


def click(doc, name):
    """Fire a Button's handler the way the browser would."""
    press(widget(doc, name))


def press(btn):
    handlers = btn._event_callbacks.get("button_click", [])
    assert handlers, f"button {btn.label!r} is wired to nothing"
    for h in handlers:
        h()


def by_label(doc, label):
    found = [m for m in doc.select({}) if getattr(m, "label", None) == label]
    assert found, f"no widget labelled {label!r}"
    return found[0]


def by_title(doc, title):
    """A widget by its title — skipping TabPanel, which also has one."""
    found = [m for m in doc.select({})
             if getattr(m, "title", None) == title
             and type(m).__name__ != "TabPanel"]
    assert found, f"no widget titled {title!r}"
    return found[0]


def goto(doc, x, y):
    """Drive the stage the way a user does: type X/Y, press Go to XY."""
    by_title(doc, "X (µm)").value = x
    by_title(doc, "Y (µm)").value = y
    press(by_label(doc, "Go to XY"))


@pytest.fixture
def doc():
    d = Document()
    modify_doc(d)
    return d


def test_document_builds_without_hardware(doc):
    """The page must render before anything is connected."""
    assert doc.roots
    assert "not connected" in widget(doc, "status").text


def test_actions_before_connecting_say_so_instead_of_raising(doc):
    click(doc, "up")
    assert "not connected" in widget(doc, "status").text
    click(doc, "snap")
    assert "not connected" in widget(doc, "status").text


@needs_demo
def test_connect_to_demo_then_snap_and_jog(doc):
    click(doc, "demo")
    assert "connected" in widget(doc, "status").text

    click(doc, "snap")
    imgs = [m for m in doc.select({}) if "image" in getattr(m, "data", {})]
    assert imgs, "no image source in the document"
    assert imgs[0].data["image"][0].shape[0] > 1, "snap produced no pixels"

    before = widget(doc, "position").text
    click(doc, "right")
    after = widget(doc, "position").text
    assert before != after, "the stage readout did not follow the jog"


@needs_demo
def test_pfs_button_updates_the_badge(doc):
    click(doc, "demo")
    assert "PFS off" in widget(doc, "pfs").text
    click(doc, "pfs_on")
    assert "PFS locked" in widget(doc, "pfs").text


@needs_demo
def test_register_a_plate_from_captured_stage_positions(doc):
    """The whole point of the plate panel, end to end."""
    click(doc, "demo")

    # Drive to three wells of a 96-well plate and capture each. The demo XY
    # stage is a real MMCore stage, so these are genuine round-trips — and
    # A1 to A12 is ~100 mm, which is how the 5 s default device timeout got
    # found.
    well_box = widget(doc, "well")
    for name, x, y in [("A1", 1000.0, 2000.0),
                       ("A12", 100000.0, 2000.0),
                       ("H1", 1000.0, -61000.0)]:
        goto(doc, x, y)
        well_box.value = name
        click(doc, "capture")

    click(doc, "calibrate")

    text = widget(doc, "calibration").text
    assert "registered" in text
    assert "96-well" in text

    wells = widget(doc, "wells")
    assert len(wells.data["name"]) == 96, "the plate map was not drawn"
    assert "A1" in wells.data["name"]
    # the three measured wells are outlined differently from the rest;
    # fill colour is reserved for the imaging selection
    marked = [n for n, lw in zip(wells.data["name"], wells.data["lw"]) if lw > 1]
    assert sorted(marked) == ["A1", "A12", "H1"]

    here = widget(doc, "here")
    assert here.data["x"], "the current position is not on the map"


@needs_demo
def test_a_bad_well_name_is_refused_and_reported(doc):
    click(doc, "demo")
    widget(doc, "well").value = "Z99"
    click(doc, "capture")
    click(doc, "calibrate")
    text = widget(doc, "calibration").text
    assert "refused" in text
    assert len(widget(doc, "wells").data["name"]) == 0


@needs_demo
def test_the_poll_does_not_look_like_a_user_edit(doc):
    """Regression: showing the current objective must not request a move.

    The poll writes hardware readings into the widgets. If those writes fire
    the same handlers a click does, connecting immediately trips the turret
    guard — and the poll then fights every value the user sets.
    """
    click(doc, "demo")
    assert "connected" in widget(doc, "status").text
    assert "turret" not in widget(doc, "status").text

    obj = by_title(doc, "Objective")
    assert obj.value, "the objective readout never populated"
    chosen = obj.value

    for _ in range(3):                      # several poll cycles
        refreshers = [cb.callback for cb in doc.session_callbacks]
        for fn in refreshers:
            fn()
    assert obj.value == chosen, "the poll changed the objective by itself"
    assert "turret" not in widget(doc, "status").text


@needs_demo
def test_a_deliberate_turret_move_still_needs_the_checkbox(doc):
    click(doc, "demo")
    obj = by_title(doc, "Objective")
    other = [o for o in obj.options if o != obj.value][0]

    obj.value = other                        # user picks, checkbox unticked
    assert "allow turret move" in widget(doc, "status").text
    assert obj.value != other, "the change should have been reverted"

    by_title(doc, "Objective")               # now tick the box and retry
    boxes = [m for m in doc.select({})
             if getattr(m, "labels", None) == ["allow turret move"]]
    boxes[0].active = [0]
    obj.value = other
    assert obj.value == other


@needs_demo
def test_build_button_writes_a_config_and_connects(tmp_path, doc, monkeypatch):
    """The 'not sure how to build the .cfg' path, end to end.

    The build runs on a next-tick callback so the browser sees the notice
    first, so the test has to drain that queue the way the server would.
    """
    out = tmp_path / "MMConfig_built.cfg"
    by_title(doc, "Micro-Manager configuration (.cfg)").value = str(out)

    # Only the demo adapter is installed here, so aim the build at it.
    from nikon_control.scope import config_build

    real_build = config_build.build
    monkeypatch.setattr(
        config_build, "build",
        lambda core=None, **kw: real_build(core, stand_adapter="DemoCamera"))

    click(doc, "build")
    assert "can take a minute" in widget(doc, "status").text

    for cb in list(doc.callbacks.session_callbacks):
        if type(cb).__name__ == "NextTickCallback":
            cb.callback()

    assert out.exists(), "no configuration was written"
    text = out.read_text()
    assert "Property,Core,Initialize,1" in text
    assert "connected" in widget(doc, "status").text
    assert "DXYStage" in widget(doc, "status").text or "DXYStage" in text

    # and the dashboard is now driving through that file
    click(doc, "snap")
    imgs = [m for m in doc.select({}) if "image" in getattr(m, "data", {})]
    assert imgs[0].data["image"][0].shape[0] > 1


@needs_demo
def test_light_panel_explains_a_dark_frame(doc, monkeypatch):
    """The rig's question: 'I can snap, but is the light on?'"""
    click(doc, "demo")
    # the demo config has a shutter, so the panel reports its actual state
    assert "shutter" in widget(doc, "light").text.lower() or \
           "auto-shutter" in widget(doc, "light").text.lower()


def test_light_panel_names_the_missing_device_when_there_is_none(doc,
                                                                 monkeypatch):
    """A camera-only config — exactly what the rig's build produced."""
    from nikon_control.dashboard import scope_app
    from nikon_control.scope.control import Scope

    from tests.test_scope_control import ROLES, FakeCore

    roles = {k: v for k, v in ROLES.items() if k != "shutter"}
    monkeypatch.setattr(Scope, "demo",
                        classmethod(lambda cls: Scope(FakeCore(), roles)))
    click(doc, "demo")

    text = widget(doc, "light").text
    assert "no light control" in text
    assert "dark frame is expected" in text


def _drain_next_ticks(doc):
    for cb in list(doc.callbacks.session_callbacks):
        if type(cb).__name__ == "NextTickCallback":
            cb.callback()


@needs_demo
def test_define_a_channel_from_the_gui(doc, tmp_path):
    """The user should not have to run a CLI command to get a channel."""
    cfg = tmp_path / "MMConfig.cfg"
    cfg.write_text("Property,Core,Initialize,1\n")
    by_title(doc, "Micro-Manager configuration (.cfg)").value = str(cfg)
    click(doc, "demo")

    widget(doc, "channel_name").value = "MyBF"
    click(doc, "capture_channel")

    assert "defined" in widget(doc, "channel_status").text
    # live in the running core...
    assert "MyBF" in by_title(doc, "Channel").options
    # ...and persisted, so it survives a restart
    assert "ConfigGroup,Channel,MyBF" in cfg.read_text()


@needs_demo
def test_a_channel_name_with_a_comma_is_refused(doc):
    click(doc, "demo")
    widget(doc, "channel_name").value = "bad,name"
    click(doc, "capture_channel")
    assert "without commas" in widget(doc, "channel_status").text


@needs_demo
def test_throughput_measures_and_answers_the_question(doc):
    """'How many cells can we image in a 5 min interval?'"""
    click(doc, "demo")
    acq = widget(doc, "acq_channels")
    acq.value = list(acq.options)[:3]

    click(doc, "measure")
    assert "measuring" in widget(doc, "timing").text
    _drain_next_ticks(doc)

    assert "measured" in widget(doc, "timing").text
    budget = widget(doc, "budget").text
    assert "positions fit" in budget
    assert "per position" in budget
    assert "timepoints over" in budget


@needs_demo
def test_changing_the_interval_updates_the_budget_without_remeasuring(doc):
    click(doc, "demo")
    click(doc, "measure")
    _drain_next_ticks(doc)
    before = widget(doc, "budget").text

    by_title(doc, "Interval (min)").value = 60

    after = widget(doc, "budget").text
    assert after != before, "the budget did not follow the interval"
    assert "60 min interval" in after


@needs_demo
def test_wells_can_be_selected_by_clicking_and_by_typing(doc, tmp_path):
    """Plugin 2 through the GUI: pick wells, keep them, save them."""
    click(doc, "demo")
    # register a plate so the map exists
    well_box = widget(doc, "well")
    for name, x, y in [("A1", 1000.0, 2000.0),
                       ("A12", 100000.0, 2000.0),
                       ("H1", 1000.0, -61000.0)]:
        goto(doc, x, y)
        well_box.value = name
        click(doc, "capture")
    click(doc, "calibrate")

    src = widget(doc, "wells")
    assert len(src.data["name"]) == 96
    assert set(src.data["color"]) == {"#e8e8e8"}, "nothing should start selected"

    # type a range
    widget(doc, "wells_text").value = "A1:B3"
    click(doc, "wells_add")
    assert "6</b> well(s)" in widget(doc, "wells_status").text
    chosen = [n for n, c in zip(src.data["name"], src.data["color"])
              if c == "#2f7ed8"]
    assert sorted(chosen) == ["A1", "A2", "A3", "B1", "B2", "B3"]

    # click a well on the map (mode defaults to select)
    src.selected.indices = [src.data["name"].index("D6")]
    assert "D6" in [n for n, c in zip(src.data["name"], src.data["color"])
                    if c == "#2f7ed8"]
    assert src.selected.indices == [], "the tap selection should be cleared"

    # clicking it again deselects
    src.selected.indices = [src.data["name"].index("D6")]
    assert "D6" not in [n for n, c in zip(src.data["name"], src.data["color"])
                        if c == "#2f7ed8"]


@needs_demo
def test_a_nonsense_well_range_is_reported_not_silently_ignored(doc):
    click(doc, "demo")
    widget(doc, "wells_text").value = "Q99"
    click(doc, "wells_add")
    assert "matched a well" in widget(doc, "wells_status").text


@needs_demo
def test_changing_plate_type_drops_wells_that_no_longer_exist(doc):
    """H12 is not on a 6-well plate; carrying it over would break a scan."""
    click(doc, "demo")
    widget(doc, "wells_text").value = "H12"
    click(doc, "wells_add")
    assert "1</b> well(s)" in widget(doc, "wells_status").text

    widget(doc, "plate_type").value = "6-well"
    click(doc, "wells_none")          # touches the selection for the new plate
    assert "no wells selected" in widget(doc, "wells_status").text


@needs_demo
def test_build_refuses_to_replace_an_existing_config_without_consent(doc, tmp_path):
    """Regression: the Build button truncated the connected .cfg.

    The target is pre-filled with the file the session connected to — the
    same file the Channels tab appends presets into — and a degraded build
    (only the camera connects) still reached the write.
    """
    cfg = tmp_path / "MMConfig.cfg"
    cfg.write_text("Property,Core,Initialize,1\n"
                   "ConfigGroup,Channel,BF,DiaLamp,State,1\n")
    original = cfg.read_text()
    by_title(doc, "Micro-Manager configuration (.cfg)").value = str(cfg)
    click(doc, "demo")

    from nikon_control.scope import config_build
    real_build = config_build.build
    monkey = lambda core=None, **kw: real_build(core, stand_adapter="DemoCamera")
    config_build.build, saved = monkey, config_build.build
    try:
        click(doc, "build")
        _drain_next_ticks(doc)
        assert cfg.read_text() == original, "the config was overwritten"
        assert "tick 'replace" in widget(doc, "status").text

        # consent, and it writes — keeping a backup and the channel preset
        overwrite = widget(doc, "overwrite")
        overwrite.active = [0]
        click(doc, "build")
        _drain_next_ticks(doc)
    finally:
        config_build.build = saved

    assert cfg.read_text() != original
    assert (tmp_path / "MMConfig.cfg.bak").exists(), "no backup was kept"
    assert "ConfigGroup,Channel,BF" in cfg.read_text(), \
        "the rebuild destroyed the channel preset"


@needs_demo
def test_a_failed_well_move_is_not_reported_as_success(doc):
    """Regression: on_well_tap printed 'moved to X' over run()'s failure."""
    click(doc, "demo")
    well_box = widget(doc, "well")
    for name, x, y in [("A1", 1000.0, 2000.0),
                       ("A12", 100000.0, 2000.0),
                       ("H1", 1000.0, -61000.0)]:
        goto(doc, x, y)
        well_box.value = name
        click(doc, "capture")
    click(doc, "calibrate")

    # arm "click = drive", then break the stage
    widget(doc, "click_mode").active = 1
    src = widget(doc, "wells")

    import nikon_control.scope.control as control_mod
    real = control_mod.Scope.move_xy

    def boom(self, *a, **kw):
        raise control_mod.ScopeError("stage not responding")

    control_mod.Scope.move_xy = boom
    try:
        src.selected.indices = [src.data["name"].index("B2")]
    finally:
        control_mod.Scope.move_xy = real

    text = widget(doc, "status").text
    assert "moved to B2" not in text
    assert "stage not responding" in text
