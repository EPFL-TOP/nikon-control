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
    found = [m for m in doc.select({}) if getattr(m, "title", None) == title]
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
    # the captured wells are marked differently from the rest
    assert len(set(wells.data["color"])) == 2

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
