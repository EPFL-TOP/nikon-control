"""Tests for generating a Micro-Manager configuration.

The strongest test here is a round-trip: build a config from live devices,
write it, load it back into a *fresh* core, and drive it. That is the only
check that proves the generated file is valid — a config that merely looks
right fails at the microscope, which is the worst place to find out.
"""
import pytest

from nikon_control.scope import config_build as cb

pymmcore = pytest.importorskip("pymmcore_plus")

from nikon_control.scope import discover  # noqa: E402
from nikon_control.scope.control import Scope  # noqa: E402

needs_demo = pytest.mark.skipif(
    "DemoCamera" not in discover.available_adapters(),
    reason="Micro-Manager demo adapters not installed (run: mmcore install)",
)


# --------------------------------------------------------------- pure bits

def test_ti2_hub_name_becomes_a_usable_label():
    """The Ti2 names its hub '*Ti2-E__0: Nikon Ti2 microscope'."""
    assert cb.safe_label("*Ti2-E__0: Nikon Ti2 microscope") == \
        "Nikon Ti2 microscope"


def test_labels_never_contain_a_comma():
    """A comma in a label would silently corrupt the config file."""
    assert "," not in cb.safe_label("Weird, Device, Name")


def test_a_blank_name_still_yields_a_label():
    assert cb.safe_label("   ") == "Device"
    assert cb.safe_label("*") == "Device"


def test_the_dia_lamp_is_skipped_by_default():
    """It has crashed Micro-Manager, and nothing here drives it."""
    assert "TIDiaLamp" in cb.SKIP_DEVICES


def test_config_text_has_the_required_shape():
    result = cb.BuildResult(
        devices=[
            cb.ConfigDevice("Scope", "NikonTI", "TIScope", type="Hub"),
            cb.ConfigDevice("TIXYDrive", "NikonTI", "TIXYDrive",
                            parent="Scope", type="XYStage", role="xystage"),
            cb.ConfigDevice("Cam", "HamamatsuHam", "HamamatsuHam_DCAM",
                            type="Camera", role="camera"),
        ],
        roles={"xystage": "TIXYDrive", "camera": "Cam"},
        failures=[("TIDiaLamp", "skipped")],
    )
    text = cb.to_text(result)
    lines = text.splitlines()

    assert "Property,Core,Initialize,0" in lines
    assert "Property,Core,Initialize,1" in lines
    # devices are declared before initialisation, roles after
    assert (lines.index("Device,Scope,NikonTI,TIScope")
            < lines.index("Property,Core,Initialize,1")
            < lines.index("Property,Core,XYStage,TIXYDrive"))
    assert "Parent,TIXYDrive,Scope" in lines
    assert "Property,Core,Camera,Cam" in lines
    # a failure is recorded, but as a comment so the file still loads
    assert any(ln.startswith("#") and "TIDiaLamp" in ln for ln in lines)


def test_no_stand_reports_a_note_instead_of_raising():
    res = cb.build(stand_adapter="NoSuchAdapter")
    assert res.devices == []
    assert res.notes and "not installed" in res.notes[0]


# ------------------------------------------------------------- round-trip

@needs_demo
def test_build_write_reload_and_drive(tmp_path):
    """The real test: does the generated file actually work?"""
    from pymmcore_plus import CMMCorePlus

    core = CMMCorePlus()
    result = cb.build(core, stand_adapter="DemoCamera")
    assert result.devices, "nothing connected"
    assert result.hub == "DHub"

    out = tmp_path / "MMConfig_built.cfg"
    out.write_text(cb.to_text(result, core))

    scope = Scope.from_config(out)          # a fresh core, the file alone
    for role in ("camera", "xystage", "focus", "autofocus"):
        assert scope.has(role), f"{role} did not survive the round-trip"
    assert scope.snap().ndim == 2
    scope.move_xy(10, 20)
    assert round(scope.xy().x) == 10


@needs_demo
def test_roles_are_never_mapped_to_an_empty_name():
    """Regression: resolving on objects without .name produced role -> ''.

    An empty string is falsy but the key is present, so every 'do we have
    this role?' check downstream reads as yes and then fails at the device.
    """
    from pymmcore_plus import CMMCorePlus

    result = cb.build(CMMCorePlus(), stand_adapter="DemoCamera")
    assert result.roles
    assert all(v for v in result.roles.values())


@needs_demo
def test_peripherals_come_from_the_initialised_hub():
    """How a Ti2's device names are learned: ask the hub, not the adapter."""
    from pymmcore_plus import CMMCorePlus

    core = CMMCorePlus()
    result = cb.build(core, stand_adapter="DemoCamera")
    labels = {d.label for d in result.devices}
    assert "DXYStage" in labels and "DAutoFocus" in labels
    # every peripheral is parented to the hub, which is what makes a hub
    # configuration reload correctly
    assert all(d.parent == "DHub" for d in result.devices
               if d.type not in ("Hub",) and d.library == "DemoCamera")


@needs_demo
def test_a_skipped_device_is_reported_not_silently_dropped(tmp_path):
    from pymmcore_plus import CMMCorePlus

    result = cb.build(CMMCorePlus(), stand_adapter="DemoCamera",
                      skip={"DGalvo"})
    assert "DGalvo" not in {d.label for d in result.devices}
    assert any(name == "DGalvo" for name, _why in result.failures)
