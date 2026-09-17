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


def test_nothing_is_excluded_by_default():
    """The dia lamp IS the brightfield light source.

    An earlier version skipped TIDiaLamp because it has crashed
    Micro-Manager with some driver versions. On a brightfield rig that
    guarantees a config which can never turn the light on — the wrong
    trade. It is flagged instead, and `--skip` is the escape hatch.
    """
    assert cb.SKIP_DEVICES == set()
    assert "TIDiaLamp" in cb.RISKY_DEVICES
    assert "--skip" in cb.RISKY_DEVICES["TIDiaLamp"]


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


def test_a_stand_that_contributes_nothing_says_so():
    """Regression: a camera-only config with no explanation for it.

    The rig produced exactly this — NikonTi2 installed, failing to load, and
    a build that walked past in silence and wrote a config with no stage.
    """
    res = cb.build(stand_adapter="NoSuchAdapter")
    assert res.notes


@needs_demo
def test_autoshutter_is_only_claimed_when_a_shutter_exists():
    result = cb.BuildResult(
        devices=[cb.ConfigDevice("Cam", "PVCAM", "Camera-1", type="Camera")],
        roles={"camera": "Cam"},
    )
    assert "AutoShutter" not in cb.to_text(result)

    result.devices.append(
        cb.ConfigDevice("Sh", "NikonTi2", "Shutter", type="Shutter"))
    result.roles["shutter"] = "Sh"
    assert "Property,Core,AutoShutter,1" in cb.to_text(result)


def test_rebuilding_preserves_channel_presets(tmp_path):
    """Regression: re-running the build deleted every channel definition.

    They are written into the same file by `nikon-control-scope channel` and
    by the dashboard's Capture button, and the generator does not produce
    them — so an unconditional rewrite silently destroyed the illumination
    settings that define the experiment.
    """
    from nikon_control.scope import channels

    cfg = tmp_path / "MMConfig.cfg"
    result = cb.BuildResult(
        devices=[cb.ConfigDevice("Cam", "PVCAM", "Camera-1", type="Camera")],
        roles={"camera": "Cam"})
    cfg.write_text(cb.to_text(result))

    channels.append_to_config(
        cfg, channels.Preset("BF", [channels.Setting("DiaLamp", "State", "1")]))
    channels.append_to_config(
        cfg, channels.Preset("GFP", [channels.Setting("FilterTurret2", "Label",
                                                      "3-GF")]))
    assert set(channels.read_presets(cfg)) == {"BF", "GFP"}

    # rebuild over the same file
    cfg.write_text(cb.to_text(result, preserve_from=cfg))

    assert set(channels.read_presets(cfg)) == {"BF", "GFP"}, \
        "the rebuild destroyed the channel presets"


def test_preserved_lines_covers_what_the_generator_does_not_write(tmp_path):
    cfg = tmp_path / "MMConfig.cfg"
    cfg.write_text("\n".join([
        "Device,Cam,PVCAM,Camera-1",              # generated — not preserved
        "Property,Core,Camera,Cam",               # generated — not preserved
        "ConfigGroup,Channel,BF,DiaLamp,State,1",
        "ConfigPixelSize,40x,Nosepiece,Label,3-Plan Apo 40x",
        "PixelSize_um,40x,0.1625",
        "Delay,FilterTurret1,50",
        "FocusDirection,ZDrive,0",
    ]))
    kept = cb.preserved_lines(cfg)
    assert len(kept) == 5
    assert not any(ln.startswith("Device,") for ln in kept)
    assert not any(ln.startswith("Property,") for ln in kept)


def test_preserved_lines_on_a_missing_file_is_empty(tmp_path):
    assert cb.preserved_lines(tmp_path / "nope.cfg") == []
