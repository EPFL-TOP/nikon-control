"""Tests for capturing imaging channels as Micro-Manager config presets."""
import pytest

from nikon_control.scope import channels
from nikon_control.scope.control import Scope

from tests.test_scope_control import ROLES, FakeCore


class ChannelCore(FakeCore):
    """A fake with the devices that actually define a channel.

    Shaped after the lab's Ti2-E: a dia lamp, two epi turret shutters, two
    filter turrets, a light path — and a nosepiece, which must NOT be
    captured.
    """

    DEVICES = {
        "DiaLamp": "Shutter",
        "Turret1Shutter": "Shutter",
        "FilterTurret1": "State",
        "LightPath": "State",
        "Turret": "State",          # the nosepiece
        "XY": "XYStage",
        "Z": "Stage",
        "Cam": "Camera",
    }
    VALUES = {
        ("DiaLamp", "State"): "1",
        ("Turret1Shutter", "State"): "0",
        ("FilterTurret1", "Label"): "GFP-HQ",
        ("LightPath", "Label"): "L100",
        ("Turret", "Label"): "Plan Fluor 40x",
    }
    PROPS = {
        "DiaLamp": ["State", "Intensity"],
        "Turret1Shutter": ["State"],
        "FilterTurret1": ["Label", "State"],
        "LightPath": ["Label", "State"],
        "Turret": ["Label", "State"],
        "XY": [], "Z": [], "Cam": ["Exposure"],
    }

    def __init__(self):
        super().__init__()
        self.values = dict(self.VALUES)
        self.values[("DiaLamp", "Intensity")] = "42"

    def getLoadedDevices(self):
        return ["Core", *self.DEVICES]

    def getDeviceType(self, label):
        # MMCore reports types as ints; these are DeviceType's own values.
        return {"Camera": 2, "Shutter": 3, "State": 4, "Stage": 5,
                "XYStage": 6}[self.DEVICES[label]]

    def getDevicePropertyNames(self, label):
        return self.PROPS.get(label, [])

    def getProperty(self, label, prop):
        return self.values.get((label, prop), "0")

    def setProperty(self, label, prop, value):
        self.values[(label, prop)] = str(value)
        self.log.append(f"{label}.{prop}={value}")

    def isPropertyReadOnly(self, label, prop):
        return False

    def hasPropertyLimits(self, label, prop):
        return prop == "Intensity"

    def getPropertyLowerLimit(self, label, prop):
        return 0.0

    def getPropertyUpperLimit(self, label, prop):
        return 100.0

    def getAllowedPropertyValues(self, label, prop):
        return []


@pytest.fixture
def scope():
    roles = dict(ROLES)
    roles.update({"shutter": "DiaLamp", "nosepiece": "Turret",
                  "lightpath": "LightPath", "camera": "Cam"})
    return Scope(ChannelCore(), roles)


def test_capture_records_the_illumination_state(scope):
    preset = channels.capture(scope, "GFP")
    got = {(s.device, s.property): s.value for s in preset.settings}
    assert got[("DiaLamp", "State")] == "1"
    assert got[("FilterTurret1", "Label")] == "GFP-HQ"
    assert got[("LightPath", "Label")] == "L100"


def test_capture_never_includes_the_objective_turret(scope):
    """A channel that rotates the turret would crash objectives into plates."""
    preset = channels.capture(scope, "GFP")
    assert not any(s.device == "Turret" for s in preset.settings)


def test_capture_never_includes_the_stage_or_focus(scope):
    """Selecting a channel must not teleport the sample."""
    preset = channels.capture(scope, "GFP")
    devices = {s.device for s in preset.settings}
    assert "XY" not in devices and "Z" not in devices


def test_capture_includes_the_lamp_intensity(scope):
    preset = channels.capture(scope, "BF")
    assert ("DiaLamp", "Intensity", "42") in [
        (s.device, s.property, s.value) for s in preset.settings]


def test_label_is_preferred_over_a_bare_state_number(scope):
    """'GFP-HQ' survives someone re-ordering a turret; '3' does not."""
    preset = channels.capture(scope, "GFP")
    turret = [s for s in preset.settings if s.device == "FilterTurret1"]
    assert turret and turret[0].property == "Label"


def test_exposure_is_opt_in(scope):
    assert not any(s.property == "Exposure"
                   for s in channels.capture(scope, "GFP").settings)
    with_exp = channels.capture(scope, "GFP", with_exposure=True)
    assert any(s.property == "Exposure" for s in with_exp.settings)


# ------------------------------------------------------------- the file

def test_preset_lines_match_the_config_format():
    preset = channels.Preset("BF", [
        channels.Setting("DiaLamp", "State", "1"),
        channels.Setting("LightPath", "Label", "L100"),
    ])
    assert preset.lines("Channel") == [
        "ConfigGroup,Channel,BF,DiaLamp,State,1",
        "ConfigGroup,Channel,BF,LightPath,Label,L100",
    ]


def test_recapturing_a_preset_replaces_it(tmp_path):
    """Appending blindly leaves two conflicting definitions of one channel."""
    cfg = tmp_path / "MMConfig.cfg"
    cfg.write_text("Property,Core,Initialize,1\n")

    channels.append_to_config(
        cfg, channels.Preset("BF", [channels.Setting("DiaLamp", "State", "1")]))
    channels.append_to_config(
        cfg, channels.Preset("BF", [channels.Setting("DiaLamp", "State", "0")]))

    presets = channels.read_presets(cfg)
    assert list(presets) == ["BF"]
    assert presets["BF"].settings[0].value == "0"
    assert cfg.read_text().count("ConfigGroup,Channel,BF,DiaLamp") == 1
    # and the rest of the configuration survived
    assert "Property,Core,Initialize,1" in cfg.read_text()


def test_a_second_preset_is_added_alongside_the_first(tmp_path):
    cfg = tmp_path / "MMConfig.cfg"
    cfg.write_text("Property,Core,Initialize,1\n")
    channels.append_to_config(
        cfg, channels.Preset("BF", [channels.Setting("DiaLamp", "State", "1")]))
    channels.append_to_config(
        cfg, channels.Preset("GFP", [channels.Setting("DiaLamp", "State", "0")]))
    assert set(channels.read_presets(cfg)) == {"BF", "GFP"}


def test_preset_names_that_would_corrupt_the_file_are_refused():
    assert channels.valid_name("GFP")
    assert not channels.valid_name("GFP,bad")
    assert not channels.valid_name("")
    assert not channels.valid_name(" leading space")
