"""Tests for the microscope control facade.

Two layers. The first uses a **fake core** that records what was asked of it:
that is where the PFS interlock is pinned down, because the demo devices have
no PFS offset and real hardware cannot be in a test suite. The second runs
against Micro-Manager's demo devices when they are installed, to check the
facade against a genuine MMCore.
"""
import pytest

from nikon_control.scope.control import MAX_JOG_UM, Position, Scope, ScopeError

ROLES = {
    "camera": "Cam", "xystage": "XY", "focus": "Z",
    "autofocus": "PFS", "pfsoffset": "PFSOffset", "nosepiece": "Turret",
    "shutter": "Shutter",
}


class FakeCore:
    """Enough MMCore to drive Scope, with a log of what happened."""

    def __init__(self):
        self.x = self.y = 0.0
        self.pos = {"Z": 0.0, "PFSOffset": 0.0}
        self.continuous = False
        self.locked = False
        self.objective = "40x"
        self.exposure = 10.0
        self.shutter = False
        self.auto = False
        self.log: list[str] = []

    # stage
    def getXPosition(self, dev): return self.x
    def getYPosition(self, dev): return self.y

    def setXYPosition(self, dev, x, y):
        self.x, self.y = x, y
        self.log.append(f"xy={x:.0f},{y:.0f}")

    # focus / offset
    def getPosition(self, dev): return self.pos[dev]

    def setPosition(self, dev, value):
        self.pos[dev] = value
        self.log.append(f"{dev}={value:g}")
        if dev == "Z":
            # The hardware behaviour this whole module exists to handle.
            self.continuous = False
            self.locked = False

    # PFS
    def isContinuousFocusEnabled(self): return self.continuous
    def isContinuousFocusLocked(self): return self.locked

    def enableContinuousFocus(self, on):
        self.continuous = bool(on)
        self.locked = bool(on)
        self.log.append(f"pfs={'on' if on else 'off'}")

    # turret
    def getStateLabels(self, dev): return ["10x", "40x"]
    def getStateLabel(self, dev): return self.objective

    def setStateLabel(self, dev, label):
        self.objective = label
        self.log.append(f"objective={label}")

    # shutter
    def getShutterOpen(self, dev): return self.shutter
    def setShutterOpen(self, dev, on):
        self.shutter = bool(on)
        self.log.append(f"shutter={'open' if on else 'closed'}")
    def getAutoShutter(self): return self.auto
    def setAutoShutter(self, on): self.auto = bool(on)

    # camera / misc
    def getExposure(self): return self.exposure
    def setExposure(self, ms): self.exposure = ms
    def snapImage(self): self.log.append("snap")
    def getImage(self): return [[0]]
    def waitForDevice(self, dev): pass
    def waitForConfig(self, group, cfg): pass
    def getAvailableConfigGroups(self): return ["Channel"]
    def getAvailableConfigs(self, group): return ["BF", "GFP"]
    def getCurrentConfig(self, group): return "BF"
    def setConfig(self, group, cfg): self.log.append(f"config={cfg}")


@pytest.fixture
def scope():
    return Scope(FakeCore(), ROLES)


# ------------------------------------------------------------------ the trap

def test_moving_z_with_pfs_on_leaves_pfs_on(scope):
    """Setting Z silently kills PFS on real hardware; we must restore it."""
    scope.engage_pfs(timeout_s=0)
    assert scope.pfs_engaged()

    scope.move_z(50)

    assert scope.pfs_engaged(), "PFS was left off after a Z move"
    assert scope.z() == 50
    # and it was suspended deliberately, not left to be clobbered
    assert scope.core.log == ["pfs=on", "pfs=off", "Z=50", "pfs=on"]


def test_moving_z_with_pfs_off_does_not_switch_it_on(scope):
    scope.move_z(20)
    assert not scope.pfs_engaged()
    assert "pfs=on" not in scope.core.log


def test_keep_pfs_false_lets_the_move_disable_it(scope):
    """An explicit opt-out, for a caller that is about to re-focus anyway."""
    scope.engage_pfs(timeout_s=0)
    scope.move_z(50, keep_pfs=False)
    assert not scope.pfs_engaged()


def test_focus_by_uses_the_offset_while_locked_not_the_z_drive(scope):
    """With PFS holding, Z is the wrong control — the offset is the right one."""
    scope.engage_pfs(timeout_s=0)
    scope.core.log.clear()

    scope.focus_by(3)

    assert scope.pfs_offset() == 3
    assert scope.z() == 0                      # Z untouched
    assert scope.core.log == ["PFSOffset=3"]


def test_focus_by_uses_z_when_pfs_is_off(scope):
    scope.focus_by(3)
    assert scope.z() == 3
    assert scope.pfs_offset() == 0


def test_focus_by_falls_back_to_z_when_there_is_no_offset_device():
    roles = dict(ROLES)
    del roles["pfsoffset"]
    s = Scope(FakeCore(), roles)
    s.engage_pfs(timeout_s=0)
    s.focus_by(3)
    assert s.z() == 3


# ----------------------------------------------------------------- guardrails

def test_a_huge_jog_is_refused(scope):
    with pytest.raises(ScopeError, match="guard"):
        scope.move_xy_by(MAX_JOG_UM + 1, 0)
    assert scope.xy() == Position(0, 0)          # nothing moved


def test_a_huge_jog_can_be_forced(scope):
    scope.move_xy_by(MAX_JOG_UM + 1, 0, force=True)
    assert scope.xy().x == MAX_JOG_UM + 1


def test_an_absolute_move_is_never_guarded(scope):
    """The guard is against a mistyped jog, not against deliberate travel."""
    scope.move_xy(90000, -60000)
    assert scope.xy() == Position(90000, -60000)


def test_rotating_the_turret_needs_confirmation(scope):
    with pytest.raises(ScopeError, match="confirm=True"):
        scope.set_objective("10x")
    assert scope.objective() == "40x"
    assert scope.set_objective("10x", confirm=True) == "10x"


def test_a_missing_role_names_itself_and_what_is_there():
    s = Scope(FakeCore(), {"camera": "Cam"})
    with pytest.raises(ScopeError) as err:
        s.xy()
    assert "'xystage'" in str(err.value)
    assert "nikon-control-scope config" in str(err.value)   # how to fix it


# ---------------------------------------------------------------- the snapshot

def test_state_survives_one_broken_device(scope):
    """A dashboard polls this; one bad read must not blank the panel."""
    def boom(*a):
        raise RuntimeError("stage not responding")
    scope.core.getXPosition = boom

    st = scope.state()

    assert st.xy is None
    assert "stage not responding" in st.error
    assert st.z == 0.0                 # the rest still populated
    assert st.objectives == ["10x", "40x"]


def test_state_reports_pfs_engaged_and_locked_separately(scope):
    st = scope.state()
    assert st.pfs_available and not st.pfs_engaged
    scope.engage_pfs(timeout_s=0)
    st = scope.state()
    assert st.pfs_engaged and st.pfs_locked


def test_describe_is_readable(scope):
    scope.move_xy(120, -30)
    text = scope.describe()
    assert "XY (120, -30) µm" in text
    assert "PFS off" in text
    assert "40x" in text


# ------------------------------------------------------- against a real MMCore

pymmcore = pytest.importorskip("pymmcore_plus")
from nikon_control.scope import discover  # noqa: E402

needs_demo = pytest.mark.skipif(
    "DemoCamera" not in discover.available_adapters(),
    reason="Micro-Manager demo adapters not installed (run: mmcore install)",
)


@needs_demo
def test_demo_scope_resolves_roles_and_moves():
    s = Scope.demo()
    assert s.has("xystage") and s.has("focus") and s.has("camera")

    s.move_xy(0, 0)
    s.move_xy_by(100, -50)
    here = s.xy()
    assert round(here.x) == 100 and round(here.y) == -50

    z0 = s.z()
    assert s.move_z_by(5) == pytest.approx(z0 + 5)

    img = s.snap()
    assert img.shape and img.ndim == 2


@needs_demo
def test_demo_scope_state_and_channels():
    s = Scope.demo()
    st = s.state()
    assert st.channels                     # demo config defines a Channel group
    assert st.channel in st.channels
    assert st.objectives                   # demo nosepiece
    assert st.exposure_ms is not None
    assert not st.error


# ------------------------------------------------------- illumination

def test_no_shutter_says_a_dark_frame_is_expected():
    """The rig's actual state: a camera-only config returning black frames.

    With nothing that can switch a light on, 'the camera is broken' is the
    wrong conclusion and the one a user reaches first.
    """
    s = Scope(FakeCore(), {"camera": "Cam"})
    text = s.illumination()
    assert "no shutter or lamp" in text
    assert "dark frame is expected" in text
    assert s.state().illumination == text


def test_shutter_state_is_reported_and_settable():
    core = FakeCore()
    core.shutter = False
    core.auto = False
    s = Scope(core, ROLES)
    assert "CLOSED" in s.illumination()
    s.set_shutter(True)
    assert s.illumination() == "shutter open"


def test_auto_shutter_takes_precedence_in_the_explanation():
    core = FakeCore()
    core.auto = True
    s = Scope(core, ROLES)
    assert "auto-shutter" in s.illumination()


# ------------------------------------------------------- device properties

class PropCore(FakeCore):
    """A core whose lamp has an intensity property, and a camera that also
    has a 'brightness' — the false positive a loose search would pick."""

    def getLoadedDevices(self):
        return ["Core", "DiaLamp", "Cam", "XY"]

    def getDevicePropertyNames(self, label):
        return {"DiaLamp": ["State", "Intensity"],
                "Cam": ["Exposure", "BeadBrightness"],
                "XY": []}.get(label, [])

    def getProperty(self, label, prop):
        return {"Intensity": "37.5", "BeadBrightness": "1.0",
                "State": "1", "Exposure": "10"}.get(prop, "0")

    def setProperty(self, label, prop, value):
        self.log.append(f"{label}.{prop}={value}")

    def isPropertyReadOnly(self, label, prop):
        return False

    def hasPropertyLimits(self, label, prop):
        return prop in ("Intensity", "BeadBrightness")

    def getPropertyLowerLimit(self, label, prop):
        return 0.0

    def getPropertyUpperLimit(self, label, prop):
        return 100.0

    def getAllowedPropertyValues(self, label, prop):
        return []


@pytest.fixture
def prop_scope():
    roles = dict(ROLES)
    roles.update({"shutter": "DiaLamp", "camera": "Cam"})
    return Scope(PropCore(), roles)


def test_properties_can_be_reached_by_role_or_by_label(prop_scope):
    by_role = {p.name for p in prop_scope.properties("shutter")}
    by_label = {p.name for p in prop_scope.properties("DiaLamp")}
    assert by_role == by_label == {"State", "Intensity"}


def test_property_limits_are_reported(prop_scope):
    info = next(p for p in prop_scope.properties("DiaLamp")
                if p.name == "Intensity")
    assert info.numeric and info.lower == 0.0 and info.upper == 100.0
    assert info.number == 37.5
    assert "DiaLamp.Intensity = 37.5" in info.describe()


def test_intensity_is_found_on_the_lamp_not_the_camera(prop_scope):
    """A camera's 'BeadBrightness' matches the same word and is not a lamp."""
    info = prop_scope.intensity_property()
    assert info is not None
    assert info.device == "DiaLamp" and info.name == "Intensity"


def test_setting_intensity_is_clamped_to_the_device_limits(prop_scope):
    prop_scope.set_intensity(500)
    assert "DiaLamp.Intensity=100.0" in prop_scope.core.log


def test_no_intensity_anywhere_raises_with_a_way_forward():
    s = Scope(FakeCore(), {"camera": "Cam"})
    assert s.intensity_property() is None
    with pytest.raises(ScopeError, match="--properties"):
        s.set_intensity(50)
