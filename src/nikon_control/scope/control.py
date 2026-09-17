"""Operate the microscope: stage, focus, PFS, objective, camera.

One class, :class:`Scope`, wrapping a Micro-Manager core. Everything is
addressed by **role** — ``xystage``, ``focus``, ``autofocus``, ``pfsoffset``,
``nosepiece`` — never by device name, so the same code drives the Ti2-E, the
older Ti-E, and Micro-Manager's demo devices. :mod:`.stand` works out which
device fills which role.

Two pieces of microscope behaviour are encoded here rather than left to the
caller, because getting either wrong ruins an overnight run:

**Moving Z turns PFS off.** Setting the focus position programmatically
disables continuous focus (micro-manager#1815). So :meth:`Scope.move_z`
suspends PFS deliberately and re-engages it afterwards, instead of leaving it
silently off.

**Nothing is illuminated unless something opens the shutter.** A camera with
no shutter device in the configuration returns a black frame and no error,
which reads as a broken camera. :meth:`Scope.illumination` reports what light
control exists at all, so "the config has no light source" is distinguishable
from "the lamp is off".

**With PFS on, you do not change focus with Z.** PFS holds a fixed distance
from the coverslip; move Z and it simply pulls back. The control that means
"focus a bit higher" while locked is the **PFS offset**. :meth:`Scope.focus_by`
dispatches to whichever is right, which is almost always what a caller wants.

Nothing here reads or writes files, and nothing imports bokeh — the dashboard
is a view onto this, and a script is an equally good one.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from .stand import ROLES, resolve_roles, role_excluded

# A jog larger than this is refused unless the caller says it means it. The
# stage travels ~10 cm; a mistyped step has crashed objectives into plates.
MAX_JOG_UM = 5000.0
# Wait for PFS to report a lock, then give up rather than block a UI forever.
PFS_LOCK_TIMEOUT_S = 5.0
# Property names that mean "how bright", across vendors. The Ti2's dia lamp
# exposes its intensity as a device property, not as anything MMCore has an
# API for, so it has to be found by name.
# Config groups that mean "which channel". Nothing else is treated as one.
CHANNEL_GROUP_NAMES = ("Channel", "Channels", "channel")

INTENSITY_HINTS = ("intensity", "brightness", "power", "level", "voltage")
# …but only looked for on devices that could plausibly BE a light source.
# Sweeping every device finds things like a demo camera's "BeadBrightness"
# and would hand the user a slider that silently does nothing useful.
LIGHT_LABEL_HINTS = ("lamp", "light", "led", "dia", "epi", "illum", "shutter")

# MMCore defaults to a 5 s device timeout, which is shorter than a plate
# traverse: crossing a 96-well plate is ~100 mm, and both a real Nikon stage
# and Micro-Manager's simulated one take longer than that. Left at the
# default, every long move to a far well raises instead of arriving.
DEVICE_TIMEOUT_MS = 60_000


class ScopeError(RuntimeError):
    """A device is missing, or the microscope refused an operation."""


@dataclass(frozen=True)
class Position:
    x: float
    y: float

    def __iter__(self):
        yield self.x
        yield self.y


@dataclass(frozen=True)
class PropertyInfo:
    """One device property, with everything needed to build a control for it."""

    device: str
    name: str
    value: str
    read_only: bool = False
    allowed: tuple[str, ...] = ()
    lower: float | None = None
    upper: float | None = None

    @property
    def numeric(self) -> bool:
        return self.lower is not None and self.upper is not None

    @property
    def number(self) -> float | None:
        try:
            return float(self.value)
        except (TypeError, ValueError):
            return None

    def describe(self) -> str:
        bits = [f"{self.device}.{self.name} = {self.value}"]
        if self.read_only:
            bits.append("(read-only)")
        elif self.numeric:
            bits.append(f"[{self.lower:g} … {self.upper:g}]")
        elif self.allowed:
            bits.append("{" + ", ".join(self.allowed[:8]) + "}")
        return " ".join(bits)


@dataclass
class ScopeState:
    """A snapshot of everything the dashboard shows. Cheap to build."""

    xy: Position | None = None
    z: float | None = None
    pfs_available: bool = False
    pfs_engaged: bool = False
    pfs_locked: bool = False
    pfs_offset: float | None = None
    pfs_in_range: bool | None = None
    objective: str = ""
    objectives: list[str] = field(default_factory=list)
    channel: str = ""
    channels: list[str] = field(default_factory=list)
    exposure_ms: float | None = None
    shutter_open: bool = False
    auto_shutter: bool = False
    illumination: str = ""
    intensity: PropertyInfo | None = None
    roles: dict[str, str] = field(default_factory=dict)
    error: str = ""


class Scope:
    """A microscope, addressed by role.

    Construct from a Micro-Manager configuration file, or from the demo
    devices for development without hardware::

        scope = Scope.from_config(r"C:\\path\\MMConfig.cfg")
        scope = Scope.demo()
    """

    def __init__(self, core, roles: dict[str, str] | None = None,
                 timeout_ms: int = DEVICE_TIMEOUT_MS,
                 channel_group: str = ""):
        self.core = core
        self.role_warnings: list[str] = []
        # A site whose channel group is named something else says so
        # explicitly, rather than having it inferred.
        self.channel_group_override = channel_group
        try:
            core.setTimeoutMs(int(timeout_ms))
        except Exception:
            pass          # a core that has no timeout knob is fine
        self.roles = dict(roles) if roles else self._discover_roles()

    # ---------------------------------------------------------------- setup

    @classmethod
    def from_config(cls, path: str | Path) -> "Scope":
        from pymmcore_plus import CMMCorePlus

        core = CMMCorePlus()
        core.loadSystemConfiguration(str(path))
        return cls(core)

    @classmethod
    def demo(cls) -> "Scope":
        """The Micro-Manager demo devices — a real core, simulated hardware."""
        from pymmcore_plus import CMMCorePlus

        core = CMMCorePlus()
        core.loadSystemConfiguration()      # ships as the demo config
        return cls(core)

    def close(self) -> None:
        """Release the hardware.

        Only one connection to a Nikon stand exists at a time: while this
        core holds the hub, a second core — another dashboard session, a
        `nikon-control-scope build`, NIS-Elements, Ti2 Control — cannot
        initialise it. So anything that replaces a Scope must close the old
        one first, or the replacement fails with a hub that "would not
        initialise" and no hint that the cause is the previous connection.
        """
        # Close the shutter first. Unloading the devices does not darken the
        # lamp, so a session that ends with the shutter open leaves the
        # transmitted light on the specimen for as long as nobody notices.
        try:
            if self.has("shutter"):
                self.set_shutter(False)
        except Exception:
            pass
        try:
            self.core.unloadAllDevices()
        except Exception:
            pass
        self.roles = {}

    def __enter__(self) -> "Scope":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _discover_roles(self) -> dict[str, str]:
        """Core roles are authoritative; stand resolution fills the rest.

        MMCore has slots for camera/XY/focus/autofocus/shutter and a
        configuration names them explicitly — trust that. It has no slot for
        the PFS offset, the nosepiece or the light path, so those are matched
        by device type the same way on both stands.
        """
        core = self.core
        roles: dict[str, str] = {}
        for role, getter in (("camera", core.getCameraDevice),
                             ("xystage", core.getXYStageDevice),
                             ("focus", core.getFocusDevice),
                             ("autofocus", core.getAutoFocusDevice),
                             ("shutter", core.getShutterDevice)):
            try:
                if name := str(getter() or ""):
                    roles[role] = name
            except Exception:
                pass

        labelled = []
        for label in core.getLoadedDevices():
            if str(label) == "Core":
                continue
            try:
                labelled.append(_Labelled(str(label),
                                          _type_name(core.getDeviceType(label))))
            except Exception:
                continue
        resolved = resolve_roles(labelled)
        for role, name in resolved.items():
            roles.setdefault(role, name)

        # A configuration can name a device that cannot possibly fill its
        # role — a .cfg written before the Ti2's TIRF positioners were
        # excluded still says the XY stage is TIRF1. Honouring that drives
        # the wrong axis and looks exactly like broken stage hardware, so
        # refuse it rather than pass it through.
        for role, name in list(roles.items()):
            if not role_excluded(name, role):
                continue
            replacement = resolved.get(role, "")
            if replacement and replacement != name:
                roles[role] = replacement
                self.role_warnings.append(
                    f"the configuration names {name!r} as the {role}, which "
                    f"can never be one — using {replacement!r} instead. "
                    f"Re-run `nikon-control-scope build --force` to fix the "
                    f".cfg itself."
                )
            else:
                roles.pop(role, None)
                self.role_warnings.append(
                    f"the configuration names {name!r} as the {role}, which "
                    f"can never be one, and nothing else fits — that role is "
                    f"unavailable."
                )
        return roles

    def has(self, role: str) -> bool:
        return bool(self.roles.get(role))

    def _require(self, role: str) -> str:
        name = self.roles.get(role)
        if not name:
            have = ", ".join(f"{r}={self.roles[r]}" for r in ROLES
                             if self.roles.get(r)) or "nothing"
            raise ScopeError(
                f"no device fills the {role!r} role in this configuration "
                f"(have: {have}). Add it in Micro-Manager's Hardware "
                f"Configuration Wizard, or check `nikon-control-scope config`."
            )
        return name

    # ---------------------------------------------------------------- stage

    def xy(self) -> Position:
        dev = self._require("xystage")
        return Position(float(self.core.getXPosition(dev)),
                        float(self.core.getYPosition(dev)))

    def move_xy(self, x: float, y: float, *, wait: bool = True) -> Position:
        dev = self._require("xystage")
        self.core.setXYPosition(dev, float(x), float(y))
        if wait:
            self.core.waitForDevice(dev)
        return self.xy()

    def move_xy_by(self, dx: float, dy: float, *, force: bool = False,
                   wait: bool = True) -> Position:
        """Jog the stage. Refuses an implausibly large step unless forced."""
        if not force and max(abs(dx), abs(dy)) > MAX_JOG_UM:
            raise ScopeError(
                f"jog of ({dx:.0f}, {dy:.0f}) µm exceeds the {MAX_JOG_UM:.0f} "
                f"µm guard. Use move_xy() for a deliberate long move."
            )
        here = self.xy()
        return self.move_xy(here.x + dx, here.y + dy, wait=wait)

    # ---------------------------------------------------------------- focus

    def z(self) -> float:
        return float(self.core.getPosition(self._require("focus")))

    def move_z(self, value: float, *, keep_pfs: bool = True,
               wait: bool = True) -> float:
        """Drive the focus drive to an absolute position.

        Moving Z disables continuous focus, so when PFS is engaged this
        suspends it explicitly and re-engages afterwards. That is a real
        change of the locked plane — to nudge focus *within* a lock, use
        :meth:`focus_by`.
        """
        dev = self._require("focus")
        resume = keep_pfs and self.pfs_engaged()
        if resume:
            self.disengage_pfs()
        self.core.setPosition(dev, float(value))
        if wait:
            self.core.waitForDevice(dev)
        if resume:
            self.engage_pfs()
        return self.z()

    def move_z_by(self, dz: float, **kw) -> float:
        return self.move_z(self.z() + float(dz), **kw)

    def focus_by(self, delta: float) -> float:
        """Nudge focus by ``delta``, using whichever control is correct.

        PFS engaged → move the PFS offset, which is the only thing that
        changes focus while locked. PFS off → move Z.

        **The units differ between those two cases**: microns for the Z
        drive, the offset device's own units for the PFS offset, and one
        offset unit is not one micron. Nothing here converts between them,
        because the conversion is per-stand and per-objective and this code
        does not know it. Callers that need a known physical distance must
        use :meth:`move_z` (microns) or :meth:`set_pfs_offset` (offset
        units) and say which they mean.
        """
        dz = delta
        if self.pfs_engaged() and self.has("pfsoffset"):
            return self.set_pfs_offset(self.pfs_offset() + float(dz))
        return self.move_z_by(dz)

    # ------------------------------------------------------------------ PFS

    def pfs_available(self) -> bool:
        return self.has("autofocus")

    def pfs_engaged(self) -> bool:
        if not self.pfs_available():
            return False
        try:
            return bool(self.core.isContinuousFocusEnabled())
        except Exception:
            return False

    def pfs_locked(self) -> bool:
        """Engaged means "switched on"; locked means "actually holding"."""
        if not self.pfs_available():
            return False
        try:
            return bool(self.core.isContinuousFocusLocked())
        except Exception:
            return False

    def engage_pfs(self, timeout_s: float = PFS_LOCK_TIMEOUT_S) -> bool:
        """Turn PFS on and wait for a lock. Returns whether it locked.

        Returning rather than raising is deliberate: failing to lock is a
        normal event (no coverslip in range, a dirty dish) and the caller
        usually wants to report it and carry on, not crash a scan.
        """
        self._require("autofocus")
        self.core.enableContinuousFocus(True)
        deadline = time.monotonic() + max(0.0, timeout_s)
        while time.monotonic() < deadline:
            if self.pfs_locked():
                return True
            time.sleep(0.05)
        return self.pfs_locked()

    def disengage_pfs(self) -> None:
        self._require("autofocus")
        self.core.enableContinuousFocus(False)

    def pfs_in_range(self) -> bool | None:
        """Can PFS see the coverslip at all? Distinct from being locked.

        "In range" means the IR beam is finding the glass; "locked" means it
        is holding. A dish outside the search range never locks, and a lock
        at the wrong OFFSET holds a plane that is nowhere near the cells —
        which looks like a focus that is working but blurry.

        Returns None when the device exposes no such property.
        """
        if not self.pfs_available():
            return None
        for info in self.properties("autofocus"):
            if "range" in info.name.lower():
                return "in range" in info.value.lower() or info.value in ("1", "Yes")
        return None

    def pfs_details(self) -> list[PropertyInfo]:
        """The PFS device's own properties — its offset range, IR LED, status.

        Worth surfacing: the search LED intensity and the objective in use
        both change whether a lock is even possible.
        """
        return self.properties("autofocus") if self.pfs_available() else []

    def pfs_offset_limits(self) -> tuple[float, float] | None:
        """The offset device's travel, if it declares any."""
        if not self.has("pfsoffset"):
            return None
        for info in self.properties("pfsoffset"):
            if info.numeric and "position" in info.name.lower():
                return (info.lower, info.upper)
        return None

    def pfs_offset(self) -> float:
        return float(self.core.getPosition(self._require("pfsoffset")))

    def set_pfs_offset(self, value: float, *, wait: bool = True) -> float:
        dev = self._require("pfsoffset")
        self.core.setPosition(dev, float(value))
        if wait:
            self.core.waitForDevice(dev)
        return self.pfs_offset()

    # ------------------------------------------------------------ objective

    def objectives(self) -> list[str]:
        if not self.has("nosepiece"):
            return []
        try:
            return [str(s) for s in
                    self.core.getStateLabels(self.roles["nosepiece"])]
        except Exception:
            return []

    def objective(self) -> str:
        if not self.has("nosepiece"):
            return ""
        try:
            return str(self.core.getStateLabel(self.roles["nosepiece"]))
        except Exception:
            return ""

    def set_objective(self, label: str, *, confirm: bool = False) -> str:
        """Rotate the turret. Requires ``confirm=True``.

        Deliberately awkward. The turret is the crash-prone device on the
        Nikon adapters, and a rotation with a plate loaded can drive a dry
        40× into glass. This project images entirely at 40×, so a turret move
        is an exceptional event and should read like one at the call site.
        """
        dev = self._require("nosepiece")
        if not confirm:
            raise ScopeError(
                "set_objective() needs confirm=True — rotating the turret "
                "under a loaded plate can crash the objective into it."
            )
        self.core.setStateLabel(dev, str(label))
        self.core.waitForDevice(dev)
        return self.objective()

    # ----------------------------------------------------------- properties

    def device_labels(self) -> list[str]:
        try:
            return [str(d) for d in self.core.getLoadedDevices()
                    if str(d) != "Core"]
        except Exception:
            return []

    def _target(self, name: str) -> str:
        """Accept either a role ('shutter') or a device label ('DiaLamp')."""
        return self.roles.get(name, name)

    def properties(self, target: str) -> list[PropertyInfo]:
        """Every property of a device, with its limits and allowed values.

        This is the escape hatch for everything MMCore has no dedicated API
        for — lamp intensity, camera binning, a filter wheel's speed.
        """
        dev = self._target(target)
        out: list[PropertyInfo] = []
        try:
            names = [str(p) for p in self.core.getDevicePropertyNames(dev)]
        except Exception:
            return out
        for prop in names:
            try:
                value = str(self.core.getProperty(dev, prop))
            except Exception:
                value = "<unreadable>"
            try:
                read_only = bool(self.core.isPropertyReadOnly(dev, prop))
            except Exception:
                read_only = False
            lower = upper = None
            try:
                if self.core.hasPropertyLimits(dev, prop):
                    lower = float(self.core.getPropertyLowerLimit(dev, prop))
                    upper = float(self.core.getPropertyUpperLimit(dev, prop))
            except Exception:
                pass
            try:
                allowed = tuple(str(a) for a in
                                self.core.getAllowedPropertyValues(dev, prop))
            except Exception:
                allowed = ()
            out.append(PropertyInfo(dev, prop, value, read_only, allowed,
                                    lower, upper))
        return out

    def get_property(self, target: str, prop: str) -> str:
        return str(self.core.getProperty(self._target(target), prop))

    def set_property(self, target: str, prop: str, value) -> str:
        dev = self._target(target)
        self.core.setProperty(dev, prop, value)
        try:
            self.core.waitForDevice(dev)
        except Exception:
            pass
        return self.get_property(dev, prop)

    # --------------------------------------------------------- illumination

    def intensity_property(self) -> PropertyInfo | None:
        """The 'how bright' knob, wherever the vendor decided to put it.

        Looked for on the shutter/lamp device first, then the hub, then
        anything else — because there is no MMCore API for brightness and
        every adapter names it differently.
        """
        order: list[str] = []
        for role in ("shutter", "lightpath"):
            if label := self.roles.get(role):
                order.append(label)
        for label in self.device_labels():
            low = label.lower()
            if label not in order and any(h in low for h in LIGHT_LABEL_HINTS):
                order.append(label)

        seen: set[str] = set()
        best: PropertyInfo | None = None
        for label in order:
            if label in seen:
                continue
            seen.add(label)
            for info in self.properties(label):
                if info.read_only:
                    continue
                low = info.name.lower()
                if not any(h in low for h in INTENSITY_HINTS):
                    continue
                if info.numeric:
                    return info          # numeric with limits is ideal
                best = best or info
        return best

    def intensity(self) -> float | None:
        info = self.intensity_property()
        return info.number if info else None

    def set_intensity(self, value: float) -> float | None:
        info = self.intensity_property()
        if info is None:
            raise ScopeError(
                "no intensity property found on any loaded device — list "
                "them with `nikon-control-scope config <cfg> --properties` "
                "and set it directly with set_property()."
            )
        if info.numeric:
            value = max(info.lower, min(info.upper, float(value)))
        self.set_property(info.device, info.name, value)
        return self.intensity()

    def shutter_open(self) -> bool:
        if not self.has("shutter"):
            return False
        try:
            return bool(self.core.getShutterOpen(self.roles["shutter"]))
        except Exception:
            return False

    def set_shutter(self, open_: bool) -> bool:
        dev = self._require("shutter")
        self.core.setShutterOpen(dev, bool(open_))
        self.core.waitForDevice(dev)
        return self.shutter_open()

    def auto_shutter(self) -> bool:
        """Whether MMCore opens the shutter around each acquisition itself."""
        try:
            return bool(self.core.getAutoShutter())
        except Exception:
            return False

    def set_auto_shutter(self, on: bool) -> bool:
        self.core.setAutoShutter(bool(on))
        return self.auto_shutter()

    def illumination(self) -> str:
        """Plain words for why a frame might be black.

        A configuration with no shutter and no lamp cannot turn a light on at
        all, and that is worth saying out loud rather than leaving someone to
        wonder whether the camera is broken.
        """
        if not self.has("shutter"):
            return ("no shutter or lamp in this configuration — nothing here "
                    "can switch a light on, so a dark frame is expected")
        if self.auto_shutter():
            return "auto-shutter on: opened for each acquisition"
        return "shutter open" if self.shutter_open() else "shutter CLOSED"

    # --------------------------------------------------------------- camera

    def snap(self):
        """Acquire one image and return it as a numpy array."""
        self._require("camera")
        self.core.snapImage()
        return self.core.getImage()

    def exposure_ms(self) -> float:
        self._require("camera")
        return float(self.core.getExposure())

    def set_exposure_ms(self, ms: float) -> float:
        self._require("camera")
        self.core.setExposure(float(ms))
        return self.exposure_ms()

    # ------------------------------------------------------------- channels

    def channel_group(self) -> str:
        """The config group that selects a channel, if there is one.

        Only a group actually named for channels counts. An earlier version
        fell back to the first group MMCore reported, which means
        :meth:`set_channel` writes to whatever that happens to be — a
        "Camera" group (silently changing resolution mid-experiment) or an
        "Objective" group (rotating the turret under a loaded plate, with
        none of the confirmation :meth:`set_objective` demands). A
        configuration with no channel group has no channels; saying so is
        the honest answer.
        """
        try:
            groups = [str(g) for g in self.core.getAvailableConfigGroups()]
        except Exception:
            return ""
        for want in CHANNEL_GROUP_NAMES:
            if want in groups:
                return want
        return self.channel_group_override if \
            self.channel_group_override in groups else ""

    def channels(self) -> list[str]:
        group = self.channel_group()
        if not group:
            return []
        try:
            return [str(c) for c in self.core.getAvailableConfigs(group)]
        except Exception:
            return []

    def channel(self) -> str:
        group = self.channel_group()
        if not group:
            return ""
        try:
            return str(self.core.getCurrentConfig(group))
        except Exception:
            return ""

    def set_channel(self, name: str) -> str:
        group = self.channel_group()
        if not group:
            raise ScopeError("this configuration defines no channel group")
        self.core.setConfig(group, str(name))
        self.core.waitForConfig(group, str(name))
        return self.channel()

    # ---------------------------------------------------------------- state

    def state(self) -> ScopeState:
        """Read everything at once, tolerating devices that are busy or gone.

        A dashboard polls this; one unreadable device must not blank the
        whole panel, so each read is guarded and the rest still populate.
        """
        st = ScopeState(roles=dict(self.roles))
        problems: list[str] = []

        def attempt(name, fn):
            try:
                return fn()
            except Exception as exc:
                problems.append(f"{name}: {exc}")
                return None

        if self.has("xystage"):
            st.xy = attempt("xy", self.xy)
        if self.has("focus"):
            st.z = attempt("z", self.z)
        st.pfs_available = self.pfs_available()
        if st.pfs_available:
            st.pfs_engaged = bool(attempt("pfs", self.pfs_engaged))
            st.pfs_locked = bool(attempt("pfs lock", self.pfs_locked))
        if self.has("pfsoffset"):
            st.pfs_offset = attempt("pfs offset", self.pfs_offset)
        if st.pfs_available:
            st.pfs_in_range = attempt("pfs range", self.pfs_in_range)
        if self.has("nosepiece"):
            st.objective = attempt("objective", self.objective) or ""
            st.objectives = attempt("objectives", self.objectives) or []
        if self.has("camera"):
            st.exposure_ms = attempt("exposure", self.exposure_ms)
        st.shutter_open = bool(attempt("shutter", self.shutter_open))
        st.auto_shutter = bool(attempt("auto shutter", self.auto_shutter))
        st.illumination = attempt("illumination", self.illumination) or ""
        st.intensity = attempt("intensity", self.intensity_property)
        st.channels = attempt("channels", self.channels) or []
        st.channel = attempt("channel", self.channel) or ""
        st.error = "; ".join(problems)
        return st

    def describe(self) -> str:
        st = self.state()
        bits = []
        if st.xy:
            bits.append(f"XY ({st.xy.x:.0f}, {st.xy.y:.0f}) µm")
        if st.z is not None:
            bits.append(f"Z {st.z:.2f} µm")
        if st.pfs_available:
            bits.append("PFS " + ("locked" if st.pfs_locked else
                                  "on (not locked)" if st.pfs_engaged else "off"))
        if st.objective:
            bits.append(st.objective)
        return "  |  ".join(bits) or "no devices"


@dataclass
class _Labelled:
    """Adapter so a loaded device's *label* feeds the role resolver."""

    name: str
    type: str


def _type_name(value) -> str:
    from pymmcore_plus import DeviceType

    try:
        return DeviceType(int(value)).name.replace("Device", "")
    except Exception:
        return str(value)
