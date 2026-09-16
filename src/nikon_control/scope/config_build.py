"""Build a Micro-Manager configuration by connecting devices one at a time.

Micro-Manager's own Hardware Configuration Wizard does this, but it is part
of the Java GUI — and ``mmcore install`` fetches the device adapters, which
may arrive without it. This module does the same job from the command line,
and does it in the order a rig is actually brought up:

1. Load the stand's **hub** device and initialise it.
2. Ask the *initialised hub* what peripherals are really attached
   (``getInstalledDevices``). This is the only way to learn a Ti2's device
   names: that adapter enumerates from Nikon's SDK, so nothing can be known
   in advance — it has to be asked, with the hardware powered on.
3. Load each peripheral **one at a time**, keeping the ones that initialise
   and reporting the ones that do not, instead of failing the whole build.
4. Add a camera — no stand adapter provides one.
5. Resolve roles and write the ``.cfg``.

The result is a configuration containing exactly the devices that answered,
which is a far better starting point than one listing everything the adapter
*could* offer.

A hand-written ``.cfg`` is a plain text file; the format is one directive per
line, ``Device``/``Parent``/``Property``/``Label``, with everything between
``Property,Core,Initialize,0`` and ``...,1`` being load-time setup.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field

from .stand import STANDS, Stand, resolve_roles, stand_for_adapter


def new_core():
    """Re-exported so callers need only this module to run a build."""
    from .discover import new_core as _new_core

    return _new_core()

# Core role property -> the role name this project uses.
CORE_ROLE_PROPERTIES = {
    "Camera": "camera",
    "Shutter": "shutter",
    "Focus": "focus",
    "XYStage": "xystage",
    "AutoFocus": "autofocus",
}

# Devices that are more trouble than they are worth in a generated config.
# TIDiaLamp has crashed Micro-Manager with some Nikon driver versions, and
# nothing in this project uses the transmitted-light lamp programmatically.
SKIP_DEVICES = {"TIDiaLamp"}


@dataclass
class ConfigDevice:
    label: str
    library: str
    device: str
    parent: str = ""
    type: str = ""
    role: str = ""


@dataclass
class BuildResult:
    devices: list[ConfigDevice] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)  # name, why
    roles: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def hub(self) -> str:
        return next((d.label for d in self.devices if d.type == "Hub"), "")

    def summary(self) -> list[str]:
        out = [f"{len(self.devices)} device(s) connected"]
        for d in self.devices:
            role = f"   [{d.role}]" if d.role else ""
            out.append(f"  {d.label:<24} {d.type:<12} {d.library}/{d.device}{role}")
        if self.failures:
            out.append(f"{len(self.failures)} did not connect:")
            out.extend(f"  {name}: {why}" for name, why in self.failures)
        return out


def safe_label(name: str) -> str:
    """A device label safe for a comma-separated config file.

    The Ti2 names its hub something like ``*Ti2-E__0: Nikon Ti2 microscope``;
    left alone that would break the file format the first time it contained a
    comma, and is unpleasant to type in any case.
    """
    label = str(name).strip().lstrip("*").strip()
    label = label.split(":")[-1].strip() if ":" in label else label
    label = label.replace(",", " ").replace("#", "")
    return " ".join(label.split()) or "Device"


def _unique(label: str, taken: set[str]) -> str:
    if label not in taken:
        return label
    i = 2
    while f"{label}-{i}" in taken:
        i += 1
    return f"{label}-{i}"


def _type_name(core, label: str) -> str:
    from .discover import device_type_name

    try:
        return device_type_name(core.getDeviceType(label))
    except Exception:
        return ""


def _pick_library(core, adapter: str | None) -> tuple[str, Stand | None]:
    """Which adapter to build from, and its stand entry if it is a Nikon one.

    An explicit adapter is used as given — including a non-Nikon one, which
    is how the demo devices are built into a config for testing. With no
    adapter named, the first installed Nikon stand wins.
    """
    from .discover import available_adapters

    installed = set(available_adapters(core))
    if adapter:
        return (adapter if adapter in installed else ""), stand_for_adapter(adapter)
    for st in STANDS:
        if st.adapter in installed:
            return st.adapter, st
    return "", None


def _hub_device(core, library: str) -> str:
    """The Hub-typed device an adapter offers, if it has one."""
    from .discover import scan_adapter

    scan = scan_adapter(library, core)
    for entry in scan.devices:
        if entry.type == "Hub":
            return entry.name
    # Some adapters expose no hub; then the devices are loaded directly.
    return ""


def build(core=None, *, stand_adapter: str | None = None,
          camera_adapter: str | None = None,
          camera_device: str | None = None,
          skip: set[str] | None = None) -> BuildResult:
    """Connect what is really attached and return a configuration for it.

    Leaves the core loaded with everything that worked, so the caller can
    inspect it further. Never raises for a device that refuses to connect —
    that is a finding, not an error.
    """
    from .discover import new_core, scan_adapter

    core = core or new_core()
    skip = set(skip or ()) | SKIP_DEVICES
    result = BuildResult()
    taken: set[str] = set()

    library, stand = _pick_library(core, stand_adapter)
    if not library:
        result.notes.append(
            f"adapter {stand_adapter!r} is not installed"
            if stand_adapter else
            "No Nikon stand adapter is installed — run "
            "`nikon-control-scope stand` first."
        )
        return result

    # --- the hub, which is what makes the rest discoverable ---------------
    hub_name = _hub_device(core, library)
    hub_label = ""
    peripherals: list[str] = []
    if hub_name:
        hub_label = _unique(safe_label(hub_name), taken)
        try:
            core.loadDevice(hub_label, library, hub_name)
            core.initializeDevice(hub_label)
        except Exception as exc:
            result.failures.append((hub_name, f"hub failed: {exc}"))
            why = (f"{stand.driver.dll} and the stand's power state are the "
                   f"usual causes." if stand else
                   "Check the adapter's own requirements.")
            result.notes.append(
                f"The {library} hub would not initialise, so nothing below it "
                f"can be found. {why}"
            )
            return result
        taken.add(hub_label)
        result.devices.append(ConfigDevice(hub_label, library, hub_name,
                                           type="Hub"))
        try:
            peripherals = [str(p) for p in core.getInstalledDevices(hub_label)]
        except Exception as exc:
            result.failures.append((hub_label, f"cannot list peripherals: {exc}"))

    if not peripherals:
        # No hub, or a hub that lists nothing: fall back to the adapter's own
        # device list (which is what the older Ti publishes anyway).
        peripherals = [e.name for e in scan_adapter(library, core).devices
                       if e.type != "Hub"]

    # --- peripherals, one at a time ---------------------------------------
    for name in peripherals:
        if name in skip:
            result.failures.append((name, "skipped by default (known to be "
                                          "unstable or unused)"))
            continue
        label = _unique(safe_label(name), taken)
        try:
            core.loadDevice(label, library, name)
            if hub_label:
                core.setParentLabel(label, hub_label)
            core.initializeDevice(label)
        except Exception as exc:
            result.failures.append((name, str(exc).split("\n")[0][:200]))
            try:
                core.unloadDevice(label)
            except Exception:
                pass
            continue
        taken.add(label)
        result.devices.append(ConfigDevice(label, library, name,
                                           parent=hub_label,
                                           type=_type_name(core, label)))

    # --- the camera, which the stand never provides ------------------------
    cam = _add_camera(core, result, taken, camera_adapter, camera_device)
    if not cam:
        result.notes.append(
            "No camera in this configuration — the stand adapter does not "
            "provide one. Pass --camera-adapter (e.g. HamamatsuHam) once the "
            "camera is powered on."
        )

    # --- roles --------------------------------------------------------------
    # Resolve on the *labels*, since that is what the config file and the
    # core will use — not the underlying device names.
    from .discover import DeviceEntry

    result.roles = resolve_roles(
        [DeviceEntry(library=d.library, name=d.label, type=d.type)
         for d in result.devices])
    if cam:
        result.roles["camera"] = cam
    for d in result.devices:
        for role, label in result.roles.items():
            if label == d.label:
                d.role = role
    return result


def _add_camera(core, result: BuildResult, taken: set[str],
                adapter: str | None, device: str | None) -> str:
    from .discover import CAMERA_ADAPTERS, available_adapters, scan_adapter

    # A camera already loaded (the demo hub offers one) needs no second copy.
    for d in result.devices:
        if d.type == "Camera":
            return d.label

    installed = set(available_adapters(core))
    candidates = ([adapter] if adapter else
                  [a for a in CAMERA_ADAPTERS if a in installed])
    for lib in candidates:
        if lib not in installed:
            result.failures.append((lib, "camera adapter not installed"))
            continue
        names = ([device] if device else
                 [e.name for e in scan_adapter(lib, core).devices
                  if e.type == "Camera"])
        for name in names:
            label = _unique(safe_label(name), taken)
            try:
                core.loadDevice(label, lib, name)
                core.initializeDevice(label)
            except Exception as exc:
                result.failures.append((f"{lib}/{name}",
                                        str(exc).split("\n")[0][:200]))
                try:
                    core.unloadDevice(label)
                except Exception:
                    pass
                continue
            taken.add(label)
            result.devices.append(ConfigDevice(label, lib, name, type="Camera",
                                               role="camera"))
            return label
    return ""


def to_text(result: BuildResult, core=None) -> str:
    """Render a ``.cfg``.

    State labels are written out for every State device that has them, so a
    reloaded configuration still calls objective 3 "Plan Fluor 40x" instead
    of "State-2".
    """
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# Generated by nikon-control-scope on {stamp}",
        "# Devices that failed to connect are listed at the end as comments.",
        "",
        "# Reset",
        "Property,Core,Initialize,0",
        "",
        "# Devices",
    ]
    lines += [f"Device,{d.label},{d.library},{d.device}"
              for d in result.devices]

    parents = [d for d in result.devices if d.parent]
    lines += ["", "# Hub (parent) references"]
    lines += [f"Parent,{d.label},{d.parent}" for d in parents]

    lines += ["", "# Initialize", "Property,Core,Initialize,1", "", "# Roles"]
    for prop, role in CORE_ROLE_PROPERTIES.items():
        if label := result.roles.get(role):
            lines.append(f"Property,Core,{prop},{label}")
    lines.append("Property,Core,AutoShutter,1")

    if core is not None:
        labels = _state_labels(core, result)
        if labels:
            lines += ["", "# Labels", *labels]

    if result.failures:
        lines += ["", "# Did not connect:"]
        lines += [f"#   {name}: {why}" for name, why in result.failures]
    return "\n".join(lines) + "\n"


def _state_labels(core, result: BuildResult) -> list[str]:
    out: list[str] = []
    for d in result.devices:
        if d.type != "State":
            continue
        try:
            states = [str(s) for s in core.getStateLabels(d.label)]
        except Exception:
            continue
        out.append(f"# {d.label}")
        out += [f"Label,{d.label},{i},{s}" for i, s in enumerate(states)]
    return out
