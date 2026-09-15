"""Find out what Micro-Manager can see on this machine, and probe it.

Micro-Manager cannot truly enumerate "what is plugged in" — there is no
universal bus scan. What it can do, and what this module exposes, is three
increasingly committal steps:

1. **List installed device adapters** and the devices each one offers. This
   says what the machine is *capable* of driving.
2. **Probe** a device: load it into a throwaway core and initialise it. This
   is the honest test of whether the hardware is actually connected and
   talking, and it is how you bring a rig up one component at a time.
3. **Inspect a config**: for an existing ``.cfg``, report the devices it
   defines, their types, and which core role each fills (camera, XY stage,
   focus, autofocus).

Probing is deliberately one device at a time in a fresh core, so a device
that hangs or crashes its adapter cannot take the rest of the inventory with
it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Adapters that matter for a Nikon Ti2 rig, with what each is responsible for.
NIKON_ADAPTERS = {
    "NikonTi2": "Ti2-E stand: XY stage, Z drive, PFS, nosepiece, shutters",
    "NikonTI": "older Ti/Ti-E stand (superseded by NikonTi2)",
}
CAMERA_ADAPTERS = {
    "HamamatsuHam": "Hamamatsu ORCA (DCAM)",
    "PVCAM": "Photometrics (PVCAM)",
    "AndorSDK3": "Andor sCMOS",
}
# The Nikon adapter is a thin wrapper over Nikon's closed SDK; this DLL has to
# sit beside the adapter or loading fails with an unhelpful error.
TI2_DRIVER_DLL = "Ti2_Mic_Driver.dll"


@dataclass
class DeviceEntry:
    library: str
    name: str
    description: str = ""
    type: str = ""


@dataclass
class LoadedDevice:
    label: str
    library: str
    name: str
    type: str
    role: str = ""          # camera / xystage / focus / autofocus, if any
    description: str = ""


@dataclass
class ProbeResult:
    library: str
    name: str
    ok: bool
    error: str = ""
    properties: dict[str, str] = field(default_factory=dict)

    def describe(self) -> str:
        head = f"{self.library}/{self.name}"
        return f"{head}: connected" if self.ok else f"{head}: FAILED — {self.error}"


def device_type_name(value) -> str:
    """Readable device type — MMCore reports these as ints."""
    from pymmcore_plus import DeviceType

    try:
        return DeviceType(int(value)).name.replace("Device", "") or str(value)
    except Exception:
        return str(value)


def new_core():
    """A fresh CMMCorePlus. Imported lazily so pure modules stay importable."""
    from pymmcore_plus import CMMCorePlus

    return CMMCorePlus()


def mm_install() -> Path | None:
    """Where the Micro-Manager device adapters live, if found."""
    from pymmcore_plus import find_micromanager

    found = find_micromanager()
    return Path(found) if found else None


def available_adapters(core=None) -> list[str]:
    """Device adapter libraries installed on this machine."""
    core = core or new_core()
    try:
        return sorted(core.getDeviceAdapterNames())
    except Exception:
        return []


def adapter_devices(library: str, core=None) -> list[DeviceEntry]:
    """Devices a given adapter offers (without loading any of them)."""
    core = core or new_core()
    try:
        names = list(core.getAvailableDevices(library))
    except Exception:
        return []
    try:
        descs = list(core.getAvailableDeviceDescriptions(library))
    except Exception:
        descs = [""] * len(names)
    try:
        types = [device_type_name(t)
                 for t in core.getAvailableDeviceTypes(library)]
    except Exception:
        types = [""] * len(names)
    return [
        DeviceEntry(library=library, name=n,
                    description=descs[i] if i < len(descs) else "",
                    type=types[i] if i < len(types) else "")
        for i, n in enumerate(names)
    ]


def probe(library: str, name: str, *, read_properties: bool = True) -> ProbeResult:
    """Try to load and initialise one device — the real 'is it connected?' test.

    Runs in its own core so a failure cannot poison anything else. Returns a
    result rather than raising: a rig bring-up wants the whole picture, not
    the first exception.
    """
    core = new_core()
    label = "__probe__"
    try:
        core.loadDevice(label, library, name)
    except Exception as exc:
        return ProbeResult(library, name, False, f"load failed: {exc}")
    try:
        core.initializeDevice(label)
    except Exception as exc:
        return ProbeResult(library, name, False, f"initialise failed: {exc}")

    props: dict[str, str] = {}
    if read_properties:
        try:
            for p in core.getDevicePropertyNames(label):
                try:
                    props[str(p)] = str(core.getProperty(label, p))
                except Exception:
                    props[str(p)] = "<unreadable>"
        except Exception:
            pass
    try:
        core.unloadDevice(label)
    except Exception:
        pass
    return ProbeResult(library, name, True, "", props)


def load_config(path: str | Path, core=None):
    """Load a Micro-Manager system configuration."""
    core = core or new_core()
    core.loadSystemConfiguration(str(path))
    return core


def loaded_devices(core) -> list[LoadedDevice]:
    """Devices in a loaded configuration, with their core roles."""
    roles = core_roles(core)
    by_label = {v: k for k, v in roles.items() if v}
    out: list[LoadedDevice] = []
    for label in core.getLoadedDevices():
        if label == "Core":
            continue
        try:
            lib = str(core.getDeviceLibrary(label))
            name = str(core.getDeviceName(label))
            dtype = device_type_name(core.getDeviceType(label))
            desc = str(core.getDeviceDescription(label))
        except Exception:
            lib = name = dtype = desc = ""
        out.append(LoadedDevice(label=str(label), library=lib, name=name,
                                type=dtype, role=by_label.get(str(label), ""),
                                description=desc))
    return out


def core_roles(core) -> dict[str, str]:
    """Which device fills each core role, for the roles this project needs."""
    getters = {
        "camera": core.getCameraDevice,
        "xystage": core.getXYStageDevice,
        "focus": core.getFocusDevice,
        "autofocus": core.getAutoFocusDevice,
        "shutter": core.getShutterDevice,
    }
    roles: dict[str, str] = {}
    for role, fn in getters.items():
        try:
            roles[role] = str(fn() or "")
        except Exception:
            roles[role] = ""
    return roles


def nikon_readiness(core=None) -> list[str]:
    """Ti2-specific findings worth knowing before a bring-up.

    Encodes the two failure modes that cost the most time on this hardware:
    the Nikon SDK DLL not sitting beside the adapter, and the fact that the
    stand adapter never provides the camera.
    """
    core = core or new_core()
    notes: list[str] = []
    adapters = set(available_adapters(core))

    ti2 = [a for a in NIKON_ADAPTERS if a in adapters]
    if not ti2:
        notes.append(
            "No Nikon stand adapter found. On Windows the NikonTi2 adapter "
            "ships with Micro-Manager; if it is missing, this is not a "
            "Micro-Manager build that can drive the stand."
        )
    else:
        notes.append(f"Nikon stand adapter(s) present: {', '.join(ti2)}.")

    install = mm_install()
    if install and os.name == "nt":
        if not any(install.rglob(TI2_DRIVER_DLL)):
            notes.append(
                f"{TI2_DRIVER_DLL} was NOT found in {install}. The NikonTi2 "
                "adapter wraps Nikon's closed SDK and will fail to load "
                "without it — copy it from the Ti2 Control installation into "
                "the Micro-Manager folder."
            )
        else:
            notes.append(f"{TI2_DRIVER_DLL} found beside the adapter.")
        notes.append(
            "Check the installed Ti2 Control version: 2.10 and 2.20 are "
            "documented to crash Micro-Manager when the nosepiece device is "
            "added (mmCoreAndDevices #44); 2.00 is reported working."
        )

    cams = [a for a in CAMERA_ADAPTERS if a in adapters]
    notes.append(
        f"Camera adapter(s) present: {', '.join(cams)}." if cams else
        "No common sCMOS camera adapter (Hamamatsu/Photometrics/Andor) found "
        "— the stand adapter does not provide the camera, it needs its own."
    )
    return notes
