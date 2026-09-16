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

Which Nikon stand is attached — Ti2 or the older Ti — lives in :mod:`.stand`;
this module is the layer that puts a core behind it.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path

from .stand import (SHARED_NOTES, STANDS, Stand, missing_roles,  # noqa: F401
                    resolve_roles, stand_for_adapter)

# Adapters that matter for a Nikon rig, with what each is responsible for.
NIKON_ADAPTERS = {s.adapter: s.label for s in STANDS}
CAMERA_ADAPTERS = {
    "HamamatsuHam": "Hamamatsu ORCA (DCAM)",
    "PVCAM": "Photometrics (PVCAM)",
    "AndorSDK3": "Andor sCMOS",
}
# Kept for callers that imported it before the stand registry existed.
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
class AdapterScan:
    """What one adapter offers — and why, when it offers nothing."""

    library: str
    installed: bool
    devices: list[DeviceEntry] = field(default_factory=list)
    error: str = ""


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


@dataclass
class DriverLocation:
    """Where a stand's vendor DLL was found — and whether that is good enough.

    Finding the DLL is not the same as the adapter being able to load it. The
    Ti2 adapter looks for its driver *beside itself*, so a copy sitting only
    in the Nikon SDK directory satisfies a search and still fails the load —
    which is exactly the state a fresh install is in.
    """

    dll: str
    path: Path | None = None
    beside_adapter: bool = False    # found in the Micro-Manager folder?
    satisfied: bool = False         # good enough for THIS stand's adapter?


@dataclass
class StandStatus:
    """Everything known about one stand generation on this machine."""

    stand: Stand
    installed: bool
    devices: list[DeviceEntry] = field(default_factory=list)
    error: str = ""
    driver: DriverLocation | None = None
    roles: dict[str, str] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        """Devices enumerated and the roles the loop needs are all present."""
        return bool(self.devices) and not missing_roles(self.roles)

    def diagnosis(self, mm_dir: Path | None = None) -> list[str]:
        """What is wrong, and what to do about it — in that order."""
        s = self.stand
        out: list[str] = []
        if not self.installed:
            out.append(f"{s.adapter} is not installed in this Micro-Manager.")
            return out

        if self.devices:
            out.append(f"{s.adapter}: {len(self.devices)} device(s).")
            if not s.dynamic:
                out.append(s.empty_list_meaning)
        else:
            out.append(f"{s.adapter}: installed but offers no devices.")
            out.append(s.empty_list_meaning)
            if self.error:
                out.append(f"Enumeration error: {self.error}")

        found = self.driver
        target = mm_dir or "the Micro-Manager folder"
        if found and found.satisfied:
            out.append(f"{s.driver.dll} found at {found.path}.")
        elif found and found.path:
            # The trap: present on the machine, wrong place for the adapter.
            out.append(
                f"{s.driver.dll} is on this machine at {found.path}, but the "
                f"{s.adapter} adapter loads it from its own folder — COPY it "
                f"to {target} (do not move or rename it)."
            )
        else:
            where = f" in {mm_dir}" if mm_dir and s.driver.beside_adapter else ""
            out.append(f"{s.driver.dll} NOT found{where} — "
                       f"{s.driver.fix(target)}.")

        missing = missing_roles(self.roles)
        if self.devices and missing:
            out.append("Roles not resolved: "
                       + ", ".join(missing)
                       + " (fine if the stand genuinely lacks them).")
        return out


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


def scan_adapter(library: str, core=None) -> AdapterScan:
    """Devices a given adapter offers, plus why it offered none.

    The distinction matters: "adapter not installed" and "adapter installed
    but its vendor SDK is unreachable" look identical in a bare device list,
    and they have completely different fixes.
    """
    core = core or new_core()
    if library not in set(available_adapters(core)):
        return AdapterScan(library, installed=False,
                           error="adapter not installed")
    try:
        names = list(core.getAvailableDevices(library))
    except Exception as exc:
        return AdapterScan(library, installed=True, error=str(exc))
    try:
        descs = list(core.getAvailableDeviceDescriptions(library))
    except Exception:
        descs = [""] * len(names)
    try:
        types = [device_type_name(t)
                 for t in core.getAvailableDeviceTypes(library)]
    except Exception:
        types = [""] * len(names)
    devices = [
        DeviceEntry(library=library, name=n,
                    description=descs[i] if i < len(descs) else "",
                    type=types[i] if i < len(types) else "")
        for i, n in enumerate(names)
    ]
    return AdapterScan(library, installed=True, devices=devices)


def adapter_devices(library: str, core=None) -> list[DeviceEntry]:
    """Devices a given adapter offers (without loading any of them)."""
    return scan_adapter(library, core).devices


def probe(library: str, name: str, *, read_properties: bool = True,
          core_factory=None) -> ProbeResult:
    """Try to load and initialise one device — the real 'is it connected?' test.

    Runs in its own core so a failure cannot poison anything else. Returns a
    result rather than raising: a rig bring-up wants the whole picture, not
    the first exception.
    """
    core = (core_factory or new_core)()

    # MMCore's "unknown adapter" error pastes all 265 installed adapter names
    # into the message, which buries the one thing you need to read. Catch a
    # misspelled adapter here instead.
    adapters = set(available_adapters(core))
    if adapters and library not in adapters:
        near = difflib.get_close_matches(library, sorted(adapters), n=3,
                                         cutoff=0.6)
        hint = f" Did you mean: {', '.join(near)}?" if near else ""
        return ProbeResult(library, name, False,
                           f"no adapter named {library!r} is installed.{hint}")

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


def find_driver(stand: Stand, mm_dir: Path | None = None) -> DriverLocation:
    """Locate the vendor DLL this stand's adapter needs, and judge the location.

    Ti2 wants its DLL copied beside the adapter; the older Ti finds its own on
    the system path from the vendor's install directory. Both places are
    searched either way, but only the right place counts as satisfied.
    """
    dll = stand.driver.dll
    mm_dir = mm_dir if mm_dir is not None else mm_install()
    if mm_dir:
        for hit in Path(mm_dir).rglob(dll):
            return DriverLocation(dll, hit, beside_adapter=True, satisfied=True)
    sdk = Path(stand.driver.sdk_path) / dll
    try:
        if sdk.exists():
            # Enough for the Ti (system path); not enough for the Ti2.
            return DriverLocation(dll, sdk, beside_adapter=False,
                                  satisfied=not stand.driver.beside_adapter)
    except OSError:
        pass
    return DriverLocation(dll)


# The Micro-Manager GUI ships the Hardware Configuration Wizard. `mmcore
# install` fetches a build that may or may not include it, so check rather
# than assume — the wizard is the right tool when a device needs pre-init
# properties (a COM port, a camera model) that cannot be guessed.
GUI_LAUNCHERS = ("ImageJ.exe", "micromanager.exe", "Micro-Manager.exe",
                 "ImageJ-win64.exe")


def gui_launcher(mm_dir: Path | None = None) -> Path | None:
    """Micro-Manager's GUI executable, if this install has one."""
    mm_dir = mm_dir if mm_dir is not None else mm_install()
    if not mm_dir:
        return None
    for name in GUI_LAUNCHERS:
        candidate = Path(mm_dir) / name
        try:
            if candidate.exists():
                return candidate
        except OSError:
            pass
    return None


# What Windows' own loader errors mean for a Micro-Manager device adapter.
# MMCore reports only "Failed to load device adapter", which is the same
# message for every one of these, and they have completely different fixes.
WINERROR_HINTS = {
    126: ("a DLL it depends on could not be found — for a Nikon adapter that "
          "is almost always the vendor driver DLL not sitting beside it"),
    127: ("an entry point is missing — the vendor DLL is a different version "
          "from the one this adapter was built against"),
    193: ("wrong architecture — a 32-bit DLL under a 64-bit Micro-Manager, "
          "or the reverse"),
    1114: ("the DLL loaded but its initialisation routine failed — usually a "
           "vendor SDK version mismatch, or hardware that is powered off"),
}


def dll_probe(path: Path, search_dir: Path | None = None) -> tuple[bool, str]:
    """Ask Windows itself to load a DLL, and report what it says.

    MMCore swallows the operating system's error behind one generic message.
    Loading the library directly gets the real code back, which is the
    difference between "copy a file" and "you installed the wrong build".

    ``search_dir`` is added to the DLL search path first, because that is
    what MMCore does when it loads an adapter — without it, a dependency
    sitting right beside the adapter would still not be found.
    """
    import ctypes
    import os

    if os.name != "nt":
        return False, "not Windows — cannot load a .dll here"
    p = Path(path)
    if not p.exists():
        return False, f"{p} does not exist"

    cookie = None
    try:
        if search_dir and hasattr(os, "add_dll_directory"):
            try:
                cookie = os.add_dll_directory(str(search_dir))
            except OSError:
                cookie = None
        ctypes.WinDLL(str(p))
        return True, ""
    except OSError as exc:
        code = getattr(exc, "winerror", None)
        detail = getattr(exc, "strerror", None) or str(exc)
        hint = WINERROR_HINTS.get(code, "")
        text = f"WinError {code}: {detail}" if code else str(exc)
        return False, (f"{text} — {hint}" if hint else text)
    except Exception as exc:                            # noqa: BLE001
        return False, str(exc)
    finally:
        if cookie is not None:
            cookie.close()


def adapter_dll(stand: Stand, mm_dir: Path | None = None) -> Path | None:
    """Where Micro-Manager's adapter library for this stand should be."""
    mm_dir = mm_dir if mm_dir is not None else mm_install()
    if not mm_dir:
        return None
    candidate = Path(mm_dir) / f"mmgr_dal_{stand.adapter}.dll"
    try:
        return candidate if candidate.exists() else None
    except OSError:
        return None


def deep_check(stand: Stand, mm_dir: Path | None = None) -> list[str]:
    """Why won't this adapter load? Ask the OS rather than guess.

    Only meaningful on Windows, and only worth running when the adapter has
    already refused to produce devices.
    """
    import os

    mm_dir = mm_dir if mm_dir is not None else mm_install()
    out: list[str] = []
    if os.name != "nt":
        return ["deep DLL check only runs on Windows"]

    found = find_driver(stand, mm_dir)
    dll = stand.driver.dll
    if not found.path:
        out.append(f"{dll}: not on this machine — {stand.driver.fix(mm_dir)}")
    else:
        ok, why = dll_probe(found.path, found.path.parent)
        out.append(f"{dll} at {found.path}: "
                   + ("loads cleanly" if ok else f"WILL NOT LOAD — {why}"))
        if not found.satisfied:
            out.append(f"…but it is not in {mm_dir}, which is where the "
                       f"{stand.adapter} adapter looks. Copy it there "
                       f"(`nikon-control-scope fix-driver`).")

    adapter = adapter_dll(stand, mm_dir)
    if not adapter:
        out.append(f"mmgr_dal_{stand.adapter}.dll is not in {mm_dir}")
        return out
    ok, why = dll_probe(adapter, mm_dir)
    out.append(f"mmgr_dal_{stand.adapter}.dll: "
               + ("loads cleanly — the adapter itself is fine" if ok
                  else f"WILL NOT LOAD — {why}"))
    return out


def install_driver(stand: Stand, mm_dir: Path | None = None,
                   source: Path | None = None) -> tuple[bool, str]:
    """Copy the vendor DLL to where the adapter looks for it.

    Copy, never move: the vendor's own software still needs its copy.
    """
    import shutil

    mm_dir = mm_dir if mm_dir is not None else mm_install()
    if not mm_dir:
        return False, "no Micro-Manager installation found"
    src = Path(source) if source else Path(stand.driver.sdk_path) / stand.driver.dll
    if not src.exists():
        return False, (f"{src} does not exist — install "
                       f"{stand.driver.installer}, or pass --from PATH")
    dest = Path(mm_dir) / stand.driver.dll
    if dest.exists():
        return True, f"{dest} is already there"
    try:
        shutil.copy2(src, dest)
    except OSError as exc:
        return False, (f"could not copy to {dest}: {exc}. Micro-Manager's "
                       f"folder may need an elevated prompt.")
    return True, f"copied {src} -> {dest}"


def stand_status(core=None, mm_dir: Path | None = None) -> list[StandStatus]:
    """Scan every known Nikon stand generation on this machine."""
    core = core or new_core()
    mm_dir = mm_dir if mm_dir is not None else mm_install()
    adapters = set(available_adapters(core))
    out: list[StandStatus] = []
    for s in STANDS:
        if s.adapter not in adapters:
            out.append(StandStatus(stand=s, installed=False))
            continue
        scan = scan_adapter(s.adapter, core)
        out.append(StandStatus(
            stand=s,
            installed=True,
            devices=scan.devices,
            error=scan.error,
            driver=find_driver(s, mm_dir),
            roles=resolve_roles(scan.devices),
        ))
    return out


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


def nikon_readiness(core=None, mm_dir: Path | None = None) -> list[str]:
    """Findings worth knowing before a bring-up, for whichever stand is here.

    Encodes the failure modes that cost the most time on this hardware: the
    vendor SDK DLL not where the adapter can find it, the two stands' very
    different meanings for an empty device list, and the fact that no stand
    adapter ever provides the camera.
    """
    core = core or new_core()
    mm_dir = mm_dir if mm_dir is not None else mm_install()
    notes: list[str] = []

    statuses = stand_status(core, mm_dir)
    present = [s for s in statuses if s.installed]
    if not present:
        notes.append(
            "No Nikon stand adapter found (looked for "
            + ", ".join(s.adapter for s in STANDS)
            + "). This is not a Micro-Manager build that can drive the stand."
        )
    for st in present:
        notes.extend(st.diagnosis(mm_dir))
        notes.extend(st.stand.notes)
    if present:
        notes.extend(SHARED_NOTES)

    adapters = set(available_adapters(core))
    cams = [a for a in CAMERA_ADAPTERS if a in adapters]
    notes.append(
        f"Camera adapter(s) present: {', '.join(cams)}." if cams else
        "No common sCMOS camera adapter (Hamamatsu/Photometrics/Andor) found "
        "— the stand adapter does not provide the camera, it needs its own."
    )
    return notes
