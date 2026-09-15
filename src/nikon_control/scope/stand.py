"""The two Nikon stand generations, and which device fills which role.

This lab has both: a Ti2-E driven through Micro-Manager's ``NikonTi2``
adapter, and an older Ti/Ti-E driven through ``NikonTI``. They are different
adapters with different device names, different SDKs and different failure
modes, so everything above this module should ask for a *role* ("the XY
stage") and never for a device name.

The two differ in one way that matters more than the names:

* ``NikonTI`` publishes a **fixed** device list. It will happily list
  eighteen devices on a laptop with no microscope attached — so a device
  list is not evidence of a connected stand.
* ``NikonTi2`` enumerates **dynamically** by asking Nikon's SDK what is
  actually on the stand. An empty list therefore means something real: the
  SDK could not be reached (usually a missing driver DLL), not that the
  adapter is absent.

Nothing here imports pymmcore, so the registry and the role resolution stay
testable anywhere. :mod:`.discover` is the layer that touches a core.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# The roles the acquisition code will actually ask for, in bring-up order.
ROLES = (
    "xystage",
    "focus",
    "autofocus",
    "pfsoffset",
    "nosepiece",
    "lightpath",
    "shutter",
)

ROLE_LABELS = {
    "xystage": "XY stage",
    "focus": "Z drive",
    "autofocus": "PFS (autofocus)",
    "pfsoffset": "PFS offset",
    "nosepiece": "objective turret",
    "lightpath": "light path",
    "shutter": "shutter",
}

# True for either stand: it bites the moment control code moves Z.
SHARED_NOTES = (
    "Setting the Z drive position programmatically DISABLES PFS "
    "(micro-manager#1815). Any control code that moves Z during a timelapse "
    "must re-engage PFS afterwards, or the focus silently drifts overnight.",
)


@dataclass(frozen=True)
class Driver:
    """A vendor DLL the adapter needs before it can load."""

    dll: str
    sdk_path: str           # where the vendor installer puts it
    installer: str          # what to install to get it
    beside_adapter: bool    # must it be copied next to mmgr_dal_*.dll?

    def fix(self, mm_dir) -> str:
        if self.beside_adapter:
            return (f"copy {self.dll} from {self.sdk_path} into {mm_dir} "
                    f"(do not rename it, and take it from the installed "
                    f"version of {self.installer})")
        return (f"install {self.installer}; it puts {self.dll} in "
                f"{self.sdk_path}, where the adapter finds it on the "
                f"system path")


@dataclass(frozen=True)
class Stand:
    key: str
    adapter: str
    label: str
    hub: str                # hub device name, "" when it is discovered
    dynamic: bool           # does the device list come from the SDK?
    driver: Driver
    notes: tuple[str, ...] = ()

    @property
    def empty_list_meaning(self) -> str:
        if self.dynamic:
            return ("This adapter asks Nikon's SDK what is on the stand, so "
                    "an empty device list means the SDK could not be reached "
                    "— not that the adapter is missing.")
        return ("This adapter publishes a fixed device list, so listing "
                "devices proves nothing about what is connected. Use `probe`.")


TI2 = Stand(
    key="ti2",
    adapter="NikonTi2",
    label="Ti2 / Ti2-E",
    hub="",  # appears as e.g. "*Ti2-E__0: Nikon Ti2 microscope"
    dynamic=True,
    driver=Driver(
        dll="Ti2_Mic_Driver.dll",
        sdk_path=r"C:\Program Files\Nikon\Ti2-SDK\bin",
        installer="Ti2 Control / the Ti2 SDK",
        beside_adapter=True,
    ),
    notes=(
        "Power on the microscope stand BEFORE the controller box. Get this "
        "backwards and only the simulator device appears.",
        "Ti2Sample.exe, shipped with the Ti2 SDK, tests the SDK connection "
        "without Micro-Manager — run it first to tell an SDK problem from a "
        "Micro-Manager one.",
        "Ti2 Control 2.10 and 2.20 are documented to crash Micro-Manager "
        "when the nosepiece device is added (mmCoreAndDevices#44); 2.00 is "
        "reported working.",
    ),
)

TI = Stand(
    key="ti",
    adapter="NikonTI",
    label="Ti / Ti-E (older stand)",
    hub="TIScope",
    dynamic=False,
    driver=Driver(
        dll="NikonTi.dll",
        sdk_path=r"C:\Program Files\Nikon\Shared\Bin",
        installer="Nikon's Ti Setup Tool (or the TiSDKRedist package)",
        beside_adapter=False,
    ),
    notes=(
        "Connect the stand to a USB 2 port. USB 3 is a documented source of "
        "connection failures on this generation.",
        "TIDiaLamp has crashed Micro-Manager with some driver versions — "
        "leave it out of the configuration unless you need it.",
        "PFS offset positions are 1/40 of the raw steps the Nikon driver "
        "reports, and the valid range depends on the objective in use.",
    ),
)

STANDS: tuple[Stand, ...] = (TI2, TI)
STANDS_BY_ADAPTER = {s.adapter: s for s in STANDS}
STANDS_BY_KEY = {s.key: s for s in STANDS}


def stand_for_adapter(adapter: str) -> Stand | None:
    return STANDS_BY_ADAPTER.get(adapter)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


# Role resolution works off the device TYPE first, because that is what both
# generations agree on, and uses name fragments only to choose between
# same-typed devices. That is what lets this handle the Ti2's dynamic names
# without anyone having written them down.
_NAME_HINTS: dict[str, tuple[str, ...]] = {
    "focus": ("zdrive", "focus", "z"),
    "pfsoffset": ("pfsoffset", "pfs"),
    "nosepiece": ("nosepiece", "objective", "turret"),
    "lightpath": ("lightpath", "eyepiece", "port"),
    "shutter": ("epishutter", "epi", "diashutter", "dia"),
}


def _candidates(devices, role: str) -> list:
    out = []
    for d in devices:
        dtype = str(getattr(d, "type", "") or "")
        name = _norm(getattr(d, "name", ""))
        if role == "xystage" and dtype == "XYStage":
            out.append(d)
        elif role == "autofocus" and dtype == "AutoFocus":
            out.append(d)
        elif role == "pfsoffset" and dtype == "Stage" and "pfs" in name:
            out.append(d)
        elif role == "focus" and dtype == "Stage":
            # the PFS offset and the TIRF drive are Stages too, and neither
            # is the focus drive
            if "pfs" not in name and "tirf" not in name:
                out.append(d)
        elif role == "nosepiece" and dtype == "State":
            if ("nose" in name or "objective" in name
                    or ("turret" in name and "filter" not in name)):
                out.append(d)
        elif role == "lightpath" and dtype == "State" and "light" in name:
            out.append(d)
        elif role == "shutter" and dtype == "Shutter":
            out.append(d)
    return out


def _rank(device, role: str) -> int:
    """Lower is better: position of the first matching name hint."""
    name = _norm(getattr(device, "name", ""))
    for i, hint in enumerate(_NAME_HINTS.get(role, ())):
        if hint in name:
            return i
    return len(_NAME_HINTS.get(role, ()))


def resolve_roles(devices) -> dict[str, str]:
    """Map each role to a device name, for any object with ``.name``/``.type``.

    Works on both stands and on a loaded configuration, which is the point:
    the control layer asks for ``roles["autofocus"]`` and never learns
    whether it is talking to a Ti or a Ti2.
    """
    roles: dict[str, str] = {}
    for role in ROLES:
        found = _candidates(devices, role)
        if not found:
            continue
        found.sort(key=lambda d: (_rank(d, role), str(getattr(d, "name", ""))))
        roles[role] = str(getattr(found[0], "name", ""))
    return roles


def missing_roles(roles: dict[str, str]) -> list[str]:
    return [r for r in ROLES if not roles.get(r)]


def describe_roles(roles: dict[str, str]) -> list[str]:
    width = max(len(v) for v in ROLE_LABELS.values())
    return [f"{ROLE_LABELS[r]:<{width}}  {roles.get(r) or '— none —'}"
            for r in ROLES]
