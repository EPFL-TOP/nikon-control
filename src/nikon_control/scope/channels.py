"""Define imaging channels as Micro-Manager configuration presets.

A "channel" in Micro-Manager is not a special object — it is a **config
group** holding presets, and each preset is a list of
``device, property, value`` settings applied together. Selecting a channel
means applying one preset. In a ``.cfg`` that looks like::

    ConfigGroup,Channel,BF,DiaLamp,State,1
    ConfigGroup,Channel,BF,LightPath,Label,L100
    ConfigGroup,Channel,GFP,Turret1Shutter,State,1
    ConfigGroup,Channel,GFP,FilterTurret1,Label,GFP-HQ

Writing those by hand means knowing every device's property names and legal
values. It is far easier — and far less error-prone — to **set the
microscope up by eye and capture what it is doing**, which is what
:func:`capture` does and what Micro-Manager's own GUI does behind its
"Group/Preset" editor.

Only the devices that actually define a channel are captured: shutters,
filter and condenser turrets, the light path, and the lamp intensity. The
stage and focus are deliberately excluded — a channel should not teleport
the sample.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

DEFAULT_GROUP = "Channel"

# Device types whose setting is part of "which channel is this".
CHANNEL_TYPES = ("Shutter", "State")
# The property that carries a State device's setting. Label is preferred:
# "GFP-HQ" survives someone re-ordering a filter turret, "3" does not.
STATE_PROPERTIES = ("Label", "State")
# Never captured into a channel, whatever their type.
EXCLUDE_TYPES = ("XYStage", "Stage", "Camera", "Hub", "AutoFocus")
# Nor these roles. The objective turret is typed State like a filter wheel,
# but a channel that rotates the turret would swing an objective under a
# loaded plate every time someone switched from BF to GFP.
EXCLUDE_ROLES = ("nosepiece", "xystage", "focus", "pfsoffset", "autofocus")


@dataclass(frozen=True)
class Setting:
    device: str
    property: str
    value: str

    def line(self, group: str, preset: str) -> str:
        return f"ConfigGroup,{group},{preset},{self.device},{self.property}," \
               f"{self.value}"


@dataclass
class Preset:
    name: str
    settings: list[Setting] = field(default_factory=list)

    def lines(self, group: str) -> list[str]:
        return [s.line(group, self.name) for s in self.settings]

    def describe(self) -> str:
        body = ", ".join(f"{s.device}.{s.property}={s.value}"
                         for s in self.settings)
        return f"{self.name}: {body or '(nothing captured)'}"


def capture(scope, name: str, *, with_exposure: bool = False,
            include: list[str] | None = None) -> Preset:
    """Record what the microscope is doing right now as a named preset.

    ``scope`` is a :class:`~nikon_control.scope.control.Scope`. Set the light
    path, shutters, filters and intensity by eye, then capture — no property
    names to look up and no illegal values to guess.
    """
    settings: list[Setting] = []
    for label in (include if include is not None else _channel_devices(scope)):
        prop = _setting_property(scope, label)
        if prop is None:
            continue
        try:
            value = scope.get_property(label, prop)
        except Exception:
            continue
        settings.append(Setting(label, prop, value))

    info = scope.intensity_property()
    if info is not None and not any(s.device == info.device
                                    and s.property == info.name
                                    for s in settings):
        settings.append(Setting(info.device, info.name, info.value))

    if with_exposure and scope.has("camera"):
        try:
            settings.append(Setting(scope.roles["camera"], "Exposure",
                                    f"{scope.exposure_ms():g}"))
        except Exception:
            pass
    return Preset(name, settings)


def _channel_devices(scope) -> list[str]:
    barred = {scope.roles[r] for r in EXCLUDE_ROLES if scope.roles.get(r)}
    out: list[str] = []
    for label in scope.device_labels():
        if label in barred:
            continue
        try:
            dtype = _device_type(scope, label)
        except Exception:
            continue
        if dtype in EXCLUDE_TYPES or dtype not in CHANNEL_TYPES:
            continue
        out.append(label)
    return out


def _device_type(scope, label: str) -> str:
    from .discover import device_type_name

    return device_type_name(scope.core.getDeviceType(label))


def _setting_property(scope, label: str) -> str | None:
    names = {p.name for p in scope.properties(label)}
    for candidate in STATE_PROPERTIES:
        if candidate in names:
            return candidate
    return None


def groups(scope) -> dict[str, list[str]]:
    """Config groups already defined, and the presets in each."""
    out: dict[str, list[str]] = {}
    try:
        for g in scope.core.getAvailableConfigGroups():
            out[str(g)] = [str(c) for c in
                           scope.core.getAvailableConfigs(str(g))]
    except Exception:
        pass
    return out


def append_to_config(path, preset: Preset, group: str = DEFAULT_GROUP) -> str:
    """Add a preset to an existing ``.cfg``, replacing one of the same name.

    Rewrites in place rather than appending blindly, so capturing "BF" twice
    updates it instead of leaving two conflicting definitions that the loader
    resolves by whichever it reads last.
    """
    from pathlib import Path

    p = Path(path)
    text = p.read_text() if p.exists() else ""
    kept = [ln for ln in text.splitlines()
            if not _is_preset_line(ln, group, preset.name)]

    while kept and not kept[-1].strip():
        kept.pop()
    if kept:
        kept.append("")
    kept.append(f"# Channel preset: {preset.name}")
    kept.extend(preset.lines(group))
    out = "\n".join(kept) + "\n"
    p.write_text(out)
    return out


def _is_preset_line(line: str, group: str, preset: str) -> bool:
    if line.startswith(f"# Channel preset: {preset}"):
        return True
    parts = line.split(",")
    return (len(parts) >= 3 and parts[0] == "ConfigGroup"
            and parts[1] == group and parts[2] == preset)


def read_presets(path, group: str = DEFAULT_GROUP) -> dict[str, Preset]:
    """Presets already written into a ``.cfg``."""
    from pathlib import Path

    out: dict[str, Preset] = {}
    text = Path(path).read_text() if Path(path).exists() else ""
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6 or parts[0] != "ConfigGroup" or parts[1] != group:
            continue
        preset = out.setdefault(parts[2], Preset(parts[2]))
        preset.settings.append(Setting(parts[3], parts[4],
                                       ",".join(parts[5:])))
    return out


def valid_name(name: str) -> bool:
    """A preset name has to survive a comma-separated file."""
    return bool(name) and "," not in name and bool(re.match(r"^[^\s].*", name))
