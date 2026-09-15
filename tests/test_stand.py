"""Tests for stand identification and role resolution.

No pymmcore needed: :mod:`nikon_control.scope.stand` is pure. The Ti device
list below is the real output of ``nikon-control-scope devices NikonTI`` on
the lab's Windows machine, so these tests pin the behaviour against hardware
rather than against an invention.
"""
from dataclasses import dataclass

from nikon_control.scope import stand


@dataclass
class Dev:
    name: str
    type: str


# Verbatim from `nikon-control-scope devices NikonTI` on the rig.
TI_DEVICES = [
    Dev("TIScope", "Hub"),
    Dev("TIAnalyzer", "Generic"),
    Dev("TIEpiShutter", "Shutter"),
    Dev("TIDiaShutter", "Shutter"),
    Dev("TIAuxShutter", "Shutter"),
    Dev("TIDiaLamp", "Shutter"),
    Dev("TINosePiece", "State"),
    Dev("TICondenserCassette", "State"),
    Dev("TIBarrierFilterWheel", "State"),
    Dev("TIExcitationFilterWheel", "State"),
    Dev("TIFilterBlock1", "State"),
    Dev("TIFilterBlock2", "State"),
    Dev("TILightPath", "State"),
    Dev("TIZDrive", "Stage"),
    Dev("TIXYDrive", "XYStage"),
    Dev("TIPFSOffset", "Stage"),
    Dev("TIPFSStatus", "AutoFocus"),
    Dev("TITIRF", "Stage"),
]

# The Ti2 adapter names its devices dynamically, so these are plausible-shape
# names rather than a transcript — which is the point: resolution must work
# off device TYPE, not off names we had to know in advance.
TI2_DEVICES = [
    Dev("*Ti2-E__0: Nikon Ti2 microscope", "Hub"),
    Dev("XYStage", "XYStage"),
    Dev("ZDrive", "Stage"),
    Dev("PFSOffset", "Stage"),
    Dev("PFStatus", "AutoFocus"),
    Dev("Nosepiece", "State"),
    Dev("FilterTurret1", "State"),
    Dev("LightPath", "State"),
    Dev("EpiShutter", "Shutter"),
]


def test_registry_covers_both_generations():
    assert set(stand.STANDS_BY_ADAPTER) == {"NikonTi2", "NikonTI"}
    assert stand.stand_for_adapter("NikonTI") is stand.TI
    assert stand.stand_for_adapter("NikonTi2") is stand.TI2
    assert stand.stand_for_adapter("DemoCamera") is None


def test_ti_roles_resolve_from_the_real_device_list():
    roles = stand.resolve_roles(TI_DEVICES)
    assert roles["xystage"] == "TIXYDrive"
    assert roles["focus"] == "TIZDrive"
    assert roles["autofocus"] == "TIPFSStatus"
    assert roles["pfsoffset"] == "TIPFSOffset"
    assert roles["nosepiece"] == "TINosePiece"
    assert roles["lightpath"] == "TILightPath"
    assert not stand.missing_roles(roles)


def test_focus_is_not_the_pfs_offset_or_the_tirf_drive():
    """Three devices report type Stage; only one of them is the focus drive."""
    roles = stand.resolve_roles(TI_DEVICES)
    assert roles["focus"] not in {"TIPFSOffset", "TITIRF"}


def test_nosepiece_is_not_a_filter_turret():
    roles = stand.resolve_roles(TI_DEVICES)
    assert roles["nosepiece"] == "TINosePiece"
    roles2 = stand.resolve_roles(TI2_DEVICES)
    assert roles2["nosepiece"] == "Nosepiece"


def test_shutter_prefers_epi_over_the_dia_lamp():
    """TIDiaLamp is typed Shutter and has crashed MM; don't pick it."""
    assert stand.resolve_roles(TI_DEVICES)["shutter"] == "TIEpiShutter"


def test_ti2_roles_resolve_without_hardcoded_names():
    roles = stand.resolve_roles(TI2_DEVICES)
    assert roles["xystage"] == "XYStage"
    assert roles["focus"] == "ZDrive"
    assert roles["autofocus"] == "PFStatus"
    assert roles["pfsoffset"] == "PFSOffset"
    assert not stand.missing_roles(roles)


def test_both_stands_report_the_same_role_vocabulary():
    """The whole point: control code never learns which stand it is on."""
    ti = stand.resolve_roles(TI_DEVICES)
    ti2 = stand.resolve_roles(TI2_DEVICES)
    assert set(ti) == set(ti2) == set(stand.ROLES)


def test_empty_device_list_means_different_things_per_stand():
    assert "SDK could not be reached" in stand.TI2.empty_list_meaning
    assert "fixed device list" in stand.TI.empty_list_meaning


def test_no_devices_resolves_no_roles_rather_than_raising():
    roles = stand.resolve_roles([])
    assert roles == {}
    assert stand.missing_roles(roles) == list(stand.ROLES)


def test_driver_fix_text_matches_how_each_dll_is_found():
    # Ti2's DLL must be copied next to the adapter...
    assert "copy" in stand.TI2.driver.fix("C:/mm")
    assert "C:/mm" in stand.TI2.driver.fix("C:/mm")
    # ...the older Ti's is found on the system path from its own install dir.
    ti_fix = stand.TI.driver.fix("C:/mm")
    assert "copy" not in ti_fix
    assert r"C:\Program Files\Nikon\Shared\Bin" in ti_fix


def test_describe_roles_lists_every_role_even_when_absent():
    lines = stand.describe_roles({"xystage": "TIXYDrive"})
    assert len(lines) == len(stand.ROLES)
    assert any("TIXYDrive" in ln for ln in lines)
    assert any("none" in ln for ln in lines)
