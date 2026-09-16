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

# Verbatim from `nikon-control-scope build` on the lab's Ti2-E, once the
# vendor DLL was in place. The Ti2 adapter names its devices dynamically, so
# this list could not have been known in advance — which is exactly why
# resolution works off device TYPE rather than names.
TI2_DEVICES = [
    Dev("Ti2-E__0", "Hub"),
    Dev("ZDrive", "Stage"),
    Dev("XYStage", "XYStage"),
    Dev("Nosepiece", "State"),
    Dev("CondenserTurret", "State"),
    Dev("FilterTurret1", "State"),
    Dev("Turret1Shutter", "Shutter"),
    Dev("FilterTurret2", "State"),
    Dev("Turret2Shutter", "Shutter"),
    Dev("LightPath", "State"),
    Dev("PFS", "AutoFocus"),
    Dev("PFSOffset", "Stage"),
    Dev("IntermediateMagnification", "Magnifier"),
    Dev("DiaLamp", "Shutter"),
    Dev("TIRF1", "XYStage"),
    Dev("TIRF2", "XYStage"),
    Dev("TIRF3", "XYStage"),
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
    assert roles["autofocus"] == "PFS"
    assert roles["pfsoffset"] == "PFSOffset"
    assert roles["nosepiece"] == "Nosepiece"
    assert roles["lightpath"] == "LightPath"
    assert not stand.missing_roles(roles)


def test_the_stage_is_not_a_tirf_positioner():
    """Regression, from the rig: the stage read 0,0 and would not move.

    A Ti2 types FOUR devices as XYStage — the stage and three TIRF
    illuminator positioners — and an alphabetical tie-break picked TIRF1.
    Driving that looks exactly like broken stage hardware.
    """
    xy_typed = [d.name for d in TI2_DEVICES if d.type == "XYStage"]
    assert xy_typed == ["XYStage", "TIRF1", "TIRF2", "TIRF3"]
    assert stand.resolve_roles(TI2_DEVICES)["xystage"] == "XYStage"
    # and a TIRF drive must never be offered as a candidate at all
    assert stand.role_choices(TI2_DEVICES)["xystage"] == ["XYStage"]


def test_the_dia_lamp_is_the_brightfield_shutter_on_a_ti2():
    """Three Shutter devices; the transmitted lamp is the brightfield one."""
    roles = stand.resolve_roles(TI2_DEVICES)
    assert roles["shutter"] == "DiaLamp"
    # the epi shutters are still offered, for fluorescence later
    assert set(stand.role_choices(TI2_DEVICES)["shutter"]) == {
        "DiaLamp", "Turret1Shutter", "Turret2Shutter"}


def test_ambiguous_roles_are_reported_so_a_wrong_pick_cannot_hide():
    ambiguous = stand.ambiguous_roles(TI2_DEVICES)
    # the condenser turret is a State device like the nosepiece
    assert ambiguous["nosepiece"][0] == "Nosepiece"
    assert "CondenserTurret" in ambiguous["nosepiece"]
    # a role with exactly one candidate is not ambiguous
    assert "xystage" not in ambiguous
    assert "focus" not in ambiguous


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
