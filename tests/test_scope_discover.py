"""Tests for Micro-Manager device discovery.

These need Micro-Manager's device adapters. The demo adapters ship with
pymmcore-plus's `mmcore install` on every platform, so they are the one thing
we can exercise without the microscope; anything needing real hardware is out
of scope here and belongs on the rig.
"""
from pathlib import Path

import pytest

pytest.importorskip("pymmcore_plus")

from nikon_control.scope import discover  # noqa: E402

_ADAPTERS = discover.available_adapters()
needs_demo = pytest.mark.skipif(
    "DemoCamera" not in _ADAPTERS,
    reason="Micro-Manager demo adapters not installed (run: mmcore install)",
)


def test_device_type_names_are_readable():
    assert discover.device_type_name(2) == "Camera"
    assert discover.device_type_name(6) == "XYStage"
    assert discover.device_type_name(9) == "AutoFocus"
    # an int we don't know must not raise
    assert discover.device_type_name(9999)


def test_adapter_list_is_sorted_and_stringy():
    assert _ADAPTERS == sorted(_ADAPTERS)
    assert all(isinstance(a, str) for a in _ADAPTERS)


def test_unknown_adapter_returns_empty_rather_than_raising():
    assert discover.adapter_devices("NoSuchAdapter") == []


@needs_demo
def test_lists_devices_of_an_adapter_with_types():
    entries = discover.adapter_devices("DemoCamera")
    by_name = {e.name: e for e in entries}
    assert "DXYStage" in by_name and "DCam" in by_name
    assert by_name["DXYStage"].type == "XYStage"
    assert by_name["DCam"].type == "Camera"
    assert by_name["DCam"].library == "DemoCamera"


@needs_demo
def test_probe_connects_a_present_device():
    res = discover.probe("DemoCamera", "DXYStage")
    assert res.ok and not res.error
    assert "connected" in res.describe()


@needs_demo
def test_probe_reports_failure_instead_of_raising():
    """A bring-up wants the whole picture, not the first exception."""
    res = discover.probe("DemoCamera", "NoSuchDevice")
    assert res.ok is False
    assert "load failed" in res.error
    assert "FAILED" in res.describe()


@needs_demo
def test_probe_can_read_properties():
    res = discover.probe("DemoCamera", "DCam", read_properties=True)
    assert res.ok
    assert res.properties          # a demo camera has plenty
    assert all(isinstance(k, str) for k in res.properties)


@needs_demo
def test_probe_isolation_a_failure_does_not_break_the_next_probe():
    discover.probe("DemoCamera", "NoSuchDevice")
    assert discover.probe("DemoCamera", "DXYStage").ok


@needs_demo
def test_config_inspection_reports_devices_and_roles(tmp_path):
    from pathlib import Path

    install = discover.mm_install()
    cfgs = sorted(Path(install).rglob("MMConfig_demo.cfg"))
    if not cfgs:
        pytest.skip("no demo config shipped")
    core = discover.load_config(cfgs[0])
    devices = discover.loaded_devices(core)
    labels = {d.label for d in devices}
    assert {"Camera", "XY", "Z"} <= labels
    assert "Core" not in labels          # not a hardware device

    roles = discover.core_roles(core)
    assert roles["camera"] == "Camera"
    assert roles["xystage"] == "XY"
    assert roles["focus"] == "Z"
    # roles are annotated back onto the device entries
    by_label = {d.label: d for d in devices}
    assert by_label["XY"].role == "xystage"
    assert by_label["Camera"].role == "camera"


def test_nikon_readiness_reports_rather_than_raises():
    notes = discover.nikon_readiness()
    assert notes and all(isinstance(n, str) for n in notes)
    # it must always say something about the stand adapter and the camera
    joined = " ".join(notes).lower()
    assert "nikon" in joined
    assert "camera" in joined


def test_scan_adapter_distinguishes_missing_from_empty():
    """'Not installed' and 'installed but silent' have different fixes."""
    scan = discover.scan_adapter("NoSuchAdapter")
    assert scan.installed is False
    assert scan.devices == []
    assert "not installed" in scan.error


def test_probe_does_not_dump_the_adapter_list_on_a_typo():
    """MMCore's own error pastes all ~265 adapter names; ours must not."""
    res = discover.probe("NikonTi8", "TIXYDrive")
    assert res.ok is False
    assert "no adapter named 'NikonTi8'" in res.error
    assert len(res.error) < 300
    assert "DemoCamera" not in res.error or "Did you mean" in res.error


def test_ti2_diagnosis_explains_an_empty_list_and_names_the_dll():
    """The rig's actual failure: adapter present, SDK DLL missing."""
    from nikon_control.scope.stand import TI2

    st = discover.StandStatus(stand=TI2, installed=True, devices=[],
                              driver=discover.DriverLocation("Ti2_Mic_Driver.dll"),
                              roles={})
    text = " ".join(st.diagnosis("C:/mm"))
    assert "offers no devices" in text
    assert "SDK could not be reached" in text      # not "adapter missing"
    assert "Ti2_Mic_Driver.dll" in text
    assert r"C:\Program Files\Nikon\Ti2-SDK\bin" in text   # where to get it
    assert st.usable is False


def test_ti_diagnosis_warns_that_a_device_list_proves_nothing():
    from nikon_control.scope.stand import TI

    entries = [discover.DeviceEntry("NikonTI", n, type=t) for n, t in [
        ("TIXYDrive", "XYStage"), ("TIZDrive", "Stage"),
        ("TIPFSOffset", "Stage"), ("TIPFSStatus", "AutoFocus"),
        ("TINosePiece", "State"), ("TILightPath", "State"),
        ("TIEpiShutter", "Shutter"),
    ]]
    st = discover.StandStatus(
        stand=TI, installed=True, devices=entries,
        driver=discover.DriverLocation("NikonTi.dll"),
        roles=discover.resolve_roles(entries),
    )
    text = " ".join(st.diagnosis("C:/mm"))
    assert "7 device(s)" in text
    assert "proves nothing about what is connected" in text
    assert "NikonTi.dll" in text
    # every role resolved, but the driver is missing, so it is not usable yet
    assert st.usable is True        # roles are complete
    assert "NOT found" in text      # and the driver problem is still reported


def test_readiness_covers_both_stand_generations():
    notes = " ".join(discover.nikon_readiness())
    assert "NikonTi2" in notes and "NikonTI" in notes


def test_the_pfs_trap_is_recorded_for_whichever_stand_is_attached():
    """Moving Z kills PFS on both generations — control code must re-engage."""
    text = " ".join(discover.SHARED_NOTES)
    assert "DISABLES PFS" in text
    assert "re-engage" in text


def test_dll_in_the_sdk_folder_is_not_enough_for_the_ti2():
    """The rig's real state: DLL present on the machine, wrong folder.

    A plain "found it" would have sent us looking elsewhere for the failure.
    """
    from nikon_control.scope.stand import TI, TI2

    sdk = Path(r"C:\Program Files\Nikon\Ti2-SDK\bin\Ti2_Mic_Driver.dll")
    loc = discover.DriverLocation("Ti2_Mic_Driver.dll", sdk,
                                  beside_adapter=False, satisfied=False)
    text = " ".join(discover.StandStatus(TI2, True, [], "", loc, {})
                    .diagnosis("C:/mm"))
    assert "is on this machine at" in text
    assert "COPY it to C:/mm" in text

    # The older Ti loads its DLL from the system path, so the same location
    # IS enough there — the two stands must not share one verdict.
    ok = discover.DriverLocation("NikonTi.dll",
                                 Path(r"C:\Program Files\Nikon\Shared\Bin\NikonTi.dll"),
                                 beside_adapter=False, satisfied=True)
    ti_text = " ".join(discover.StandStatus(TI, True, [], "", ok, {})
                       .diagnosis("C:/mm"))
    assert "found at" in ti_text
    assert "COPY" not in ti_text
