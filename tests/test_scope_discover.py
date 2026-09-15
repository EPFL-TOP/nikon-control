"""Tests for Micro-Manager device discovery.

These need Micro-Manager's device adapters. The demo adapters ship with
pymmcore-plus's `mmcore install` on every platform, so they are the one thing
we can exercise without the microscope; anything needing real hardware is out
of scope here and belongs on the rig.
"""
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
