"""Tests for the well scanner — plugin 3.

The manifest is the product, not the images: without a correct per-frame
stage position, a cell found offline cannot be driven back to. So the
pixel-to-stage arithmetic is pinned hardest.
"""
import json

import pytest

from nikon_control.scope import plate as plate_mod
from nikon_control.scope import scan, timing, wells

pytest.importorskip("useq")
pytest.importorskip("tifffile")

# Their rig: 2048 px of 6.5 µm at 40× -> 0.1625 µm/px -> a 332.8 µm field.
FOV = (332.8, 332.8)
WELL96 = (6400.0, 6400.0)


@pytest.fixture
def cal():
    return plate_mod.calibrate("96-well", [
        plate_mod.WellRef("A1", 1000, 2000),
        plate_mod.WellRef("A12", 100000, 2000),
        plate_mod.WellRef("H1", 1000, -61000),
    ])


@pytest.fixture
def two_wells():
    return wells.Selection("96-well").add(["A1", "A2"])


# ---------------------------------------------------------------- geometry

def test_a_grid_is_laid_in_every_selected_well(cal, two_wells):
    sp = scan.plan(cal, two_wells, fov_um=FOV, rows=3, columns=3)
    assert sp.n_fields == 18
    assert sorted({f.well for f in sp.fields}) == ["A1", "A2"]
    assert all(f.index < 9 for f in sp.fields)


def test_fields_are_centred_on_the_well(cal):
    """A 3x3 grid's middle field sits at the well centre."""
    sp = scan.plan(cal, ["A1"], fov_um=FOV, rows=3, columns=3, overlap=0.0)
    middle = [f for f in sp.fields
              if abs(f.x - 1000) < 1 and abs(f.y - 2000) < 1]
    assert middle, [(f.x, f.y) for f in sp.fields]


def test_field_spacing_follows_the_overlap(cal):
    sp = scan.plan(cal, ["A1"], fov_um=FOV, rows=1, columns=2, overlap=0.0)
    xs = sorted(f.x for f in sp.fields)
    assert xs[1] - xs[0] == pytest.approx(FOV[0], rel=0.01)

    lapped = scan.plan(cal, ["A1"], fov_um=FOV, rows=1, columns=2, overlap=0.5)
    xs = sorted(f.x for f in lapped.fields)
    assert xs[1] - xs[0] == pytest.approx(FOV[0] * 0.5, rel=0.01)


def test_covering_a_96_well_at_40x_takes_the_expected_field_count():
    """The number that decides whether a survey is minutes or hours."""
    rows, cols = scan.fields_for_coverage(FOV, WELL96, 1.0, overlap=0.0)
    assert (rows, cols) == (20, 20)          # 6400 / 332.8 = 19.2 -> 20
    third = scan.fields_for_coverage(FOV, WELL96, 0.33, overlap=0.0)
    assert third == (7, 7)


def test_coverage_reports_when_the_grid_runs_off_the_well(cal):
    sp = scan.plan(cal, ["A1"], fov_um=FOV, rows=40, columns=40)
    assert sp.coverage(WELL96) > 1.0
    assert "past the well onto the plastic" in " ".join(sp.describe(WELL96))


def test_a_plan_with_no_wells_is_refused(cal):
    with pytest.raises(ValueError, match="no wells"):
        scan.plan(cal, [], fov_um=FOV, rows=3, columns=3)


def test_a_nonsense_grid_is_refused(cal):
    with pytest.raises(ValueError, match="at least one row"):
        scan.plan(cal, ["A1"], fov_um=FOV, rows=0, columns=3)
    with pytest.raises(ValueError, match="overlap"):
        scan.plan(cal, ["A1"], fov_um=FOV, rows=1, columns=1, overlap=1.0)


def test_wells_are_visited_in_the_selections_order(cal):
    sel = wells.Selection("96-well").add(wells.parse("96-well", "A1:B2"))
    sel.serpentine_order = True
    order = []
    for f in scan.plan(cal, sel, fov_um=FOV, rows=1, columns=1).fields:
        if f.well not in order:
            order.append(f.well)
    assert order == ["A1", "A2", "B2", "B1"], "serpentine order was lost"


# ----------------------------------------------------------------- timing

def test_time_estimate_grows_with_fields_and_refocusing(cal):
    t = timing.Timings(move_overhead_ms=100, move_ms_per_mm=50,
                       refocus_ms=400, channel_ms={"": 60},
                       channel_switch_ms=0)
    small = scan.plan(cal, ["A1"], fov_um=FOV, rows=3, columns=3)
    big = scan.plan(cal, ["A1"], fov_um=FOV, rows=19, columns=19)
    assert scan.estimate_seconds(big, t) > scan.estimate_seconds(small, t)

    every = scan.estimate_seconds(big, t, refocus_every=1)
    never = scan.estimate_seconds(big, t, refocus_every=0)
    assert never < every
    # refocusing 361 times at 400 ms is 144 s of the difference
    assert every - never == pytest.approx(361 * 0.4, rel=0.05)


# ------------------------------------------------ pixel -> stage, the point

def test_a_pixel_maps_back_to_a_stage_coordinate():
    manifest = {"pixel_size_um": 0.1625}
    frame = {"x": 1000.0, "y": 2000.0, "shape": [2048, 2048]}

    # the middle of the frame is the frame's own stage position
    assert scan.stage_of_pixel(frame, 1024, 1024, manifest) == \
        pytest.approx((1000.0, 2000.0))

    # one pixel right is +0.1625 µm in x
    x, _y = scan.stage_of_pixel(frame, 1025, 1024, manifest)
    assert x == pytest.approx(1000.1625)

    # image rows increase DOWNWARDS, stage y increases upwards
    _x, y = scan.stage_of_pixel(frame, 1024, 1025, manifest)
    assert y == pytest.approx(1999.8375)


def test_pixel_mapping_covers_the_whole_field():
    manifest = {"pixel_size_um": 0.1625}
    frame = {"x": 0.0, "y": 0.0, "shape": [2048, 2048]}
    corner = scan.stage_of_pixel(frame, 0, 0, manifest)
    assert corner[0] == pytest.approx(-1024 * 0.1625)
    assert corner[1] == pytest.approx(+1024 * 0.1625)


def test_a_scan_with_no_pixel_size_refuses_to_invent_one():
    """Silently guessing would put every revisit in the wrong place."""
    with pytest.raises(ValueError, match="no pixel size"):
        scan.stage_of_pixel({"x": 0, "y": 0, "shape": [10, 10]}, 1, 1,
                            {"pixel_size_um": 0})


# ----------------------------------------------------- acquiring, for real

pytest.importorskip("pymmcore_plus")
from nikon_control.scope import discover  # noqa: E402
from nikon_control.scope.control import Scope  # noqa: E402

needs_demo = pytest.mark.skipif(
    "DemoCamera" not in discover.available_adapters(),
    reason="Micro-Manager demo adapters not installed (run: mmcore install)",
)


@needs_demo
def test_a_scan_writes_frames_and_a_usable_manifest(tmp_path, cal, two_wells):
    scope = Scope.demo()
    scope.set_pixel_size_um(0.1625)
    sp = scan.plan(cal, two_wells, fov_um=scope.fov_um(), rows=2, columns=2)

    result = scan.run(scope, sp, tmp_path / "frames")

    assert len(result.frames) == 8
    assert not result.failures
    manifest = scan.load_manifest(tmp_path / "frames")
    assert manifest["kind"] == "well-scan"
    assert manifest["pixel_size_um"] == pytest.approx(0.1625)
    assert manifest["plate"] == "96-well"
    assert manifest["wells"] == ["A1", "A2"]

    for entry in manifest["frames"]:
        assert (tmp_path / "frames" / entry["file"]).exists()
        assert entry["shape"] and entry["x"] is not None
    # and every frame's position is distinct — a grid, not one spot
    assert len({(f["x"], f["y"]) for f in manifest["frames"]}) == 8


@needs_demo
def test_the_stage_actually_visits_each_field(tmp_path, cal):
    scope = Scope.demo()
    scope.set_pixel_size_um(0.1625)
    sp = scan.plan(cal, ["A1"], fov_um=scope.fov_um(), rows=1, columns=3)

    scan.run(scope, sp, tmp_path / "frames")

    here = scope.xy()
    last = sp.fields[-1]
    assert round(here.x) == round(last.x)
    assert round(here.y) == round(last.y)


@needs_demo
def test_one_bad_field_does_not_lose_the_scan(tmp_path, cal):
    """361 fields must not be thrown away because one could not be acquired."""
    scope = Scope.demo()
    scope.set_pixel_size_um(0.1625)
    sp = scan.plan(cal, ["A1"], fov_um=scope.fov_um(), rows=1, columns=4)

    calls = {"n": 0}
    real_snap = scope.snap

    def flaky():
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("camera timed out")
        return real_snap()

    scope.snap = flaky
    result = scan.run(scope, sp, tmp_path / "frames")

    assert len(result.frames) == 3
    assert len(result.failures) == 1
    assert "camera timed out" in result.failures[0]["error"]
    # the failure is in the manifest, with where it happened
    manifest = scan.load_manifest(tmp_path / "frames")
    assert manifest["failures"][0]["x"] is not None


@needs_demo
def test_a_stopped_scan_leaves_a_usable_partial_dataset(tmp_path, cal):
    scope = Scope.demo()
    scope.set_pixel_size_um(0.1625)
    sp = scan.plan(cal, ["A1"], fov_um=scope.fov_um(), rows=2, columns=4)

    seen = {"n": 0}

    def should_stop():
        seen["n"] += 1
        return seen["n"] > 3

    result = scan.run(scope, sp, tmp_path / "frames", should_stop=should_stop)

    assert 0 < len(result.frames) < sp.n_fields
    manifest = scan.load_manifest(tmp_path / "frames")
    assert len(manifest["frames"]) == len(result.frames)
    for entry in manifest["frames"]:
        assert (tmp_path / "frames" / entry["file"]).exists()


@needs_demo
def test_run_iter_yields_between_fields(tmp_path, cal):
    """A GUI drives this so a 361-field scan does not freeze the page."""
    scope = Scope.demo()
    scope.set_pixel_size_um(0.1625)
    sp = scan.plan(cal, ["A1"], fov_um=scope.fov_um(), rows=1, columns=3)

    counts = [len(r.frames)
              for r in scan.run_iter(scope, sp, tmp_path / "frames")]
    assert counts == [1, 2, 3, 3]          # one per field, then the final


@needs_demo
def test_a_scan_records_the_objective_it_was_taken_with(tmp_path, cal):
    """A 40x scan and a 10x scan are not interchangeable downstream."""
    scope = Scope.demo()
    scope.set_pixel_size_um(0.1625)
    sp = scan.plan(cal, ["A1"], fov_um=scope.fov_um(), rows=1, columns=1)
    scan.run(scope, sp, tmp_path / "frames")
    assert scan.load_manifest(tmp_path / "frames")["objective"]
