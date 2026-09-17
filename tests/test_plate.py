"""Tests for well-plate registration.

The important ones are round trips against useq's own forward model: we
generate measurements from a known plate placement, recover it, and check we
get the original numbers back. That way the fit can never drift from useq's
conventions (row direction, mm vs µm, where rotation is centred).
"""
import math

import pytest

from nikon_control.scope import plate
from nikon_control.scope.plate import (
    SCALE_TOLERANCE,
    PlateCalibration,
    WellRef,
    calibrate,
    nominal_offsets,
    suggested_refs,
)


def _measure(plate, a1, rotation, well_names):
    """Ground-truth stage positions from useq for a known plate placement."""
    import numpy as np
    from useq import WellPlatePlan

    plan = WellPlatePlan(plate=plate, a1_center_xy=a1, rotation=rotation)
    names = np.asarray(plan.all_well_names).ravel()
    lookup = {str(n): p for n, p in zip(names, plan.all_well_positions)}
    return [WellRef(n, float(lookup[n].x), float(lookup[n].y))
            for n in well_names]


# ---- nominal geometry ----------------------------------------------------

def test_nominal_offsets_match_the_plate_definition():
    off = nominal_offsets("96-well")
    assert off["A1"] == (0.0, 0.0)
    # 96-well spacing is 9 mm; stage units are µm
    assert off["A2"][0] == pytest.approx(9000.0)
    assert abs(off["B1"][1]) == pytest.approx(9000.0)
    assert len(off) == 96


def test_suggested_refs_are_the_far_corners():
    assert suggested_refs("96-well") == ("A1", "A12", "H1")
    assert suggested_refs("384-well") == ("A1", "A24", "P1")


# ---- round trips ---------------------------------------------------------

@pytest.mark.parametrize("a1,rot", [
    ((0.0, 0.0), 0.0),
    ((1234.5, -6789.0), 0.0),
    ((1000.0, 2000.0), 2.0),
    ((-500.0, 400.0), -3.75),
    ((10.0, 20.0), 17.5),
])
def test_recovers_a_known_placement(a1, rot):
    refs = _measure("96-well", a1, rot, ["A1", "A12", "H1"])
    cal = calibrate("96-well", refs)
    assert cal.a1_center_xy[0] == pytest.approx(a1[0], abs=1e-6)
    assert cal.a1_center_xy[1] == pytest.approx(a1[1], abs=1e-6)
    assert cal.rotation == pytest.approx(rot, abs=1e-9)
    assert cal.residual_um < 1e-6


def test_two_wells_are_enough_for_rotation():
    refs = _measure("96-well", (300.0, -200.0), 1.5, ["A1", "H12"])
    cal = calibrate("96-well", refs)
    assert cal.rotation == pytest.approx(1.5, abs=1e-9)
    assert cal.rotation_estimated is True


def test_one_well_fixes_position_and_assumes_no_rotation():
    refs = _measure("96-well", (777.0, 888.0), 0.0, ["A1"])
    cal = calibrate("96-well", refs)
    assert cal.a1_center_xy == pytest.approx((777.0, 888.0))
    assert cal.rotation == 0.0
    assert cal.rotation_estimated is False
    assert "assumed" in cal.describe()


def test_one_well_that_is_not_a1_still_locates_the_plate():
    refs = _measure("96-well", (1000.0, 2000.0), 0.0, ["D6"])
    cal = calibrate("96-well", refs)
    assert cal.a1_center_xy[0] == pytest.approx(1000.0, abs=1e-6)
    assert cal.a1_center_xy[1] == pytest.approx(2000.0, abs=1e-6)


def test_calibration_feeds_straight_into_a_plan():
    refs = _measure("96-well", (1500.0, -900.0), 3.0, ["A1", "A12", "H1"])
    plan = calibrate("96-well", refs).to_plan(selected_wells=((0, 7), (0, 11)))
    got = {str(n): (p.x, p.y) for n, p in
           zip(plan.selected_well_names, plan.selected_well_positions)}
    want = {r.name: (r.x, r.y) for r in
            _measure("96-well", (1500.0, -900.0), 3.0, ["A1", "H12"])}
    for name, (x, y) in want.items():
        assert got[name][0] == pytest.approx(x, abs=1e-6)
        assert got[name][1] == pytest.approx(y, abs=1e-6)


@pytest.mark.parametrize("plate", ["6-well", "24-well", "96-well", "384-well"])
def test_works_across_plate_formats(plate):
    corners = suggested_refs(plate)
    refs = _measure(plate, (250.0, 125.0), -1.25, list(corners))
    cal = calibrate(plate, refs)
    assert cal.a1_center_xy[0] == pytest.approx(250.0, abs=1e-6)
    assert cal.rotation == pytest.approx(-1.25, abs=1e-9)


# ---- noise and guards ----------------------------------------------------

def test_averages_out_centring_error():
    """Hand-centring is imprecise; extra wells should reduce the error, and
    the residual should report roughly how bad the centring was."""
    refs = _measure("96-well", (0.0, 0.0), 0.0, ["A1", "A12", "H1", "H12"])
    jitter = [(12.0, -9.0), (-11.0, 8.0), (9.0, 10.0), (-10.0, -9.0)]
    noisy = [WellRef(r.name, r.x + dx, r.y + dy)
             for r, (dx, dy) in zip(refs, jitter)]
    cal = calibrate("96-well", noisy)
    assert abs(cal.a1_center_xy[0]) < 15 and abs(cal.a1_center_xy[1]) < 15
    assert abs(cal.rotation) < 0.05
    assert 0 < cal.residual_um < 40   # reports the centring error honestly


def test_wrong_plate_type_is_refused_not_silently_fitted():
    """Measuring a 96-well plate but saying 384-well doubles every spacing —
    the fit would 'succeed' and put every position badly off."""
    refs = _measure("96-well", (0.0, 0.0), 0.0, ["A1", "A12", "H1"])
    with pytest.raises(ValueError, match="off the 384-well definition"):
        calibrate("384-well", refs)


def test_misidentified_well_is_refused():
    refs = _measure("96-well", (0.0, 0.0), 0.0, ["A1", "A12"])
    # user thought the second well was A6 when it was really A12
    bad = [refs[0], WellRef("A6", refs[1].x, refs[1].y)]
    with pytest.raises(ValueError, match="off the 96-well definition"):
        calibrate("96-well", bad)


def test_small_scale_error_is_tolerated_and_reported():
    refs = _measure("96-well", (0.0, 0.0), 0.0, ["A1", "A12"])
    stretched = [refs[0],
                 WellRef(refs[1].name, refs[1].x * 1.005, refs[1].y)]
    cal = calibrate("96-well", stretched)
    assert 0 < cal.scale_error < SCALE_TOLERANCE


def test_unknown_well_name_is_rejected():
    with pytest.raises(ValueError, match="not on a 96-well plate"):
        calibrate("96-well", [WellRef("Z99", 0.0, 0.0)])


def test_duplicate_reference_is_rejected():
    with pytest.raises(ValueError, match="given twice"):
        calibrate("96-well", [WellRef("A1", 0, 0), WellRef("a1", 5, 5)])


def test_empty_refs_rejected():
    with pytest.raises(ValueError, match="at least one"):
        calibrate("96-well", [])


def test_well_names_are_case_insensitive():
    cal = calibrate("96-well", [WellRef("a1", 100.0, 200.0)])
    assert cal.a1_center_xy == pytest.approx((100.0, 200.0))


def test_describe_is_human_readable():
    refs = _measure("96-well", (1000.0, 2000.0), 2.0, ["A1", "A12", "H1"])
    text = calibrate("96-well", refs).describe()
    assert "96-well" in text and "A1 at" in text and "rotation" in text


# ------------------------------------------ finding a centre you cannot see

def test_opposite_walls_give_the_centre_with_no_diameter_assumed():
    """At 40x a well is not visible — the field is ~333 um, the well 6400.

    So the centre is never eyeballed: touch one wall, touch the opposite
    one, take the midpoint.
    """
    e = plate.centre_from_edges("96-well", "A1",
                                left=1000 - 3200, right=1000 + 3200,
                                bottom=2000 - 3200, top=2000 + 3200)
    assert (e.x, e.y) == pytest.approx((1000.0, 2000.0))
    assert e.measured_width_um == pytest.approx(6400.0)
    assert e.assumed_axes == ()
    assert e.trustworthy


def test_a_single_wall_per_axis_uses_the_nominal_radius():
    e = plate.centre_from_edges("96-well", "A1", left=-2200, bottom=-1200)
    assert (e.x, e.y) == pytest.approx((1000.0, 2000.0))
    assert set(e.assumed_axes) == {"x", "y"}
    assert e.measured_width_um is None


def test_touching_the_wrong_well_is_caught_by_the_implied_diameter():
    """The free sanity check: two walls imply a diameter."""
    e = plate.centre_from_edges("96-well", "A1",
                                left=-2200, right=7000,
                                bottom=-1200, top=5200)
    assert not e.trustworthy
    assert e.diameter_error > plate.DIAMETER_TOLERANCE
    assert "vs 6400" in e.describe()


def test_an_axis_with_no_wall_at_all_is_refused():
    with pytest.raises(ValueError, match="no Y edge"):
        plate.centre_from_edges("96-well", "A1", left=0, right=6400)
    with pytest.raises(ValueError, match="no X edge"):
        plate.centre_from_edges("96-well", "A1", bottom=0, top=6400)
    with pytest.raises(ValueError, match="at least one well edge"):
        plate.centre_from_edges("96-well", "A1")


def test_an_edge_centre_feeds_the_calibration_directly():
    refs = [
        plate.centre_from_edges("96-well", name,
                                left=cx - 3200, right=cx + 3200,
                                bottom=cy - 3200, top=cy + 3200).to_ref()
        for name, cx, cy in [("A1", 1000, 2000), ("A12", 100000, 2000),
                             ("H1", 1000, -61000)]
    ]
    cal = plate.calibrate("96-well", refs)
    assert cal.residual_um < 1.0
    assert cal.a1_center_xy == pytest.approx((1000.0, 2000.0))


def test_the_residual_threshold_matches_what_a_scan_can_absorb():
    """A registration error shifts the scan grid; it does not compound.

    50 um was the old threshold and would have flagged a perfectly usable
    registration as suspect.
    """
    assert plate.GOOD_RESIDUAL_UM >= 100.0
    # a few hundred um of error is well inside a well's spare margin
    width, _height = plate.well_size_um("96-well")
    assert plate.GOOD_RESIDUAL_UM < width / 10


def test_well_size_is_reported_in_stage_units():
    """useq holds plate dimensions in mm; the stage speaks um."""
    assert plate.well_size_um("96-well") == pytest.approx((6400.0, 6400.0))
    assert plate.well_size_um("384-well")[0] < 6400.0
