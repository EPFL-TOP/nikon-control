"""Tests for well selection — plugin 2 of the acquisition pipeline.

A wrong well set is only discovered hours into an overnight run, so the
range logic is pinned here rather than trusted to a GUI.
"""
import pytest

from nikon_control.scope import plate as plate_mod
from nikon_control.scope import wells

pytest.importorskip("useq")


@pytest.fixture
def cal96():
    return plate_mod.calibrate("96-well", [
        plate_mod.WellRef("A1", 1000, 2000),
        plate_mod.WellRef("A12", 100000, 2000),
        plate_mod.WellRef("H1", 1000, -61000),
    ])


# ------------------------------------------------------------------ names

def test_names_come_from_useq_not_from_us():
    g = wells.grid("96-well")
    assert g.n_rows == 8 and g.n_columns == 12
    assert g.rows[0] == "A" and g.rows[-1] == "H"
    assert g.columns[0] == "1" and g.columns[-1] == "12"
    assert g.names[0][0] == "A1" and g.names[-1][-1] == "H12"


def test_big_plate_rows_run_past_z():
    """A 1536-well plate has 32 rows: A..Z then AA..AF. Worth not reinventing."""
    g = wells.grid("1536-well")
    assert g.n_rows == 32
    assert g.rows[25] == "Z" and g.rows[26] == "AA"
    assert g.index("AF48") == (31, 47)


def test_user_typing_is_normalised_to_useqs_names():
    assert wells.normalise("a1") == "A1"
    assert wells.normalise("A01") == "A1"
    assert wells.normalise(" h 12 ") == "H12"


# -------------------------------------------------------------- selecting

def test_toggle_adds_then_removes():
    sel = wells.Selection("96-well")
    sel.toggle("a1")
    assert sel.wells == {"A1"}
    sel.toggle("A1")
    assert sel.wells == set()


def test_a_rectangle_spans_both_corners_whichever_way_round():
    sel = wells.Selection("96-well")
    forward = sel.rectangle("A1", "C3")
    backward = sel.rectangle("C3", "A1")
    assert forward == backward
    assert forward == ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]


def test_rows_and_columns():
    sel = wells.Selection("96-well")
    assert sel.row("C") == [f"C{i}" for i in range(1, 13)]
    assert sel.column("3") == [f"{r}3" for r in "ABCDEFGH"]
    assert sel.row("Z") == []          # not on this plate


def test_parse_understands_the_shorthands():
    got = wells.parse("96-well", "A1, B2-B5, D*, *12")
    assert got[0] == "A1"
    assert {"B2", "B3", "B4", "B5"} <= set(got)
    assert len([g for g in got if g.startswith("D")]) == 12
    assert "H12" in got
    assert len(got) == len(set(got)), "parse returned duplicates"


def test_parse_rectangle_with_a_colon():
    assert wells.parse("96-well", "A1:B3") == ["A1", "A2", "A3",
                                               "B1", "B2", "B3"]


def test_parse_all():
    assert len(wells.parse("96-well", "all")) == 96


def test_parse_skips_nonsense_rather_than_raising():
    """Typed by hand mid-experiment; one typo must not lose the rest."""
    got = wells.parse("96-well", "A1, Q99, zzz, B2")
    assert got == ["A1", "B2"]


def test_parse_rejects_a_well_that_is_not_on_this_plate():
    assert wells.parse("6-well", "H12") == []
    assert wells.parse("6-well", "B3") == ["B3"]


# --------------------------------------------------------------- ordering

def test_ordered_follows_plate_order_not_click_order():
    sel = wells.Selection("96-well")
    sel.add(["C3", "A1", "B2"])
    assert sel.ordered() == ["A1", "B2", "C3"]


def test_serpentine_reverses_alternate_rows():
    """A raster drives back across the whole plate at every row end."""
    sel = wells.Selection("96-well").add(wells.parse("96-well", "A1:B4"))
    assert sel.ordered() == ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]
    assert sel.serpentine() == ["A1", "A2", "A3", "A4", "B4", "B3", "B2", "B1"]


def test_serpentine_travel_is_shorter_than_raster(cal96):
    """The reason serpentine exists, measured rather than asserted."""
    import math

    sel = wells.Selection("96-well").select_all()
    lay = plate_mod.layout(cal96)
    at = {n: (x, y) for n, x, y in zip(lay.names, lay.x, lay.y)}

    def travel(order):
        return sum(math.dist(at[a], at[b])
                   for a, b in zip(order, order[1:]))

    assert travel(sel.serpentine()) < travel(sel.ordered()) * 0.6


# ----------------------------------------------------------------- output

def test_to_plan_yields_only_the_selected_wells(cal96):
    sel = wells.Selection("96-well").add(["A1", "B2", "H12"])
    plan = sel.to_plan(cal96)
    names = [p.name for p in plan.image_positions]
    assert names == ["A1", "B2", "H12"]


def test_to_plan_positions_match_the_calibration(cal96):
    sel = wells.Selection("96-well").add(["A1", "A2"])
    got = {p.name: (round(p.x), round(p.y)) for p in sel.to_plan(cal96).image_positions}
    assert got["A1"] == (1000, 2000)
    assert got["A2"] == (10000, 2000)          # 9 mm spacing


def test_to_plan_refuses_a_calibration_for_a_different_plate(cal96):
    sel = wells.Selection("384-well").add(["A1"])
    with pytest.raises(ValueError, match="384-well"):
        sel.to_plan(cal96)


# ------------------------------------------------------------ persistence

def test_selection_rides_along_with_the_calibration(tmp_path, cal96):
    path = tmp_path / "plate.json"
    plate_mod.save(cal96, path)
    sel = wells.Selection("96-well").add(["A1", "C5"])
    wells.save_selection(sel, path)

    # the calibration survives...
    back = plate_mod.load(path)
    assert back.plate == "96-well"
    assert back.a1_center_xy == cal96.a1_center_xy
    # ...and so does the selection
    loaded = wells.load_selection(path)
    assert loaded.wells == {"A1", "C5"}


def test_saving_a_selection_for_the_wrong_plate_is_refused(tmp_path, cal96):
    path = tmp_path / "plate.json"
    plate_mod.save(cal96, path)
    with pytest.raises(ValueError, match="refusing"):
        wells.save_selection(wells.Selection("384-well").add(["A1"]), path)


def test_loading_a_file_with_no_selection_returns_none(tmp_path, cal96):
    path = tmp_path / "plate.json"
    plate_mod.save(cal96, path)
    loaded = wells.load_selection(path)
    assert loaded is not None and loaded.wells == set()


def test_loading_a_missing_or_broken_file_returns_none(tmp_path):
    assert wells.load_selection(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert wells.load_selection(bad) is None
