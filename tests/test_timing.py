"""Tests for the acquisition timing model.

The arithmetic is what a ten-hour experiment gets planned on, so it is
tested against fixed numbers rather than measured ones; the measurement
itself is exercised against the demo devices.
"""
import pytest

from nikon_control.scope import timing


def fixed() -> timing.Timings:
    """Round numbers, so the expected answers are obvious by hand."""
    return timing.Timings(
        move_overhead_ms=100.0,
        move_ms_per_mm=50.0,
        refocus_ms=400.0,
        channel_ms={"BF": 50.0, "GFP": 300.0, "mCherry": 300.0},
        channel_switch_ms=100.0,
    )


def test_move_time_grows_with_distance():
    t = fixed()
    assert t.move_ms(1000) == 150.0          # 1 mm
    assert t.move_ms(40000) == 2100.0        # a well-to-well hop


def test_not_moving_costs_nothing():
    """The overhead is the price of issuing and settling a move."""
    assert fixed().move_ms(0) == 0.0


def test_position_cost_adds_every_channel_and_its_switch():
    t = fixed()
    # 100 + 50*2mm move, 400 refocus, then (100+50)+(100+300)+(100+300)
    assert t.position_ms(["BF", "GFP", "mCherry"], move_um=2000) == \
        pytest.approx(200 + 400 + 150 + 400 + 400)


def test_an_unmeasured_channel_falls_back_to_the_mean():
    t = fixed()
    solo = t.position_ms(["YFP"], move_um=0, refocus=False)
    mean = (50 + 300 + 300) / 3
    assert solo == pytest.approx(100 + mean)


def test_more_channels_means_fewer_positions():
    t = fixed()
    one = timing.plan(t, ["BF"], interval_s=300)
    three = timing.plan(t, ["BF", "GFP", "mCherry"], interval_s=300)
    assert three.positions < one.positions


def test_headroom_is_held_back_from_the_interval():
    """A timelapse scheduled to 100% of its interval drifts later forever."""
    t = fixed()
    full = timing.plan(t, ["BF"], interval_s=300, headroom=1.0)
    default = timing.plan(t, ["BF"], interval_s=300)
    assert default.positions < full.positions
    assert default.positions == pytest.approx(full.positions * 0.8, rel=0.02)


def test_a_request_that_does_not_fit_says_so():
    t = fixed()
    budget = timing.plan(t, ["BF", "GFP", "mCherry"], interval_s=300,
                         requested=10000, move_um=2000)
    assert not budget.fits
    assert budget.duty > 1.0
    assert "DOES NOT FIT" in " ".join(budget.describe())


def test_a_request_that_fits_says_so_with_the_duty_cycle():
    t = fixed()
    budget = timing.plan(t, ["BF"], interval_s=300, requested=10,
                         move_um=2000, duration_h=10)
    assert budget.fits
    assert 0 < budget.duty < 1
    text = " ".join(budget.describe())
    assert "fits" in text
    assert "120 timepoints over 10 h" in text     # 10 h / 5 min


def test_timepoint_count_follows_the_interval():
    t = fixed()
    assert timing.plan(t, ["BF"], interval_s=300, duration_h=10).timepoints == 120
    assert timing.plan(t, ["BF"], interval_s=600, duration_h=10).timepoints == 60


def test_line_fit_recovers_overhead_and_rate():
    overhead, rate = timing._fit_line([(1000.0, 150.0), (9000.0, 550.0)])
    assert overhead == pytest.approx(100.0)
    assert rate == pytest.approx(50.0)


def test_line_fit_never_returns_a_negative_cost():
    """Noise at short distances can fit a negative intercept; physics cannot."""
    overhead, rate = timing._fit_line([(200.0, 10.0), (9000.0, 500.0)])
    assert overhead >= 0.0 and rate >= 0.0


def test_describe_flags_what_was_assumed_rather_than_measured():
    t = fixed()
    t.assumed.append("PFS lock 400 ms (not measured)")
    assert any("assumed" in line for line in t.describe())


# ------------------------------------------------------ against real devices

pytest.importorskip("pymmcore_plus")
from nikon_control.scope import discover  # noqa: E402
from nikon_control.scope.control import Scope  # noqa: E402

needs_demo = pytest.mark.skipif(
    "DemoCamera" not in discover.available_adapters(),
    reason="Micro-Manager demo adapters not installed (run: mmcore install)",
)


@needs_demo
def test_measure_returns_the_stage_where_it_found_it():
    """A measurement that moved the sample would be worse than useless."""
    scope = Scope.demo()
    scope.move_xy(1234, -567)
    before = scope.xy()

    timing.measure(scope, channels=[], repeats=1, distances=(200.0,))

    after = scope.xy()
    assert round(after.x) == round(before.x)
    assert round(after.y) == round(before.y)


@needs_demo
def test_measure_times_every_requested_channel():
    scope = Scope.demo()
    wanted = scope.channels()[:2]
    t = timing.measure(scope, channels=wanted, repeats=1)
    assert set(t.channel_ms) == set(wanted)
    assert t.samples > 0
    assert all(ms >= 0 for ms in t.channel_ms.values())


@needs_demo
def test_measure_leaves_the_channel_it_found():
    scope = Scope.demo()
    names = scope.channels()
    scope.set_channel(names[-1])
    before = scope.channel()
    timing.measure(scope, channels=names[:2], repeats=1)
    assert scope.channel() == before


@needs_demo
def test_measure_with_no_channels_still_times_a_snap():
    scope = Scope.demo()
    t = timing.measure(scope, channels=[], repeats=1, distances=(200.0,))
    assert t.channel_ms
    assert t.channel_switch_ms == 0.0
    assert any("no channels" in a for a in t.assumed)


def test_each_pfs_sample_starts_from_disengaged():
    """Engaging while already locked returns instantly — a timed no-op.

    Only the first of N samples would be real, and the median would read as
    a PFS that locks in no time at all.
    """
    from tests.test_scope_control import ROLES, FakeCore
    from nikon_control.scope.control import Scope

    class Recording(FakeCore):
        def __init__(self):
            super().__init__()
            self.order: list[str] = []

        def enableContinuousFocus(self, on):
            self.order.append("on" if on else "off")
            super().enableContinuousFocus(on)

    scope = Scope(Recording(), ROLES)
    timing.measure(scope, channels=[], repeats=3, distances=(200.0,))

    order = scope.core.order
    engages = [i for i, v in enumerate(order) if v == "on"]
    assert len(engages) >= 3
    for i in engages[:3]:
        assert i > 0 and order[i - 1] == "off", \
            f"an engage at {i} was not preceded by a disengage: {order}"


def test_a_pfs_that_never_locks_is_assumed_not_measured():
    """Regression: a 5 s timeout was recorded as the measured refocus cost.

    The whole experiment gets planned on that number — it turned ~300
    positions per interval into 47, in the module whose docstring says a
    silent plausible default is worse than no estimate.
    """
    from nikon_control.scope.control import Scope
    from tests.test_scope_control import ROLES, NeverLocks

    scope = Scope(NeverLocks(), ROLES)
    t = timing.measure(scope, channels=[], repeats=2, distances=(200.0,))

    assert t.refocus_ms == timing.ASSUMED_PFS_LOCK_MS
    assert any("never locked" in a for a in t.assumed)
    assert any("assumed" in line for line in t.describe())


def test_a_pfs_that_locks_is_measured_normally():
    from nikon_control.scope.control import Scope
    from tests.test_scope_control import ROLES, FakeCore

    scope = Scope(FakeCore(), ROLES)
    t = timing.measure(scope, channels=[], repeats=2, distances=(200.0,))
    assert not any("never locked" in a for a in t.assumed)
    assert t.samples > 0
