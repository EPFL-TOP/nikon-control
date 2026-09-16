"""How long does a timepoint take, and how many positions fit in one?

A ten-hour movie at a five-minute interval is 120 timepoints. Whether it can
also visit 200 positions in three channels depends on numbers nobody can
guess: how fast this stage settles, how long this camera takes to read out,
how long PFS takes to lock, how long the filter turret takes to turn.

So this module **measures them on the actual microscope** and then does
arithmetic. Everything here is honest about which numbers were measured and
which were assumed, because a throughput estimate that silently uses a
plausible-looking default is worse than no estimate at all.

The model per timepoint is::

    per position = move + refocus + sum over channels(switch + expose + read)
    timepoint    = positions x per position
    it fits      <=> timepoint < interval

Stage travel is modelled as ``overhead + distance x rate``, fitted from two
measured distances, because a 200 um hop between fields inside a well and a
40 mm hop between wells are not the same cost.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

# Distances to time, in um. Short is a field-to-field hop inside one well;
# long is roughly a well-to-well move on a 96-well plate.
PROBE_DISTANCES_UM = (200.0, 9000.0)
REPEATS = 3
# Used only when a measurement was not possible, and always reported as such.
ASSUMED_CHANNEL_SWITCH_MS = 100.0
ASSUMED_PFS_LOCK_MS = 400.0


@dataclass
class Timings:
    """Measured (or assumed) costs, all in milliseconds."""

    move_overhead_ms: float = 0.0
    move_ms_per_mm: float = 0.0
    refocus_ms: float = 0.0
    channel_ms: dict[str, float] = field(default_factory=dict)
    channel_switch_ms: float = ASSUMED_CHANNEL_SWITCH_MS
    assumed: list[str] = field(default_factory=list)
    samples: int = 0

    def move_ms(self, distance_um: float) -> float:
        """Cost of a move. Zero distance costs nothing — the overhead is the
        fixed price of *issuing and settling* a move, not of staying put."""
        if distance_um <= 0:
            return 0.0
        return self.move_overhead_ms + self.move_ms_per_mm * (distance_um / 1000.0)

    def channel_cost_ms(self, channel: str) -> float:
        """Cost of one channel; falls back to the mean of what was measured."""
        if channel in self.channel_ms:
            return self.channel_ms[channel]
        if self.channel_ms:
            return statistics.fmean(self.channel_ms.values())
        return 0.0

    def position_ms(self, channels, move_um: float = 0.0,
                    refocus: bool = True) -> float:
        chans = list(channels) or [""]
        total = self.move_ms(move_um)
        if refocus:
            total += self.refocus_ms
        for name in chans:
            total += self.channel_switch_ms + self.channel_cost_ms(name)
        return total

    def describe(self) -> list[str]:
        out = [
            f"stage move   {self.move_overhead_ms:.0f} ms + "
            f"{self.move_ms_per_mm:.1f} ms/mm",
            f"refocus      {self.refocus_ms:.0f} ms",
            f"channel swap {self.channel_switch_ms:.0f} ms",
        ]
        out += [f"  {name:<10} {ms:.0f} ms" for name, ms in self.channel_ms.items()]
        if self.assumed:
            out.append("assumed (not measured): " + "; ".join(self.assumed))
        return out


@dataclass
class Budget:
    """What fits in one timepoint, and what it means over a whole movie."""

    per_position_ms: float
    interval_s: float
    positions: int                 # how many fit
    requested: int = 0             # how many were asked for
    duration_h: float = 0.0

    @property
    def timepoint_s(self) -> float:
        return self.per_position_ms * max(self.requested, 0) / 1000.0

    @property
    def fits(self) -> bool:
        return self.requested <= self.positions

    @property
    def timepoints(self) -> int:
        return int(self.duration_h * 3600 / self.interval_s) if self.interval_s else 0

    @property
    def duty(self) -> float:
        """Fraction of the interval spent acquiring, for the requested count."""
        return self.timepoint_s / self.interval_s if self.interval_s else 0.0

    def describe(self) -> list[str]:
        out = [
            f"{self.per_position_ms / 1000:.2f} s per position",
            f"{self.positions} positions fit in a {self.interval_s / 60:g} min "
            f"interval",
        ]
        if self.requested:
            verdict = "fits" if self.fits else "DOES NOT FIT"
            out.append(f"{self.requested} requested -> "
                       f"{self.timepoint_s:.0f} s per timepoint "
                       f"({self.duty * 100:.0f}% of the interval) — {verdict}")
        if self.duration_h:
            out.append(f"{self.timepoints} timepoints over {self.duration_h:g} h")
        return out


def plan(timings: Timings, channels, interval_s: float, *,
         requested: int = 0, move_um: float = 2000.0,
         refocus: bool = True, headroom: float = 0.8,
         duration_h: float = 0.0) -> Budget:
    """How many positions fit in one interval.

    ``headroom`` leaves part of the interval unused. A timelapse scheduled to
    100% of its interval has no slack for a slow refocus or a retry, and
    drifts later at every timepoint until it is minutes behind.
    """
    per = timings.position_ms(channels, move_um, refocus)
    usable_ms = interval_s * 1000.0 * max(0.0, min(1.0, headroom))
    fit = int(usable_ms // per) if per > 0 else 0
    return Budget(per_position_ms=per, interval_s=interval_s, positions=fit,
                  requested=requested, duration_h=duration_h)


def measure(scope, *, channels=None, repeats: int = REPEATS,
            distances=PROBE_DISTANCES_UM, with_pfs: bool = True,
            progress=None) -> Timings:
    """Time the real hardware. Returns the stage where it found it.

    Moves the stage by the probe distances and back, snaps in each channel,
    and (optionally) times a PFS lock. Nothing here changes the objective or
    leaves the stage somewhere new.
    """
    t = Timings()
    say = progress or (lambda _msg: None)

    # --- stage: fit overhead + rate from two distances --------------------
    if scope.has("xystage"):
        say("timing stage moves")
        start = scope.xy()
        points: list[tuple[float, float]] = []
        try:
            for dist in distances:
                samples = []
                for _ in range(repeats):
                    samples.append(_time_ms(lambda: scope.move_xy(
                        start.x + dist, start.y)))
                    samples.append(_time_ms(lambda: scope.move_xy(
                        start.x, start.y)))
                points.append((dist, statistics.median(samples)))
            t.samples += repeats * 2 * len(distances)
        finally:
            scope.move_xy(start.x, start.y)
        t.move_overhead_ms, t.move_ms_per_mm = _fit_line(points)
    else:
        t.assumed.append("no XY stage — move time counted as zero")

    # --- refocus ----------------------------------------------------------
    if with_pfs and scope.pfs_available():
        say("timing PFS lock")
        try:
            was = scope.pfs_engaged()
            # Each timed engage must start from disengaged — engaging while
            # already locked returns instantly and would time a no-op.
            samples = []
            for _ in range(repeats):
                scope.disengage_pfs()
                samples.append(_time_ms(scope.engage_pfs))
            if not was:
                scope.disengage_pfs()
            t.refocus_ms = statistics.median(samples)
            t.samples += repeats
        except Exception:
            t.refocus_ms = ASSUMED_PFS_LOCK_MS
            t.assumed.append(f"PFS lock {ASSUMED_PFS_LOCK_MS:.0f} ms "
                             "(measurement failed)")
    else:
        t.refocus_ms = 0.0 if not scope.pfs_available() else ASSUMED_PFS_LOCK_MS
        if scope.pfs_available():
            t.assumed.append(f"PFS lock {ASSUMED_PFS_LOCK_MS:.0f} ms (not measured)")

    # --- channels ---------------------------------------------------------
    names = list(channels) if channels is not None else scope.channels()
    if not names:
        say("timing a snap")
        t.channel_ms[""] = statistics.median(
            [_time_ms(scope.snap) for _ in range(repeats)])
        t.samples += repeats
        t.channel_switch_ms = 0.0
        t.assumed.append("no channels defined — timed a bare snap, and "
                         "counted channel switching as zero")
        return t

    started = scope.channel()
    switch_samples: list[float] = []
    try:
        for name in names:
            say(f"timing channel {name}")
            switch_samples.append(_time_ms(lambda: scope.set_channel(name)))
            t.channel_ms[name] = statistics.median(
                [_time_ms(scope.snap) for _ in range(repeats)])
            t.samples += repeats + 1
    finally:
        if started:
            try:
                scope.set_channel(started)
            except Exception:
                pass
    if switch_samples:
        t.channel_switch_ms = statistics.median(switch_samples)
    return t


def _time_ms(fn) -> float:
    start = time.perf_counter()
    fn()
    return (time.perf_counter() - start) * 1000.0


def _fit_line(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Least-squares ``ms = overhead + rate * mm`` over the probe distances."""
    if not points:
        return 0.0, 0.0
    if len(points) == 1:
        dist_mm = points[0][0] / 1000.0
        return 0.0, (points[0][1] / dist_mm if dist_mm else 0.0)
    xs = [d / 1000.0 for d, _ in points]
    ys = [ms for _, ms in points]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    rate = (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
            if denom else 0.0)
    overhead = my - rate * mx
    # A negative fit is noise, not physics.
    return max(0.0, overhead), max(0.0, rate)
