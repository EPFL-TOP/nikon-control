"""Register a well plate against the stage — GUI-free, hardware-free, pure.

The question this answers: *where is the plate, in stage coordinates?* Once
that is known, every well and every field inside a well follows from the
plate's own geometry, so nothing downstream has to think about stage
coordinates again.

``useq.WellPlatePlan`` already models a plate as:

- ``plate`` — rows, columns, well spacing and well size (96-well, 384-well …
  come from useq's registry);
- ``a1_center_xy`` — where the centre of well A1 sits on the stage;
- ``rotation`` — how the plate is turned relative to the stage axes.

So calibration means measuring exactly three numbers: **A1's stage position
and the plate's rotation.** This module fits them from a few wells the user
has driven to and centred by eye.

The fit is a rigid 2-D registration (rotation + translation) with the scale
*fixed* by the plate definition — spacing is a manufacturing property, not
something to fit. That means two reference wells are enough, and a third
gives a residual worth trusting.

The nominal well geometry is read back out of ``useq`` itself rather than
re-derived here, so this module cannot drift from useq's conventions (which
way rows run, mm vs µm, where rotation is centred).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# A plate reading well-to-well distances more than this far from the plate
# definition is almost certainly the wrong plate type, or a mis-identified
# well — worth refusing rather than silently imaging empty plastic.
SCALE_TOLERANCE = 0.02  # 2%


@dataclass(frozen=True)
class WellRef:
    """A well the user centred under the objective, with the stage readout."""

    name: str          # "A1", "H12" …
    x: float           # stage x when centred (µm)
    y: float           # stage y when centred (µm)


@dataclass(frozen=True)
class PlateCalibration:
    plate: str
    a1_center_xy: tuple[float, float]
    rotation: float               # degrees, useq's convention
    residual_um: float            # worst |measured − predicted| over the refs
    scale_error: float            # measured spacing vs the plate definition
    n_refs: int

    @property
    def rotation_estimated(self) -> bool:
        """False when only one well was given — rotation was assumed."""
        return self.n_refs >= 2

    def to_plan(self, **kwargs):
        """Build the ``useq.WellPlatePlan`` this calibration describes.

        Pass ``selected_wells`` / ``well_points_plan`` to say which wells to
        visit and how to sample inside each.
        """
        from useq import WellPlatePlan

        return WellPlatePlan(
            plate=self.plate,
            a1_center_xy=self.a1_center_xy,
            rotation=self.rotation,
            **kwargs,
        )

    def describe(self) -> str:
        rot = (f"{self.rotation:+.3f}°" if self.rotation_estimated
               else "0° (assumed — only one reference well)")
        return (
            f"{self.plate}: A1 at ({self.a1_center_xy[0]:.1f}, "
            f"{self.a1_center_xy[1]:.1f}) µm, rotation {rot}, "
            f"residual {self.residual_um:.1f} µm over {self.n_refs} well(s)"
        )


def nominal_offsets(plate: str) -> dict[str, tuple[float, float]]:
    """Well-centre offsets from A1, in stage units, per useq's own geometry.

    Built by asking useq for a plate pinned at the origin with no rotation,
    so the conventions are useq's rather than ours.
    """
    import numpy as np
    from useq import WellPlatePlan

    plan = WellPlatePlan(plate=plate, a1_center_xy=(0.0, 0.0), rotation=0.0)
    names = np.asarray(plan.all_well_names).ravel()
    return {str(n): (float(p.x), float(p.y))
            for n, p in zip(names, plan.all_well_positions)}


def calibrate(plate: str, refs: list[WellRef]) -> PlateCalibration:
    """Fit A1's stage position and the plate rotation from centred wells.

    One reference well fixes the position and assumes zero rotation. Two or
    more also fit the rotation; three or more make the residual meaningful.

    Raises ``ValueError`` for an unknown well name, duplicate references, or
    a well-to-well spacing that disagrees with the plate definition by more
    than ``SCALE_TOLERANCE`` — which means the wrong plate type was chosen or
    a well was misidentified, and would otherwise put every position off by a
    growing amount across the plate.
    """
    if not refs:
        raise ValueError("need at least one reference well")

    offsets = nominal_offsets(plate)
    seen: set[str] = set()
    d: list[tuple[float, float]] = []
    m: list[tuple[float, float]] = []
    for r in refs:
        key = r.name.strip().upper()
        if key not in offsets:
            raise ValueError(
                f"well {r.name!r} is not on a {plate} plate "
                f"(expected e.g. {', '.join(sorted(offsets)[:3])} …)"
            )
        if key in seen:
            raise ValueError(f"well {key} given twice")
        seen.add(key)
        d.append(offsets[key])
        m.append((float(r.x), float(r.y)))

    n = len(d)
    if n == 1:
        a1 = (m[0][0] - d[0][0], m[0][1] - d[0][1])
        return PlateCalibration(plate, a1, 0.0, 0.0, 0.0, 1)

    # scale check before fitting: compare the longest reference baseline
    # against what the plate definition says it should be
    scale_error = _scale_error(d, m)
    if abs(scale_error) > SCALE_TOLERANCE:
        raise ValueError(
            f"measured well spacing is {scale_error:+.1%} off the {plate} "
            "definition — wrong plate type, or a reference well was "
            "misidentified. Check the well labels and the plate choice."
        )

    # rigid fit (Kabsch in 2-D): rotation from the cross/dot products of the
    # centred point sets, then translation from the centroids.
    dxm = sum(p[0] for p in d) / n
    dym = sum(p[1] for p in d) / n
    mxm = sum(p[0] for p in m) / n
    mym = sum(p[1] for p in m) / n
    cross = sum((d[i][0] - dxm) * (m[i][1] - mym)
                - (d[i][1] - dym) * (m[i][0] - mxm) for i in range(n))
    dot = sum((d[i][0] - dxm) * (m[i][0] - mxm)
              + (d[i][1] - dym) * (m[i][1] - mym) for i in range(n))
    theta = math.atan2(cross, dot)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    a1x = mxm - (cos_t * dxm - sin_t * dym)
    a1y = mym - (sin_t * dxm + cos_t * dym)

    residual = 0.0
    for i in range(n):
        px = a1x + cos_t * d[i][0] - sin_t * d[i][1]
        py = a1y + sin_t * d[i][0] + cos_t * d[i][1]
        residual = max(residual, math.hypot(px - m[i][0], py - m[i][1]))

    return PlateCalibration(
        plate=plate,
        a1_center_xy=(a1x, a1y),
        rotation=math.degrees(theta),
        residual_um=residual,
        scale_error=scale_error,
        n_refs=n,
    )


def _scale_error(d: list[tuple[float, float]],
                 m: list[tuple[float, float]]) -> float:
    """Relative error between measured and nominal reference spacing.

    Uses the longest baseline among the references, where a scale error shows
    up most clearly.
    """
    best_nom = best_meas = 0.0
    for i in range(len(d)):
        for j in range(i + 1, len(d)):
            nom = math.hypot(d[i][0] - d[j][0], d[i][1] - d[j][1])
            if nom > best_nom:
                best_nom = nom
                best_meas = math.hypot(m[i][0] - m[j][0], m[i][1] - m[j][1])
    if best_nom <= 0:
        return 0.0
    return best_meas / best_nom - 1.0


def suggested_refs(plate: str) -> tuple[str, str, str]:
    """Three wells that make a good calibration: A1 and the two far corners.

    A long baseline is what pins down the rotation — adjacent wells barely
    constrain it, so corners are worth the extra stage travel.
    """
    import numpy as np
    from useq import WellPlatePlan

    names = np.asarray(
        WellPlatePlan(plate=plate, a1_center_xy=(0.0, 0.0)).all_well_names
    )
    return (str(names[0, 0]), str(names[0, -1]), str(names[-1, 0]))


# --------------------------------------------------------------- persistence
# The CLI writes a calibration to JSON and the dashboard reads it back, so the
# shape of that file belongs here rather than in either of them.

def to_dict(cal: PlateCalibration) -> dict:
    return {
        "plate": cal.plate,
        "a1_center_xy": list(cal.a1_center_xy),
        "rotation": cal.rotation,
        "residual_um": cal.residual_um,
        "scale_error": cal.scale_error,
        "n_refs": cal.n_refs,
    }


def from_dict(obj: dict) -> PlateCalibration:
    try:
        a1 = tuple(float(v) for v in obj["a1_center_xy"])
        return PlateCalibration(
            plate=str(obj["plate"]),
            a1_center_xy=(a1[0], a1[1]),
            rotation=float(obj.get("rotation", 0.0)),
            residual_um=float(obj.get("residual_um", 0.0)),
            scale_error=float(obj.get("scale_error", 0.0)),
            n_refs=int(obj.get("n_refs", 1)),
        )
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"not a plate calibration file: {exc}") from exc


def save(cal: PlateCalibration, path) -> None:
    """Write a calibration, merging into whatever else the file holds.

    The well selection lives in this same file (``wells.save_selection``
    merges into it the other way round). A whole-file overwrite here would
    make the merge one-way: re-registering the plate would silently delete
    the wells someone chose.
    """
    import json
    from pathlib import Path

    p = Path(path)
    obj = {}
    if p.exists():
        try:
            obj = json.loads(p.read_text())
        except (ValueError, OSError):
            obj = {}
        if not isinstance(obj, dict):
            obj = {}
        if obj.get("plate") and obj["plate"] != cal.plate:
            # A different plate type invalidates the well names too, so
            # carrying them over would be worse than dropping them.
            obj = {}
    obj.update(to_dict(cal))
    p.write_text(json.dumps(obj, indent=2))


def load(path) -> PlateCalibration:
    import json
    from pathlib import Path

    return from_dict(json.loads(Path(path).read_text()))


# ------------------------------------------------------------------- drawing

@dataclass(frozen=True)
class PlateLayout:
    """Everything needed to draw a registered plate in stage coordinates."""

    names: list[str]
    x: list[float]
    y: list[float]
    well_width_um: float
    well_height_um: float
    circular: bool
    rows: int
    columns: int


def layout(cal: PlateCalibration) -> PlateLayout:
    """Stage position and size of every well, for a map of the plate.

    Positions come from ``useq`` applying the calibration, so the map shows
    where the microscope believes the wells are — which is the thing worth
    checking against reality.
    """
    import numpy as np

    plan = cal.to_plan()
    names = [str(n) for n in np.asarray(plan.all_well_names).ravel()]
    xs = [float(p.x) for p in plan.all_well_positions]
    ys = [float(p.y) for p in plan.all_well_positions]
    # useq holds plate dimensions in mm; the stage speaks µm.
    w, h = (float(v) * 1000.0 for v in plan.plate.well_size)
    return PlateLayout(
        names=names, x=xs, y=ys,
        well_width_um=w, well_height_um=h,
        circular=bool(plan.plate.circular_wells),
        rows=int(plan.plate.rows), columns=int(plan.plate.columns),
    )


def nearest_well(cal: PlateCalibration, x: float, y: float
                 ) -> tuple[str, float, float, float]:
    """Well nearest a stage point: ``(name, well_x, well_y, distance_um)``."""
    lay = layout(cal)
    best = min(range(len(lay.names)),
               key=lambda i: (lay.x[i] - x) ** 2 + (lay.y[i] - y) ** 2)
    dist = math.hypot(lay.x[best] - x, lay.y[best] - y)
    return lay.names[best], lay.x[best], lay.y[best], dist


# ------------------------------------------------------- finding a well centre
# You cannot centre a well by eye at 40x — the field of view is ~333 um and a
# 96-well well is 6400 um across, so the well is not even visible. Two things
# make this a non-problem:
#
# 1. **The tolerance is not microns.** A position error shifts the whole scan
#    grid by that amount; it does not compound. Scan 15x15 fields of a 19-field
#    well and there is ~700 um of margin on each side, so a few hundred um of
#    registration error costs nothing. Rotation is the part worth care, which
#    is why the references should be far apart.
# 2. **Opposite edges beat a guessed centre.** Drive until the well wall sits
#    in the middle of the image on one side, then the other: the centre is the
#    midpoint. That needs no judgement about where a centre is, works at any
#    magnification, and the implied diameter is a free check that you touched
#    the right wall.
#
# Registration is also fine to do at 4x or 10x and then image at 40x. The
# objectives are not parcentric, but that offset is a CONSTANT translation —
# it shifts every reference equally, so it lands in a1_center_xy as a fixed
# bias of tens of um, far inside tolerance. (It is a real problem only when
# you must hit a specific cell across a magnification change, which is a
# different job with a ~10 um budget.)

# A registration this good is enough for a well scan with a sensible margin.
GOOD_RESIDUAL_UM = 300.0
# Two opposite edge touches imply a well diameter; this far off the plate
# definition means a wall was mis-identified or the plate type is wrong.
DIAMETER_TOLERANCE = 0.10  # 10%


@dataclass(frozen=True)
class EdgeCentre:
    """A well centre worked out from wall touches, and how trustworthy it is."""

    name: str
    x: float
    y: float
    measured_width_um: float | None      # from an opposite X pair
    measured_height_um: float | None     # from an opposite Y pair
    nominal_um: float
    assumed_axes: tuple[str, ...] = ()   # axes where only one edge was given

    @property
    def diameter_error(self) -> float:
        """Worst relative disagreement with the plate's own well size."""
        worst = 0.0
        for measured in (self.measured_width_um, self.measured_height_um):
            if measured and self.nominal_um:
                worst = max(worst, abs(measured / self.nominal_um - 1.0))
        return worst

    @property
    def trustworthy(self) -> bool:
        return self.diameter_error <= DIAMETER_TOLERANCE

    def to_ref(self) -> WellRef:
        return WellRef(name=self.name, x=self.x, y=self.y)

    def describe(self) -> str:
        bits = [f"{self.name} centre ({self.x:.0f}, {self.y:.0f}) µm"]
        if self.measured_width_um:
            bits.append(f"width {self.measured_width_um:.0f} µm")
        if self.measured_height_um:
            bits.append(f"height {self.measured_height_um:.0f} µm")
        if self.assumed_axes:
            bits.append("assumed the nominal radius in "
                        + "/".join(self.assumed_axes))
        if self.measured_width_um or self.measured_height_um:
            bits.append(f"vs {self.nominal_um:.0f} µm nominal "
                        f"({self.diameter_error:+.1%})")
        return ", ".join(bits)


def well_size_um(plate: str) -> tuple[float, float]:
    """The plate's well size in stage units (useq holds it in mm)."""
    from useq import WellPlatePlan

    plan = WellPlatePlan(plate=plate, a1_center_xy=(0.0, 0.0))
    w, h = plan.plate.well_size
    return float(w) * 1000.0, float(h) * 1000.0


def centre_from_edges(plate: str, name: str, *,
                      left: float | None = None,
                      right: float | None = None,
                      top: float | None = None,
                      bottom: float | None = None) -> EdgeCentre:
    """A well centre from stage readings taken at the well walls.

    Give whichever you have. An opposite pair in an axis is best — the centre
    is the midpoint and no well diameter is assumed. A single edge works too,
    offset by the nominal radius.

    ``left``/``right`` are stage X readings, ``top``/``bottom`` stage Y. Which
    physical wall is "left" does not matter; only that the two are opposite.
    """
    width, height = well_size_um(plate)
    if not any(v is not None for v in (left, right, top, bottom)):
        raise ValueError("give at least one well edge")

    assumed: list[str] = []

    def axis(lo, hi, nominal, axis_name):
        if lo is not None and hi is not None:
            return (lo + hi) / 2.0, abs(hi - lo)
        assumed.append(axis_name)
        radius = nominal / 2.0
        if lo is not None:
            # A single edge cannot say which side of it the well lies, so the
            # convention is: the well is in the +axis direction from `lo`.
            return lo + radius, None
        return hi - radius, None

    if left is None and right is None:
        raise ValueError(f"no X edge for {name} — give left and/or right")
    if top is None and bottom is None:
        raise ValueError(f"no Y edge for {name} — give top and/or bottom")

    cx, measured_w = axis(left, right, width, "x")
    cy, measured_h = axis(bottom, top, height, "y")
    return EdgeCentre(name=name.strip().upper(), x=cx, y=cy,
                      measured_width_um=measured_w,
                      measured_height_um=measured_h,
                      nominal_um=(width + height) / 2.0,
                      assumed_axes=tuple(assumed))
