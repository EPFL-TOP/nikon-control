"""Scan selected wells field by field — plugin 3 of the pipeline.

Output is a folder of 16-bit TIFFs plus a JSON manifest that records, for
every frame, **where on the stage it came from**. That is the whole point:
detection runs offline on the frames, and each detected cell's pixel position
plus its frame's stage position gives a stage coordinate the acquisition can
drive back to. A scan without per-frame stage coordinates is just pictures.

Geometry comes from ``useq``: the plate calibration says where the wells are,
``GridRowsColumns`` lays fields inside each one, and the field order is
already serpentine. Nothing here re-derives plate arithmetic.

Two numbers decide the shape of a scan, and both are worth seeing before
starting one:

* **Field of view.** At 40× with 6.5 µm camera pixels the field is ~333 µm, so
  covering a 6400 µm well takes 19×19 = 361 fields. Full coverage is rarely
  what you want for a survey — a centred 11×11 samples the middle third and
  leaves margin for plate-registration error.
* **Time.** ``timing.py`` measures this microscope, so the estimate is
  measured rather than guessed.

The scan is deliberately **brightfield-only by default**: a survey exists to
find cells, and firing fluorescence at every field of every well bleaches the
sample before the experiment starts.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

MANIFEST = "scan.json"
# Fraction of the field that overlaps its neighbour. A little overlap means a
# cell on a boundary appears whole in one of the two frames.
DEFAULT_OVERLAP = 0.10
# Frames are written as plain 16-bit TIFF: the annotation and training tools
# already read TIFF, and a flat folder survives a crashed scan.
FRAME_FMT = "{well}_f{index:04d}.tif"


@dataclass(frozen=True)
class Field:
    """One field of view to visit."""

    name: str
    well: str
    index: int
    x: float
    y: float


@dataclass
class ScanPlan:
    """Everything a scan will do, before it does any of it."""

    plate: str
    wells: list[str]
    rows: int
    columns: int
    fov_um: tuple[float, float]
    overlap: float
    fields: list[Field] = field(default_factory=list)

    @property
    def n_fields(self) -> int:
        return len(self.fields)

    @property
    def covered_um(self) -> tuple[float, float]:
        """Extent of the grid in each well, accounting for overlap."""
        step_x = self.fov_um[0] * (1.0 - self.overlap)
        step_y = self.fov_um[1] * (1.0 - self.overlap)
        return (step_x * (self.columns - 1) + self.fov_um[0],
                step_y * (self.rows - 1) + self.fov_um[1])

    def coverage(self, well_um: tuple[float, float]) -> float:
        """Fraction of the well's width the grid spans (1.0 = edge to edge)."""
        if not well_um or not well_um[0]:
            return 0.0
        return min(self.covered_um[0] / well_um[0],
                   self.covered_um[1] / well_um[1])

    def describe(self, well_um: tuple[float, float] | None = None) -> list[str]:
        out = [
            f"{len(self.wells)} well(s) x {self.rows}x{self.columns} fields "
            f"= {self.n_fields} frames",
            f"field {self.fov_um[0]:.0f} x {self.fov_um[1]:.0f} µm, "
            f"{self.overlap:.0%} overlap → {self.covered_um[0]:.0f} x "
            f"{self.covered_um[1]:.0f} µm per well",
        ]
        if well_um:
            frac = self.coverage(well_um)
            out.append(f"{frac:.0%} of the {well_um[0]:.0f} µm well width"
                       + ("" if frac <= 1.0 else
                          " — the grid runs past the well onto the plastic"))
        return out


def plan(calibration, wells, *, fov_um, rows: int, columns: int,
         overlap: float = DEFAULT_OVERLAP) -> ScanPlan:
    """Lay a field grid inside each selected well.

    ``wells`` is a list of names or a :class:`~nikon_control.scope.wells.Selection`.
    Positions come from ``useq`` applying the calibration, so they are in the
    same stage frame everything else uses.
    """
    from useq import GridRowsColumns

    names = list(getattr(wells, "visiting_order", lambda: list(wells))())
    if not names:
        raise ValueError("no wells selected")
    if rows < 1 or columns < 1:
        raise ValueError("a grid needs at least one row and one column")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")

    from . import wells as wells_mod

    selection = (wells if hasattr(wells, "to_plan")
                 else wells_mod.Selection(calibration.plate, set(names)))
    useq_plan = selection.to_plan(
        calibration,
        well_points_plan=GridRowsColumns(
            rows=rows, columns=columns,
            fov_width=fov_um[0], fov_height=fov_um[1],
            overlap=overlap * 100.0,       # useq takes a percentage
        ),
    )

    fields: list[Field] = []
    per_well: dict[str, int] = {}
    for pos in useq_plan.image_positions:
        well = str(pos.name or "").split("_")[0] or "?"
        idx = per_well.get(well, 0)
        per_well[well] = idx + 1
        fields.append(Field(name=str(pos.name or f"{well}_f{idx:04d}"),
                            well=well, index=idx,
                            x=float(pos.x), y=float(pos.y)))
    return ScanPlan(plate=calibration.plate, wells=names, rows=rows,
                    columns=columns, fov_um=(float(fov_um[0]), float(fov_um[1])),
                    overlap=float(overlap), fields=fields)


def fields_for_coverage(fov_um, well_um, fraction: float = 1.0,
                        overlap: float = DEFAULT_OVERLAP) -> tuple[int, int]:
    """How many rows and columns span ``fraction`` of a well.

    Handy because the useful question is "sample the middle third", not "give
    me 11 columns".
    """
    import math

    out = []
    for fov, well in zip(fov_um, well_um):
        step = fov * (1.0 - overlap)
        want = max(0.0, well * fraction - fov)
        out.append(max(1, int(math.ceil(want / step)) + 1) if step > 0 else 1)
    return out[1], out[0]          # (rows, columns)


def estimate_seconds(scan_plan: ScanPlan, timings, channels=("",),
                     refocus_every: int = 1) -> float:
    """How long the scan will take, from measured hardware timings.

    ``refocus_every`` is how often PFS is re-engaged: 1 means at every field,
    which is safest and slowest; a well's fields are coplanar enough that
    refocusing once per well is usually enough.
    """
    step = scan_plan.fov_um[0] * (1.0 - scan_plan.overlap)
    per_well = max(1, scan_plan.n_fields // max(1, len(scan_plan.wells)))
    total_ms = 0.0
    for i in range(scan_plan.n_fields):
        # A field-to-field hop, except at the start of a well, which is a
        # long move across the plate.
        hop = step if i % per_well else 9000.0
        refocus = refocus_every > 0 and (i % refocus_every == 0)
        total_ms += timings.position_ms(channels, move_um=hop, refocus=refocus)
    return total_ms / 1000.0


@dataclass
class ScanResult:
    directory: Path
    frames: list[dict] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)

    @property
    def manifest_path(self) -> Path:
        return self.directory / MANIFEST

    def describe(self) -> list[str]:
        out = [f"{len(self.frames)} frame(s) written to {self.directory}"]
        if self.failures:
            out.append(f"{len(self.failures)} position(s) failed:")
            out += [f"  {f['name']}: {f['error']}" for f in self.failures[:6]]
        return out


def run(scope, scan_plan: ScanPlan, out_dir, **kwargs) -> ScanResult:
    """Visit every field, acquire, and write frames plus a manifest.

    Blocking. A GUI should drive :func:`run_iter` instead, which yields
    between fields so the page stays responsive and can be stopped.
    """
    result = None
    for result in run_iter(scope, scan_plan, out_dir, **kwargs):
        pass
    return result if result is not None else ScanResult(directory=Path(out_dir))


def run_iter(scope, scan_plan: ScanPlan, out_dir, *, channel: str = "",
             refocus_every: int = 1, progress=None,
             should_stop=None, pixel_size_um: float | None = None):
    """Acquire the scan, yielding the running result after every field.

    A field that fails is recorded and skipped — a survey of 361 fields must
    not be lost because one of them could not focus. The manifest is written
    at the end *and* whenever a well starts, so a crashed or stopped scan
    leaves a usable partial dataset rather than a folder of anonymous TIFFs.
    """
    import numpy as np
    import tifffile

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    say = progress or (lambda _msg: None)
    result = ScanResult(directory=out)

    if channel:
        scope.set_channel(channel)
    px = pixel_size_um if pixel_size_um is not None else scope.pixel_size_um()
    started = time.time()
    last_well = None

    for i, fld in enumerate(scan_plan.fields):
        if should_stop is not None and should_stop():
            say(f"stopped after {len(result.frames)} frame(s)")
            break
        if fld.well != last_well:
            say(f"well {fld.well} ({i + 1}/{scan_plan.n_fields})")
            last_well = fld.well
            _write_manifest(result, scan_plan, scope, px, started, channel)

        try:
            scope.move_xy(fld.x, fld.y)
            locked = None
            if refocus_every > 0 and i % refocus_every == 0 \
                    and scope.pfs_available():
                locked = scope.engage_pfs()
            frame = np.asarray(scope.snap())
            name = FRAME_FMT.format(well=fld.well, index=fld.index)
            tifffile.imwrite(out / name, frame)
        except Exception as exc:                        # noqa: BLE001
            result.failures.append({"name": fld.name, "well": fld.well,
                                    "index": fld.index,
                                    "x": fld.x, "y": fld.y,
                                    "error": f"{type(exc).__name__}: {exc}"})
            yield result
            continue

        result.frames.append({
            "file": name, "well": fld.well, "index": fld.index,
            "x": fld.x, "y": fld.y,
            "z": _safe(scope.z) if scope.has("focus") else None,
            "pfs_locked": locked,
            "shape": list(frame.shape),
            "dtype": str(frame.dtype),
            "t": round(time.time() - started, 3),
        })
        yield result

    _write_manifest(result, scan_plan, scope, px, started, channel)
    say(f"{len(result.frames)} frame(s), {len(result.failures)} failure(s)")
    yield result


def _safe(fn):
    try:
        return round(float(fn()), 3)
    except Exception:
        return None


def _write_manifest(result: ScanResult, scan_plan: ScanPlan, scope,
                    pixel_size_um: float, started: float,
                    channel: str) -> None:
    """Write the manifest. Called per well too, so a crash leaves it usable."""
    payload = {
        "kind": "well-scan",
        "version": "1.0",
        "plate": scan_plan.plate,
        "wells": scan_plan.wells,
        "grid": {"rows": scan_plan.rows, "columns": scan_plan.columns,
                 "overlap": scan_plan.overlap},
        "fov_um": list(scan_plan.fov_um),
        # The two numbers that turn a pixel in a frame into a stage
        # coordinate. Without them the frames cannot be revisited.
        "pixel_size_um": pixel_size_um,
        "channel": channel,
        "objective": _text(scope.objective) if scope.has("nosepiece") else "",
        "elapsed_s": round(time.time() - started, 1),
        "frames": result.frames,
        "failures": result.failures,
    }
    try:
        result.manifest_path.write_text(json.dumps(payload, indent=2))
    except OSError:
        pass


def _text(fn) -> str:
    try:
        return str(fn())
    except Exception:
        return ""


def load_manifest(path) -> dict:
    """Read a scan manifest, from the folder or the file itself."""
    p = Path(path)
    if p.is_dir():
        p = p / MANIFEST
    return json.loads(p.read_text())


def stage_of_pixel(frame: dict, px: float, py: float, manifest: dict,
                   ) -> tuple[float, float]:
    """Stage coordinate of a pixel in one scanned frame.

    This is the arithmetic that makes a scan actionable: a detector finds a
    cell at pixel (px, py) of a frame, and this says where to drive to put
    that cell in the middle of the field.

    The frame's stage position is its **centre**, so pixel offsets are taken
    from the middle of the image. Y is inverted: image rows increase
    downwards while stage Y increases upwards.
    """
    pixel_size = float(manifest.get("pixel_size_um") or 0.0)
    if pixel_size <= 0:
        raise ValueError(
            "this scan recorded no pixel size, so pixel positions cannot be "
            "converted to stage coordinates. Re-scan with a pixel size set."
        )
    shape = frame.get("shape") or [0, 0]
    height, width = float(shape[0]), float(shape[1])
    dx = (float(px) - width / 2.0) * pixel_size
    dy = (float(py) - height / 2.0) * pixel_size
    return float(frame["x"]) + dx, float(frame["y"]) - dy
