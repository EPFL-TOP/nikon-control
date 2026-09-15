# Bringing the microscope up on Micro-Manager

The scope is driven directly through Micro-Manager via `pymmcore-plus` — no
NIS-Elements in the loop. This document covers getting from "nothing works"
to "the plate is registered on the stage".

```bat
conda activate single-cells
pip install -e ".[scope]"
```

That extra pulls only `pymmcore-plus` and `useq-schema` — no torch, so the
control side installs on a machine that never trains anything.

## 1 · What can this machine drive?

```bat
nikon-control-scope adapters
```

Lists the Micro-Manager install, every installed device adapter, and a Nikon
readiness check. Two findings there are worth acting on before anything else:

- **`Ti2_Mic_Driver.dll` must sit beside the adapter.** The `NikonTi2`
  adapter is a thin wrapper over Nikon's closed SDK and fails to load without
  it. Copy it from the Ti2 Control installation into the Micro-Manager
  folder.
- **Check the Ti2 Control version.** 2.10 and 2.20 are documented to crash
  Micro-Manager when the nosepiece device is added
  (`mmCoreAndDevices` #44); 2.00 is reported working. This costs minutes to
  check and hours to debug.

The stand adapter does **not** provide the camera — Hamamatsu (`HamamatsuHam`)
and Photometrics (`PVCAM`) have their own.

## 2 · What does one adapter offer?

```bat
nikon-control-scope devices NikonTi2
```

Names, types and descriptions of every device that adapter can create — the
XY drive, Z drive, PFS, nosepiece, shutters — without loading any of them.

## 3 · Connect components one at a time

```bat
nikon-control-scope probe NikonTi2 TIXYDrive --properties
nikon-control-scope probe NikonTi2              :: every device it offers
```

`probe` loads and initialises a single device in a **throwaway core**, so a
device that hangs or crashes its adapter cannot take the rest of the
inventory with it. This is the honest test of whether hardware is connected
and talking, and it is how to bring a rig up incrementally: probe, fix, probe
the next.

Once a set of devices probes clean, build a `.cfg` with Micro-Manager's
Hardware Configuration Wizard and check it:

```bat
nikon-control-scope config C:\path\to\MMConfig.cfg
```

which lists the devices and shows which fills each core role — camera, XY
stage, focus, autofocus. An empty role is what makes a later script fail with
an unhelpful error.

## 4 · Register the plate against the stage

This is the step that makes every later position meaningful. A plate's
geometry is fixed by its manufacturer, so locating it takes exactly **three
numbers**: where the centre of well A1 sits on the stage, and how the plate is
rotated relative to the stage axes.

```bat
nikon-control-scope plate --plate 96-well --suggest
```

It names three wells — A1 and the two far corners. Use corners: adjacent
wells barely constrain the rotation, so the extra travel buys real accuracy.

Drive to each, centre it under the objective, read the stage coordinates
(Micro-Manager's stage control panel will do while there is no GUI of our
own), then:

```bat
nikon-control-scope plate --plate 96-well ^
    --well A1 1000 2000 --well A12 100000 2000 --well H1 1000 -61000 ^
    --json plate.json
```

Output reports A1's position, the fitted rotation, and the **residual** —
how far the measured wells sit from where the fit says they should. A few
tens of µm is normal for hand-centring; a few hundred means a well was
misidentified.

Two guards worth knowing:

- **One well is allowed** — it fixes the position and assumes zero rotation.
  Fine for a quick look, not for a plate scan.
- **A wrong plate type is refused, not fitted.** If the measured well spacing
  disagrees with the plate definition by more than 2%, calibration stops.
  Without that check the fit would happily "succeed" and put every position
  progressively further off across the plate.

### What the calibration gives you

It produces a `useq.WellPlatePlan`, which is the standard description of a
plate acquisition in this ecosystem. From it, every well's stage position and
every field inside a well follow from the plate geometry — nothing downstream
has to think in stage coordinates again:

```python
from nikon_control.scope.plate import WellRef, calibrate
from useq import GridRowsColumns

cal = calibrate("96-well", [WellRef("A1", 1000, 2000),
                            WellRef("A12", 100000, 2000),
                            WellRef("H1", 1000, -61000)])

plan = cal.to_plan(
    selected_wells=((0, 0, 1), (0, 1, 0)),               # A1, A2, B1
    well_points_plan=GridRowsColumns(rows=3, columns=3,  # 3x3 fields per well
                                     fov_width=133, fov_height=133),
)
for pos in plan.image_positions:
    print(pos.name, pos.x, pos.y)
```

That list of positions is what the 40× scan will iterate over.

## Testing without the microscope

`pymmcore-plus` ships Micro-Manager's demo devices, so discovery, probing and
config inspection can be developed and tested on any machine:

```bash
mmcore install          # fetches the device adapters (demo ones on macOS)
nikon-control-scope adapters
nikon-control-scope probe DemoCamera DXYStage
```

The plate calibration needs no hardware at all — it is pure geometry, and its
tests round-trip against `useq`'s own forward model.
