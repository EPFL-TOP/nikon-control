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
readiness check.

The stand adapter does **not** provide the camera — Hamamatsu
(`HamamatsuHam`), Photometrics (`PVCAM`) and Andor (`AndorSDK3`) have their
own.

## 2 · Which Nikon stand is this?

```bat
nikon-control-scope stand          :: add --notes for each stand's gotchas
```

This lab has both generations, and Micro-Manager drives them through
different adapters with different device names and different SDKs:

| | Ti2-E | Ti-E (older) |
|---|---|---|
| adapter | `NikonTi2` | `NikonTI` |
| vendor DLL | `Ti2_Mic_Driver.dll` | `NikonTi.dll` |
| where it lives | `C:\Program Files\Nikon\Ti2-SDK\bin` | `C:\Program Files\Nikon\Shared\Bin` |
| how it is found | **must be copied** into the Micro-Manager folder | found on the system path |
| device list | built by asking the SDK what is attached | fixed, published by the adapter |

That last row is the one that costs time. **An empty device list means
opposite things on the two stands:**

- On the **Ti2**, `devices NikonTi2` printing nothing means the SDK could not
  be reached — almost always the missing `Ti2_Mic_Driver.dll`. It does *not*
  mean the adapter is absent.
- On the **Ti-E**, `devices NikonTI` prints eighteen devices on a laptop with no
  microscope in the room. A device list there is evidence of nothing; only
  `probe` is.

`stand` reports both, checks for each DLL, and resolves the roles (below).
Two more findings worth acting on before anything else:

- **Ti2 Control version.** 2.10 and 2.20 are documented to crash
  Micro-Manager when the nosepiece device is added (`mmCoreAndDevices` #44);
  2.00 is reported working. Minutes to check, hours to debug.
- **Ti2 power order.** Power on the stand *before* the controller box, or
  only the simulator device appears. `Ti2Sample.exe` from the SDK tests the
  connection without Micro-Manager, which separates an SDK problem from a
  Micro-Manager one.

### Roles, not device names

Nothing above `scope/stand.py` names a device. Each command reports **roles**
— XY stage, Z drive, PFS, PFS offset, objective turret, light path, shutter
— resolved from the device *type* first and the name only as a tiebreak. So
the same control code runs on both stands:

```
roles:
  XY stage          TIXYDrive          <- Ti-E         XYStage    <- Ti2
  Z drive           TIZDrive                           ZDrive
  PFS (autofocus)   TIPFSStatus                        PFStatus
  PFS offset        TIPFSOffset                        PFSOffset
  objective turret  TINosePiece                        Nosepiece
```

Type-first matters more than it looks: on the Ti-E, *three* devices report type
`Stage` — the Z drive, the PFS offset and the TIRF drive — and picking the
wrong one moves the wrong axis.

One trap applies to **both** stands and will matter as soon as there is
control code: setting the Z position programmatically **disables PFS**
(`micro-manager` #1815). Anything that moves Z during a timelapse has to
re-engage PFS afterwards or the focus silently drifts overnight.

## 3 · What does one adapter offer, and does it answer?

```bat
nikon-control-scope devices NikonTi2
nikon-control-scope probe NikonTi2 XYStage --properties
nikon-control-scope probe NikonTI              :: every device it offers
```

`probe` loads and initialises a single device in a **throwaway core**, so a
device that hangs or crashes its adapter cannot take the rest of the
inventory with it. This is the honest test of whether hardware is connected
and talking, and it is how to bring a rig up incrementally: probe, fix, probe
the next.

## 4 · Write the configuration file

A Micro-Manager `.cfg` is the list of devices to load and which role each
fills; nothing can be driven without one. Two ways to get it.

### Build it from what is actually attached (no GUI needed)

```bat
nikon-control-scope build --out MMConfig.cfg
```

or press **Build from hardware** in the `/scope` dashboard. Either one:

1. loads the stand's **hub** and initialises it;
2. asks the *initialised hub* what peripherals are really there
   (`getInstalledDevices`) — **this is the only way to learn a Ti2's device
   names**, since that adapter enumerates from Nikon's SDK and nothing can be
   known in advance;
3. loads each peripheral **one at a time**, keeping what initialises and
   reporting what does not, rather than failing the whole build on one bad
   device;
4. adds a camera (`--camera-adapter HamamatsuHam`, or it tries each installed
   one) — no stand adapter provides a camera;
5. resolves the roles and writes the file.

The result contains exactly the devices that answered, which is a much better
starting point than one listing everything the adapter *could* offer.
`--dry-run` prints it instead of writing; `--skip NAME` leaves a device out.
`TIDiaLamp` is skipped by default — it has crashed Micro-Manager with some
Nikon driver versions and nothing here drives it.

The stand has to be **powered on** for this to find anything, since step 2
asks the hardware.

### When the stand contributes nothing

If `build` reports devices from the camera only, the stand adapter did not
load. `stand --deep` then loads the DLLs **directly** and reports Windows'
own error, which MMCore hides behind one generic message:

```bat
nikon-control-scope stand --deep
```

- **WinError 126** — a dependency is missing. For a Nikon adapter that is
  nearly always the vendor DLL not sitting beside it. Fix it with:
  ```bat
  nikon-control-scope fix-driver --stand ti2
  ```
  which copies (never moves) `Ti2_Mic_Driver.dll` from the Ti2 SDK into the
  Micro-Manager folder, then re-checks. Micro-Manager's folder may need an
  elevated prompt.
- **WinError 193** — wrong architecture: a 32-bit DLL under a 64-bit
  Micro-Manager, or the reverse.
- **WinError 1114** — the DLL loaded but its init routine failed: a vendor
  SDK version mismatch, or hardware that is powered off.
- **WinError 127** — a missing entry point: the vendor DLL is a different
  version from the one the adapter was built against.

### Or use Micro-Manager's Hardware Configuration Wizard

The wizard ships with the Micro-Manager **GUI**, which the adapter download
may or may not include (`nikon-control-scope build` tells you if it is
there). Use it when a device needs a **pre-init property** that cannot be
guessed — a COM port, a camera model selection. Build the config there, then
point this project at the file.

### Why is the image black?

A camera with no shutter or lamp device in the configuration returns a black
frame and **no error**, which reads as a broken camera. The dashboard's light
panel says which case you are in — *no light control in this config* (a dark
frame is expected), *auto-shutter*, *shutter open*, or *shutter CLOSED* — and
`Scope.illumination()` returns the same in a script.

So a config with no stand has no light. Two further things to check once the
stand is in: the **light path** must send light to the camera rather than the
eyepieces, and on a brightfield rig the **dia lamp** must be on. That lamp is
`TIDiaLamp` on the Ti-E; it is included by default, and `--skip TIDiaLamp` is
the escape hatch if it destabilises your driver version.

### Check it either way

```bat
nikon-control-scope config C:\path\to\MMConfig.cfg
```

which lists the devices, shows which fills each **core** role (camera, XY
stage, focus, autofocus, shutter) and then the **stand** roles — the ones
MMCore has no slot for, like the PFS offset and the nosepiece. An empty role
is what makes a later script fail with an unhelpful error.


## 5 · Operate it — the `/scope` dashboard

```bat
nikon-control-dashboard --data-dir . --mm-config C:\path\MMConfig.cfg
```

then open `http://localhost:5006/scope`. It has three panels, in session
order: **Connect** (a `.cfg`, *Build from hardware*, or Micro-Manager's demo
devices to try the page with no microscope), **Drive** (snap/live image, stage jog, focus, PFS,
objective, channel) and **Plate** (below).

All the hardware behaviour lives in `scope/control.py`, not in the dashboard,
so everything the page does is equally available to a script — which is what
the unattended 40× scan will be:

```python
from nikon_control.scope.control import Scope

s = Scope.from_config(r"C:\path\MMConfig.cfg")
s.move_xy(12000, -4000)
s.engage_pfs()                     # returns False if it did not lock
img = s.snap()
```

Three behaviours are built into that class rather than left to callers:

- **Moving Z suspends and restores PFS.** Setting Z disables continuous focus
  on this hardware, so `move_z()` turns PFS off deliberately and back on
  afterwards instead of leaving it silently off for the rest of the night.
- **`focus_by()` picks the right control.** While PFS is locked it moves the
  **PFS offset** — the only thing that changes focus under a lock — and moves
  **Z** when PFS is off. The dashboard's `Focus ±` buttons use it, and the
  line under them says which one is live right now.
- **The turret needs `confirm=True`.** Rotating a nosepiece under a loaded
  plate can drive a dry 40× into glass, and this project images entirely at
  40×, so a turret move should read as the exceptional event it is. The
  dashboard hides it behind a checkbox for the same reason.

There is also a jog guard: a relative move over 5 mm is refused, because a
mistyped step is the classic way to crash an objective. Absolute moves are
never guarded — crossing the plate is legitimate travel.

## 6 · Register the plate against the stage

This is the step that makes every later position meaningful. A plate's
geometry is fixed by its manufacturer, so locating it takes exactly **three
numbers**: where the centre of well A1 sits on the stage, and how the plate is
rotated relative to the stage axes.

Either from the dashboard's **Plate** panel — pick the plate type, drive to
each suggested well, type its name and press *Capture current XY*, then
*Calibrate* — or from the command line:

```bat
nikon-control-scope plate --plate 96-well --suggest
```

Either way it names three wells — A1 and the two far corners. Use corners:
adjacent wells barely constrain the rotation, so the extra travel buys real
accuracy. Drive to each, centre it under the objective, capture it, then:

```bat
nikon-control-scope plate --plate 96-well ^
    --well A1 1000 2000 --well A12 100000 2000 --well H1 1000 -61000 ^
    --json plate.json
```

Output reports A1's position, the fitted rotation, and the **residual** —
how far the measured wells sit from where the fit says they should. A few
tens of µm is normal for hand-centring; a few hundred means a well was
misidentified.

The dashboard then draws **every well in stage coordinates with the
objective's position marked on it**, and the captured wells highlighted. That
map is the check on the registration: if the marker is not inside the well you
are actually looking at, the fit is wrong — far easier to see here than to
infer from a scan that missed. Arm *click a well to drive there* and the same
map becomes the navigator.

`--json` / the panel's **Save** write one file that both read, so a plate
registered at the microscope is available to a script, and
`nikon-control-scope plate --show plate.json` prints it back.

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

The `/scope` dashboard has a **Demo devices** button that connects to exactly
those, so the whole page — jog, snap, PFS, plate registration — can be driven
with no microscope in the room. That is also how it is tested: the suite
builds the real Bokeh document and fires the real handlers against the demo
core.

The plate calibration needs no hardware at all — it is pure geometry, and its
tests round-trip against `useq`'s own forward model.

## What is not built yet

The 40× multi-well scan: walk the selected wells and fields of a
`WellPlatePlan`, engage PFS at each, acquire, and hand the frames to the
detector. Everything it needs — positions from the plate registration, PFS
handling, snap — is in place; the scan itself is the next piece.
