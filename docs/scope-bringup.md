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

### Roles with more than one candidate

`build`, `stand` and `config` print a *chosen from several candidates* block
whenever a role had competition. Read it — this is where a wrong pick hides.
On the lab's Ti2-E, **four devices are typed `XYStage`**: the stage and three
TIRF illuminator positioners. Driving the wrong one looks exactly like
broken stage hardware: the readout sits at 0,0 and nothing moves. TIRF
drives are now excluded outright, and the remaining contests are shown:

```
  chosen from several candidates (override in the .cfg if wrong):
    objective turret   Nosepiece    (also: CondenserTurret)
    shutter            DiaLamp      (also: Turret1Shutter, Turret2Shutter)
```

`DiaLamp` is the transmitted-light source, which is the right shutter for
brightfield; `Turret1Shutter` / `Turret2Shutter` are the epi shutters you
will want for fluorescence. To change one, edit the `Property,Core,Shutter,…`
line in the `.cfg`.

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

### Lamp intensity, and any other knob

Micro-Manager has an API for the stage, the focus and the shutter, but **not
for brightness** — a lamp's intensity is a plain device property whose name
differs per vendor. To see the real names:

```bat
nikon-control-scope config MMConfig.cfg --properties
```

which dumps every property of every device with its limits and allowed
values. The dashboard finds the intensity knob automatically (searching only
devices that could plausibly be a light source, so a camera's
"BeadBrightness" is not mistaken for a lamp) and **labels the slider with the
property it is driving**, e.g. `Intensity — DiaLamp.Intensity`. If it finds
nothing, the slider says so instead of pretending. From a script:

```python
scope.set_intensity(40)                       # the found knob
scope.set_property("DiaLamp", "Intensity", 40)   # or name it yourself
for p in scope.properties("shutter"):
    print(p.describe())
```

### Is the light switched on and off automatically?

Yes, when **auto-shutter** is on — which the generated config sets whenever
there is a shutter. MMCore then opens the Core shutter device before each
acquisition and closes it after, for snaps and for every live frame.

For brightfield that is often not what you want: at a few frames a second
the lamp is being switched constantly. Turn auto-shutter off in the light
panel and open the shutter once; the lamp then simply stays on. For
fluorescence, leave auto-shutter on — it minimises the light the sample
sees, which is the entire point.

### Defining channels

A "channel" in Micro-Manager is a **config group preset**: a set of
`device, property, value` settings applied together. Rather than writing
those by hand, set the microscope up by eye and capture what it is doing.

**From the dashboard** — the `/scope` **Channels** tab: set the light path,
filter turret, shutter and intensity by eye, type a name, press *Capture
current settings*. The preset is defined in the running core (so it works
immediately) **and** written to the `.cfg` (so it survives a restart), and it
appears in the Channel menu at once.

**Or from the command line:**

```bat
nikon-control-scope channel --config MMConfig.cfg --name BF
nikon-control-scope channel --config MMConfig.cfg --name GFP --with-exposure
nikon-control-scope channel --config MMConfig.cfg --list
```

It captures the shutters, filter and condenser turrets, the light path and
the lamp intensity. It deliberately does **not** capture the objective
turret, the stage or the focus: a channel that rotated the turret would
swing an objective under a loaded plate on every BF→GFP switch, and one that
moved the stage would teleport the sample. Re-capturing a name replaces that
preset rather than leaving two conflicting definitions in the file.

Presets appear in the dashboard's **Channel** menu after a reload, and in a
script as `scope.channels()` / `scope.set_channel("GFP")`.

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

## 6 · PFS: engaged, in range, and at the right offset

Three different things, and only the third decides whether the image is
sharp:

| state | means |
|---|---|
| **engaged** | continuous focus is switched on |
| **in range** | the IR beam can see the coverslip at all |
| **offset** | where the focal plane sits *relative to* that coverslip |

**A lock at the wrong offset focuses on the glass, not the cells.** PFS
reports everything is fine and the image is blurry — which is the single most
confusing failure on this microscope, because nothing looks broken.

The procedure:

1. PFS **off**. Use **Z ±** to bring the cells sharp by eye.
2. **Engage PFS.** It locks, and may jump away from your plane — that is
   expected, it has gone to whatever offset was stored.
3. Use **Offset ±** to bring the cells sharp again. Do **not** use Z here:
   PFS pulls straight back, which is exactly what "Z does nothing" feels like.
4. Note the offset. It is a property of the dish and objective, not of the
   sample, so it is reusable.

Z and the offset are separate controls in the dashboard on purpose. An
earlier version had one **Focus ±** that silently switched between them, with
a step labelled in µm — but the PFS offset is in the offset device's own
units, and one unit is not one micron. A control that lies about its units is
worse than two controls.

Two more things worth checking when PFS will not hold: the **objective in
use** (its offset range differs per objective — `nikon-control-scope config
MMConfig.cfg --properties` shows `Nosepiece.Label`), and `PFS.LEDIntensity`,
the search LED, which can be too low for a dish with weak reflectivity.

## 7 · Register the plate against the stage

This is the step that makes every later position meaningful. A plate's
geometry is fixed by its manufacturer, so locating it takes exactly **three
numbers**: where the centre of well A1 sits on the stage, and how the plate is
rotated relative to the stage axes.

### You cannot centre a well by eye, and you do not need to

At 40× the field of view is ~333 µm and a 96-well well is 6400 µm across —
the well is not even visible, let alone centrable. Two facts make that a
non-problem.

**1 · The tolerance is hundreds of microns, not microns.** A registration
error shifts the whole scan grid by that amount; it does not compound across
the plate. Scan a centred 11×11 of a well that would take 20×20 to cover and
there is ~1.7 mm of spare margin on each side. So ±300 µm per reference is
fine — and the tool now says so instead of warning at 50 µm, which was far
too strict and would have made a perfectly good registration look broken.

Rotation is the part worth care, because it acts over the whole plate. That
is why the references should be far apart: with ±300 µm references on a
100 mm baseline, the fitted rotation is good to ~0.17°, which displaces the
far corner by ~300 µm — the same order as the input noise, not worse.

**2 · Opposite walls beat a guessed centre.** Drive until the well *wall*
sits mid-image, capture; do the opposite side; the centre is the midpoint. No
judgement about where a centre is, works at any magnification, and the
implied diameter is a free check — touch a wall of the wrong well and the
number comes out 40% off, which the tool refuses.

The Plate tab has four wall buttons (`◀ −X wall`, `+X wall ▶`, `▼ −Y wall`,
`▲ +Y wall`) and *Centre from walls*. One wall per axis also works, offset by
the nominal radius; two per axis is better because nothing is assumed.

**Or just calibrate at 4× or 10×.** At 4× the field is ~3.3 mm, so you can
see the well and its wall and eyeball the centre to a few hundred microns —
inside tolerance. The objectives are not parcentric, but that offset is a
**constant translation**: it shifts every reference equally, so it lands in
`a1_center_xy` as a fixed bias of tens of microns. (Parcentricity is a real
problem only when you must hit a specific *cell* across a magnification
change, a ~10 µm job. It does not apply here — worth saying, because it is
easy to over-apply.)

### Then register

Either from the dashboard's **Plate** panel — pick the plate type, drive to
each suggested well, capture its centre (by walls, or *Centre is here*), then
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

## 8 · Choose the wells to image

Once the plate is registered, the Plate tab's map is also the **well
selector**. Click wells to toggle them (the mode switch turns clicks back
into "drive there"), or type a range:

| typed | means |
|---|---|
| `A1` | one well |
| `B2-B5`, `A1:D6` | the rectangle between two corners |
| `C*` | all of row C |
| `*3` | all of column 3 |
| `all` | every well |

A typo is skipped rather than refused, and the count is shown — so compare
what you asked for against what you got. Fill colour is the imaging
selection; the gold outlines are the wells you measured for calibration, and
the two are independent.

**Serpentine order** is on by default: alternate rows are visited in reverse.
A raster drives back across the whole plate at the end of every row; on a
full 96-well plate that is more than twice the travel, which comes straight
out of the time budget below.

The selection is saved into the same JSON as the calibration, so one file
describes both where the plate is and which wells matter. From a script:

```python
from nikon_control.scope import plate, wells

cal = plate.load("plate.json")
sel = wells.load_selection("plate.json")
plan = sel.to_plan(cal)            # a useq.WellPlatePlan
for pos in plan.image_positions:
    print(pos.name, pos.x, pos.y)
```

## 9 · Scan the selected wells at 40×

The survey: visit a grid of fields in each selected well, save the frames,
and record **where each frame came from**. That last part is the point —
detection runs offline, and a cell's pixel position plus its frame's stage
position gives a stage coordinate to drive back to. A scan without per-frame
stage coordinates is just pictures.

The dashboard's **Scan** tab, or:

```bat
nikon-control-scope scan --wells A1:B3 --coverage 0.33 --out survey
nikon-control-scope scan --wells A1 --rows 11 --columns 11 --dry-run
```

### A pixel size is required

A field of view is pixels × µm/pixel, and `nikon-control-scope build` writes
no pixel-size configuration — there is nothing to derive one from. Without
it a grid would put every field in the same place, so the scan refuses rather
than guessing. Set it in the Scan tab (it suggests a value from the camera's
pixel pitch and the objective's magnification, which is a starting point to
check against a stage micrometer, not a calibration). It is stored per
objective, so it follows a turret move instead of silently going wrong.

For this rig: 6.5 µm camera pixels ÷ 40× = **0.1625 µm/px**, so a 2048²
frame is a **333 µm** field.

### How many fields

| coverage of a 6400 µm well | grid | frames/well |
|---|---|---|
| everything | 20 × 20 | 400 |
| half | 10 × 10 | 100 |
| the middle third | 7 × 7 | 49 |

Full coverage is rarely what a survey wants. *Fit grid to coverage* works out
the grid from a fraction, and the **Throughput** tab's measured timings turn
that into minutes.

`--refocus-every N` decides how often PFS re-engages: `1` at every field is
safest and slowest, and a well's fields are coplanar enough that once per
well is usually enough. On measured numbers, 361 refocuses at 400 ms is
~2.4 minutes of a single well's scan.

The survey is **brightfield** — `--channel` defaults to leaving the current
one alone. Firing fluorescence at every field of every well bleaches the
sample before the experiment starts.

### What comes out

A folder of 16-bit TIFFs plus `scan.json`, recording per frame: file, well,
field index, stage x/y, z, whether PFS locked, shape and elapsed time — and
at the top, the pixel size and objective. A field that fails is recorded with
its position and skipped; the manifest is rewritten at every well, so a
stopped or crashed scan still leaves a usable dataset.

```python
from nikon_control.scope import scan

m = scan.load_manifest("survey")
frame = m["frames"][0]
x, y = scan.stage_of_pixel(frame, 1024, 960, m)   # a detection -> the stage
```

## How long will a timepoint take?

The `/scope` **Throughput** tab measures this microscope — stage settling,
camera readout, PFS lock, channel switching — and answers "how many positions
fit in one interval?". Nothing there is a default: the numbers that matter
cannot be guessed, and anything it could not measure is listed as assumed.

```
per position = move + PFS lock + Σ channels(switch + expose + read)
```

Pick the channels, the interval and the movie length, and it reports the
per-position cost, how many positions fit, and whether the number you want
fits. 20% of the interval is held back as slack — a timelapse scheduled to
the full interval drifts later at every timepoint.

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

The acquisition loop itself. See
[acquisition-architecture.md](acquisition-architecture.md) for how the seven
plugins fit together, what is already in place, and what carries data between
them.
