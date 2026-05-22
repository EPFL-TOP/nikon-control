# nikon-control

Tooling to drive a Nikon microscope (NIS Elements + JOBS 6.20) for
single-cell imaging in well plates, with a cell-targeted 10× → 40× pipeline.

```text
  10x BF scan  ─►  detect cells  ─►  plan 40x tile path  ─►  40x time-lapse
  (one-shot per experiment, per well)             (repeats for the experiment)
```

The 10× scan and path planning happen **once** at the start of an experiment.
The 40× time-lapse then runs over the frozen point list for the experiment's
duration, optionally with adaptive bursts on cells showing a signal of
interest.

## Status

- **Phase 0 (data-path scaffold)**: done — loader, placeholder BF detector,
  CLI, overlay PNG, `jobs_log` helper, tests.
- **Phase 1 (microscope connection)**: *paused*. The
  [capability-discovery snippet](jobs/README.md) is ready to paste into a
  JOBS Python task on the next microscope visit; its output unblocks the
  first real microscope-control module.
- **Current focus (May 2026)**: two upstream workstreams opened in parallel,
  both runnable off-microscope on the Mac:
  - **Cross-calibration of 4×/10×/40× objectives** — they are not
    parcentric; switching mag shifts the FOV centre. Plan: a physical
    calibration target + a JOBS macro that derives per-objective XY
    offsets. Without this, the 40× tile placement from a 10×-detected cell
    list is systematically misaligned, so it's a prerequisite for the tile
    planner.
  - **Cell detector at 4× / 10× with annotation dashboard** — a 4× scan
    takes under a minute; if we can detect ~90 % of cells at low mag, the
    40× tile planner has a real input. Detector must distinguish single
    cells from doublets, with the category list left extensible.
    Annotation dashboard (likely napari-based) on existing 40× ND2 data is
    the first concrete step.

The 4-phase roadmap below is still the destination; these two tracks feed it.

## Quick start (Mac, dev)

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e '.[nd2,dev]'
python -m nikon_control.cli data/some_10x_BF.nd2 --out out/overlay.png
pytest
```

The `nd2` extra is optional — omit it if you only have `.tif` test images.

## Annotation tool

To label cells in ND2 files for detector training:

```sh
pip install -e '.[annotate]'
nikon-control-annotate path/to/file.nd2
nikon-control-annotate path/to/file.nd2 --classes single doublet dividing
```

Opens the file in napari. One shape layer per class, colour-coded
(red `single`, yellow `doublet`, cyan `debris`).

**To draw boxes:**

1. Click the class name in the **layer list on the left** (e.g. `single`)
   to make that layer active.
2. In the **layer toolbar** (small icons that appeared above the layer
   list), pick the **rectangle tool**.
3. Draw around each cell. The bbox is visible at every timepoint — cells
   don't move much, no need to redraw.

**Lifecycle model.** Each bbox has three optional fields, exposed as
button rows in the **Lifecycle** dock on the right:

- **`t_start`** — first frame the cell/debris is visible (defaults to 0).
- **`t_end`** — last frame visible (e.g. cell drifts out of FOV). Default
  unset = visible until the end of the recording.
- **`t_death`** — frame the cell is marked dead. May be earlier than
  `t_end` if the corpse is still visible afterwards.

Bboxes are **hidden entirely** outside their visibility range
`[t_start, t_end]`. A `↑T=N` label sits above the box whenever it's
visible and `t_start > 0`; a `†T=N` label appears alongside only at
frames at or after `t_death` (so scrubbing back before death shows the
box without the dagger).

**To mark any lifecycle field:**

1. Scrub the T slider to the relevant frame.
2. **Switch the layer toolbar from the rectangle tool to the *select
   shapes* tool** (arrow icon, second from the left). This is the step
   most users miss — leaving the rectangle tool active means clicking on
   a box draws a new one instead of selecting it.
3. Click the box.
4. In the **Lifecycle** dock, click the relevant **Mark @ current T**
   button under Birth, End, or Death.

**To clear a mark**, same select flow, then click the matching **Clear**
button. Each clear resets just that one field — birth back to 0, end
back to "visible until end", death back to unset.

If you ever see "WARNING: ... did NOT persist", the napari install is
misbehaving — tell me with the napari version.

**To save:** click **Save annotations** in the right dock. Output is
`<file>.annotations.json` next to the ND2. The status bar reports how
many annotations were saved and how many have a death marked.

Schema documented in [src/nikon_control/annotate.py](src/nikon_control/annotate.py).
Bounding boxes only (no masks). Each annotation has optional `t_start` and
`t_end` recording the cell's lifecycle; defaults mean "alive the whole
recording".

### Training-format export (planned, not built)

Training frameworks (Ultralytics YOLO, Detectron2, mmdetection) want
images on disk with sidecar annotations, not raw ND2s. When we have
enough annotated data we'll add `nikon-control-export`, which reads
`.annotations.json` files and dumps either:

- **Full-frame TIFFs** with per-frame bboxes in COCO JSON (for detectors).
- **Per-cell crops** at a fixed size (for classifiers).

Not built yet — format depends on which trainer we land on. Tell me when
you're close to needing it.

For multi-user deployment on a Windows server (RDP install, shared venv,
launcher batch file, troubleshooting), see
[docs/windows-deploy.md](docs/windows-deploy.md).

## Design proposals (awaiting sign-off)

- [Calibration approach](docs/calibration.md) — image-content registration
  via phase correlation, no physical probe. Implemented as a JOBS Python
  task; produces a `calibration.json` with the (Δx, Δy) parcentric offsets
  between 4×/10×/40×.
- [Experimental sessions](docs/experimental-sessions.md) — three session
  types (bulk 40× collection, multi-mag annotation seed, periodic
  calibration check) and the plan for propagating 40× annotations to 10×
  and 4× image space.

## The experiment we want to automate

The microscope images single cells in well plates. The plate is fixed — it
does not move. Each well is one experimental condition (treatment, etc.); a
single experiment may image one well or several.

### Today's manual procedure

For each well of interest:

1. Live mode, **10× BF**. Focus on the glue at the well border with autoscale
   on so the image is not too bright/dark.
2. Joystick across the well; for each candidate cell:
   - right-click "center" to put the cell in the FOV centre,
   - record the position.
   Cells too close to the border, or near dust or filaments, are rejected.
3. Click **Optimise** to reorder the recorded path for shortest stage travel.
4. Switch to **40×**. For each recorded position:
   - refocus Z (and possibly XY by joystick),
   - engage **Perfect Focus (PFS)**,
   - **lock XY, lock Z, lock PFS** — three clicks per point,
   - move to the next point.
5. Repeat per well.
6. Re-check time-lapse parameters (interval, total duration, channels), click
   **Run**.

### Automation goals (target dashboard)

Preferred direction: an **external Python dashboard** orchestrates as much of
steps 1–5 as possible, using JOBS as the acquisition primitive (rationale in
[NIS-Elements integration model](#nis-elements-integration-model) below).

What the dashboard should let the user do:

- **Plate view**: render the well plate; click to select wells to image.
- **Per-well budget**: enter target cells/well and Δt between time-lapse
  acquisitions. Dashboard estimates whether
  `N_wells × N_cells × t_per_position < Δt` and warns if too ambitious.
  `t_per_position` is dominated by stage move + PFS settle + per-channel
  exposure — measurable empirically once we have hardware.
- **Adaptive sampling**: if a cell shows a "signal of interest", switch it
  to a higher acquisition rate. Requires live segmentation and per-cell
  intensity time-series in the dashboard.
- **Find well limits**: detect each selected well's spatial bounds (either
  from the plate calibration NIS already has, or by detecting the dark
  border ring in BF).
- **10× autofocus**: built-in JOBS autofocus to start; long-term, a small
  CNN trained on (out-of-focus image → Δz direction & magnitude) so we
  don't need a full z-stack each time.
- **10× scan + cell detection**: own model — Cellpose fine-tuned, or a
  Stardist-class detector — operating on stitched 10× BF tiles.
- **40× tile planning**: cover the maximum number of cells with the minimum
  number of 40× tiles, avoiding cells on tile edges (cropped) and
  duplicate cells across tiles.
- **40× Z + PFS per tile**: build a sparse focus surface across each well
  at setup time, interpolate Z for each tile, engage PFS to lock.

### Notes on the open questions

A few of the items above are non-trivial; recording where the head of each
problem is so future sessions don't relitigate them:

- **40× tile planner — algorithm.** Effectively a max-coverage problem with
  axis-aligned rectangles of fixed size W×H over a 2D point cloud.
  First-pass heuristic: greedy sliding window — at each step find the
  rectangle position containing the most uncovered points, mark them
  covered, repeat. Add a safety margin so cells don't sit on tile edges.
  Tractable up to thousands of cells per well. Better-than-greedy
  algorithms exist (LP relaxation, local search) but greedy is good enough
  to start.
- **Focus per 40× tile.** Don't try to AF every tile from scratch. Build a
  per-well **focus surface** (~5–10 probe points spread across the well) at
  experiment setup, interpolate Z for each tile, then engage PFS to lock.
  JOBS already has a Focus Surface task — example at
  `../nikon-microscopy/JOBS-examples/NIS_v6.10/10-Create_focus_surface_slide/`.
- **CNN-for-focus.** Real and published (DeepFocus / FocusNet style).
  Strong research line but training-data curation is non-trivial. Defer
  until Phase 1 is running with built-in AF; revisit as an upgrade.
- **Cell detector at 10×.** ResNet-18 alone is a classifier, not a
  segmenter. For "find cells in a BF tile" we want either Cellpose
  (works decently on BF out of the box, fine-tunes well) or a U-Net /
  Stardist trained on EPFL data. Start curating training crops from the
  manual workflow as soon as Phase 1 is producing 10× scans.
- **Adaptive sampling on signal.** Architecturally the hardest piece — it
  requires the dashboard or JOBS to *modify the running acquisition*.
  JOBS' Conditional Acquisition pattern can do it natively; if we want
  full dashboard control, the dashboard must push modified point sets
  back into JOBS mid-run. The External Form task (see below) was
  designed for exactly this kind of state sync, so it is worth
  prototyping early to see whether the round-trip latency is acceptable.

## NIS-Elements integration model

External connection to the microscope is supported, but the integration
model is **JOBS-anchored**, not the other way around — JOBS or NIS-Elements
must be running and orchestrating; external code attaches to that, not the
reverse. Three patterns are documented in
`../nikon-microscopy/JOBS-examples`:

### 1. Embedded Python inside JOBS

JOBS workflows can contain Python Script tasks that run inside the
NIS-Elements process.
([NIS_v7.01/61-Python_in_JOBs](../nikon-microscopy/JOBS-examples/NIS_v7.01/61-Python_in_JOBs/README.md))

- The `nis` module exposes the **entire NIS macro language as Python
  functions**: `nis.mac.PiezoXYMoveToXYPosition(x, y)`,
  `nis.mac.GA3_Execute(recipe, file, out)`,
  `nis.mac.Jobs_GetJobrunFolder(...)`, etc. Output pointers come back
  via `nis.ptr.double()` / `nis.ptr.char()` wrappers. This is effectively
  the SDK from inside.
- `limjob` adds direct stage primitives (`XY_GetPosition`, `XY_Move`,
  `Z_GetPosition`, `Z_Move`), pixel↔stage transforms
  (`Image.transformPxToStage`), and PointSet manipulation.
- `sys.path.append(R'C:\...\nikon-control\src')` lets a Python task
  import our git-versioned modules from anywhere on disk — no copying
  into the NIS install dir.
- Pre-installed packages: numpy, scipy, scikit-image, scikit-learn,
  matplotlib, pandas, requests, httpx, paramiko, uvicorn, pywin32,
  playwright, ome_types. Heavy ML libs (torch, cellpose) need separate
  conda envs invoked via subprocess.

### 2. External Form task

A JOBS task that **JOBS itself spawns** as a subprocess (a Python venv
running Uvicorn/FastAPI in the example). JOBS communicates with it over
HTTP on three configurable routes (`/health`, `/ui`, `/value`). The same
URL is reachable in a normal browser on the same PC, so it doubles as an
external dashboard with built-in JOBS state sync.
([NIS_v7.01/63-Simple_text_area](../nikon-microscopy/JOBS-examples/NIS_v7.01/63-Simple_text_area/README.md))

### 3. Embedded Python as HTTP client

Because the embedded Python in pattern 1 has `requests` / `httpx` /
`paramiko` / `subprocess` / `pywin32`, a JOBS Python task can act as a
*client* of an external server we run ourselves. This gives us pattern
2's benefits without depending on the External Form task — useful if it
turns out not to be available in v6.20.

### Version caveat — what's confirmed in v6.20

The cleanest `limjob` / `nis` API and the External Form task are
documented under v7.01. We have **6.20**. The v6.10 Python-in-JOBS
examples confirm `limjob` was already there, but External Form and the
DeviceManager `XY_Move` primitives may or may not be in 6.20. Verifying
this is part of [Phase 1](#coming-up--what-to-verify-on-the-microscope-pc).

### Target architecture

```text
   [microscope PC, Windows]
   ┌──────────────────────────┐
   │ NIS-Elements 6.20        │
   │  ┌────────────────────┐  │
   │  │ JOBS workflow      │  │  ── HTTP push (10× tile, 40× frame, status,
   │  │  Python tasks      │──┼─    intensities)                          │
   │  │  (thin: capture,   │  │  ── HTTP pull (point lists, exposure,     ▼
   │  │   stage moves,     │  │     adaptive triggers)
   │  │   nis.mac.* calls) │  │                          ┌────────────────────────┐
   │  └────────────────────┘  │                          │ nikon-control          │
   └──────────────────────────┘                          │ external server        │
                                                         │   FastAPI + dashboard  │
                                                         │   segmentation         │
                                                         │   tile-set-cover       │
                                                         │   focus-surface        │
                                                         │  (Mac dev → Win deploy)│
                                                         └────────────────────────┘
```

The JOBS-side stays thin (capture, stage move, push image bytes).
Heavy logic — segmentation, tile planner, dashboard rendering — lives in
this repo, developed and tested on the Mac.

## Phase 0 — what's in the repo

Phase 0 is *not* about detecting cells well. It is to prove the data path:
load a real 10× BF image into Python, run something, draw an overlay.

- `src/nikon_control/io.py` — `.nd2` / `.tif` loader.
- `src/nikon_control/detect.py` — placeholder BF detector (high-pass + Otsu).
- `src/nikon_control/cli.py` — `python -m nikon_control.cli IMAGE --out OVERLAY.png`.
- `jobs/` — notes for the JOBS-side workflows.
- `data/` — git-ignored, drop your `.nd2` / `.tif` files here.

## Roadmap

| Phase | Goal | Key deliverable |
| ----- | ---- | --------------- |
| 0 (done) | Data-path scaffold | Loader, placeholder detector, CLI, overlay |
| 1 (next) | Smallest possible end-to-end loop with the real scope | One JOBS Python task: stage-move → objective change → capture → push image to a local FastAPI dashboard that displays it live. No segmentation, no tile planning. Prove the connection. |
| 2 | Real cell detection | Trained BF detector (Cellpose fine-tune to start). 10× scan-per-well in JOBS, results posted to dashboard. |
| 3 | 40× tile planning | Stage↔pixel coordinate mapping, focus-surface per well, generate JOBS point list from dashboard. |
| 4 | Time-lapse + adaptive sampling | Run the time-lapse with live preview; trigger higher-rate bursts on cells showing a signal of interest. Most architectural risk lives here. |

## Coming up — what to verify on the microscope PC

To unblock Phase 1 we need to confirm what NIS-Elements 6.20 actually
exposes. When you next sit down at the microscope PC:

1. **Help → About → Modules**: list the licensed modules. Look for *SDK*,
   *C-API*, *Macro Interface*, *JOBS*. Tells us whether anything beyond
   the JOBS-anchored model is available.
2. **JOBS task palette**: open a JOBS workflow editor and check whether
   the **External Form** task is present. Determines whether pattern 2 or
   pattern 3 is our dashboard sync mechanism.
3. **Python Script task**: add one and confirm:
   - `import limjob` works.
   - `import nis` works.
   - `dir(limjob)` and `dir(nis.mac)` — paste the relevant subset back
     to me; this is what tells us which macro functions we have.
4. **Macro reference** (NIS' Macro Editor → Edit → Find macro function):
   look up the names of these — we will need them in Phase 1:
   - Stage XY move (likely `StgMoveXY`, `StgMoveTo`, or similar; piezo
     equivalents `PiezoXYMove*`).
   - Stage Z move.
   - **Objective change** (probably `ObjectiveChange`, `OptConfig*`, or
     part of an Optical Configuration call).
   - **PFS** engage / disengage / get-offset.
   - Single-image capture.
   - JOBS point set get / set.
5. **File output convention**: where does NIS save acquisitions by
   default on this machine? We need a known path the dashboard can watch
   for new files.

Once we have the answers, the first commit of `nikon_control.microscope`
will wrap whichever macro calls are present, and the first JOBS workflow
file will land in `jobs/` (binary, small).

## Reference

Sibling repo with Nikon's official JOBS examples:
`../nikon-microscopy/JOBS-examples`.

Most relevant examples for our pipeline:

- `NIS_v5.42/15-Mouse_embryo/` — lo-mag detect → hi-mag region loop.
- `NIS_v6.10/10-Create_focus_surface_slide/` — focus surface across a slide.
- `NIS_v6.10/11-Conditional_acquisition/` — GA3-driven conditional capture.
- `NIS_v6.10/61-JOBs-Python-1_variables/` — Python ↔ JOBS macro variables.
- `NIS_v7.01/61-Python_in_JOBs/` — `limjob` / `nis` API reference.
- `NIS_v7.01/63-Simple_text_area/` — External Form task with FastAPI.
