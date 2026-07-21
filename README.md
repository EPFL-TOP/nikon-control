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

## Annotation workflow

Two steps: an automated **pre-annotation** pass that places ROIs with the
trained model, then a **Bokeh dashboard** where a human reviews, curates,
and categorises. Both share the v0.5 annotation schema
([src/nikon_control/schema.py](src/nikon_control/schema.py)); annotations
are stored as `<file>.annotations.json` next to each ND2.

### 1. Pre-annotate with the model

```sh
pip install -e '.[detect]'
nikon-control-preannotate path/to/file.nd2 cell_detection_model.pth --device cuda
# → path/to/file.annotations.json   (omit --device to auto-pick; CPU on Mac)
```

Runs the trained Faster R-CNN on the brightfield channel at every
timepoint, links detections into tracks (IoU association), and writes one
provisional annotation (label `cell`) per tracked object, with a keyframe
per timepoint (RDP-simplified). The human then reclassifies each `cell`
as `single` / `doublet` / etc. in the dashboard.

### 2. Review & curate in the dashboard

```sh
pip install -e '.[dashboard]'
nikon-control-dashboard --data-dir path/to/folder --show
# multi-user / remote: add --allow-websocket-origin host:5006
```

Opens in the browser. **Open a file** without the terminal: use the
**Drive / volume** dropdown to switch drives (C:, E:, G:, …), the folder
field + Up / Refresh / Subfolders dropdown to navigate, then pick the ND2
from the dropdown and click **Load**. The detection model is picked the same way
(dropdown of `.pth` in the folder, or paste a path). `--data-dir` /
`--weights` at launch just set the starting folder — you can browse
anywhere from the page. Then:

- **Detect cells** — the `Detect cells (refresh)` button runs the model
  in-process (no terminal needed) and populates provisional `cell` tracks.
  Re-running refreshes the `cell` boxes but **keeps** anything you already
  reclassified. Point it at the model with `--weights` (below) or the
  "Detection model (.pth)" field (auto-filled from any `.pth` in the data
  folder).
- **Detect debris** — the `Detect debris (refresh)` button finds *moving*
  debris with no model: a temporal-median background is subtracted (static
  cell + fixed structures drop out) and the transients are tracked with a
  velocity tracker (debris moves too fast for the cell IoU tracker). Fixed
  bubbles may show up as short tracks — delete the ones you don't want.
- **Navigate time** — the T slider plus `◀ Prev` / `Next ▶` buttons.
- **Track numbers** — each box shows `#N class` so you can confirm a track
  stays on the same cell as you scrub; boxes are coloured per class (see
  the legend).
- **Draw / move / resize / delete ROIs** with the box-edit tool
  (delete = shift-click). A drag at any frame auto-records a keyframe
  there, so drifting debris tracks correctly.
- **Category** — tap a box, pick its class from the dropdown. Resize with
  the width/height spinners or the **±10%** buttons; add a fresh box with
  **➕ Add ROI**.
- **Lifecycle** — `Mark birth @T`, `Mark end @T` (leaves FOV),
  `Add death @T` (multiple allowed, e.g. a doublet's two cells), and
  `Mark division @T` for a cell that divides — the track reads as its
  category (e.g. single) before that frame and **doublet** after, and the
  box recolours at the division frame.
- **Channel + contrast** — annotate on BF, flip to mCherry-H2B or GFP to
  judge single-vs-doublet or death.
- **Save** writes back to the sibling JSON.

Boxes are hidden outside their `[t_start, t_end]` window; `↑T=` / `⑂T=` /
`†T=` markers show birth / division / death at the current frame. Default
classes: `single`, `doublet`, `debris`, `fission_fusion`.

To enable in-dashboard detection, pass the model path at launch:

```sh
nikon-control-dashboard --data-dir path/to/folder \
    --weights cell_detection_model.pth --show
```

**Typical workflow — classify a track and record its death:**

1. Click `Detect cells (refresh)` (or open an ND2 that already has a
   pre-annotation). Provisional `cell` boxes appear.
2. Tap a box (tap tool). It selects the whole track (all timepoints).
3. Choose `single` (or `doublet`, …) in the Category dropdown — the box
   recolours.
4. Scrub with the slider / `Next ▶` to the frame where the cell dies,
   then click `Add death @T`. (For a doublet, do it again at the second
   cell's death frame.)
5. If the cell leaves the FOV, `Mark end @T` at the last visible frame.
6. `Save annotations`.

The lifecycle/keyframe model is documented in
[docs/tracked-rois.md](docs/tracked-rois.md). The correctness-critical
logic lives in the GUI-free, unit-tested
[dashboard/state.py](src/nikon_control/dashboard/state.py).

### Legacy: napari annotator

The original `nikon-control-annotate` (napari) is retained but
**superseded** by the dashboard above — it proved unstable for multi-user
Windows/RDP use, which is why we moved to Bokeh. Its data model is
unchanged (now in `schema.py`).

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
