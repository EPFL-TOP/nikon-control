# The acquisition loop: seven plugins, one document

The target experiment: **>10 h, ≥3 channels (BF + ≥2 fluorescence), many
cells across many wells, ~5 min interval, everything at 40×.** This note
says how the seven plugins fit together, what carries data between them, and
what already exists.

## The tool: pymmcore-plus + useq, not a new engine

We are not writing an acquisition engine. `useq.MDASequence` is the standard
description of a multi-dimensional acquisition in this ecosystem — positions,
channels, time plan, z plan, autofocus plan — and `CMMCorePlus.run_mda()`
executes it, with events we can hook. Two consequences worth stating plainly:

- **Every plugin's job is to produce the `stage_positions` and `channels` of
  an `MDASequence`.** Nothing below builds its own loop over wells.
- **Faro** (pertzlab, built on the same two libraries) is the feedback-loop
  framework for plugin 7. Staying on pymmcore-plus + useq now means it drops
  in later without rework.

## The shape: a pipeline over one run document

The seven plugins are stages of one pipeline, and the thing that makes them
independent is that each reads and writes a **run document** — one JSON file
per experiment — rather than calling the next stage directly.

```
plate calibration ─▶ wells ─▶ scan ─▶ detections ─▶ selection ─▶ path ─▶ MDASequence
       (1)            (2)     (3)        (4)           (5)        (6)        ▲
                                                                             │
                                             (7) status trigger ─────────────┘
```

Why a document rather than a chain of calls:

- **Each stage is re-runnable in isolation.** Re-select cells with a different
  filter without re-scanning 24 wells — a scan is the expensive step.
- **Every stage is inspectable.** "Why did it image *that* position?" is
  answered by reading the file, not by re-deriving it.
- **The GUI is a view, not the pipeline.** Each plugin gets a dashboard tab,
  and each is equally usable from a script — which matters because the real
  run is unattended overnight.
- **A crashed run resumes** from the last completed stage.

### The plugins, and where each stands

| # | Plugin | State | Reads → writes |
|---|--------|-------|----------------|
| 1 | Plate identify + calibrate | **done** | — → `plate` (type, `a1_center_xy`, `rotation`) |
| 2 | Well selector | **done** | `plate` → `wells` |
| 3 | Well scanner | to build | `wells` → `scan` (frames + their stage coords) |
| 4 | Cell identifier | model exists | `scan` → `detections` (bbox, class, score, stage coords) |
| 5 | Position selector | to build | `detections` → `positions` (+ the filter that chose them) |
| 6 | Path builder | to build | `positions` → `route` (ordered, with travel cost) |
| 7 | Status trigger | later | live frames → events that edit the running sequence |

**1 — Plate.** `scope/plate.py` + the dashboard's Plate tab. Registering a
plate is fitting three numbers (`a1_center_xy` + `rotation`); `useq` turns
those into every well and field position. "Identify" is plate *type*
selection today; automatic identification (barcode, or a low-mag corner
image) is a later refinement, not a blocker.

**2 — Well selector.** `scope/wells.py` + the Plate tab's map. Click wells to
toggle, or type `A1:D6` / `C*` / `*3` / `all`. Produces a
`useq.WellPlatePlan` with `selected_wells` set, and saves into the same JSON
as the calibration. Visiting order is **serpentine** by default — alternate
rows reversed, which more than halves the travel of a full-plate scan and so
comes straight off the time budget.

**3 — Well scanner.** A grid of fields per well
(`useq.GridRowsColumns` with the camera's real FOV) run as an `MDASequence` in
**BF only** — the survey does not need fluorescence. Output: one frame per
field, each with its stage coordinates. This is the expensive stage, which is
why its output is persisted.

**4 — Cell identifier.** The trained Faster R-CNN
(`detector.py`) applied to the scan frames. It already classifies
single/doublet/debris at 40×, and overlap suppression already exists. The one
new piece is arithmetic: **pixel bbox → stage coordinates**, via the camera
pixel size and the frame's own stage position.

**5 — Position selector.** Filters over detections, defined as we learn what
matters: class, score, distance from a frame edge, isolation from neighbours,
per-well quota, total count. Design note: keep filters **named and stored in
the run document**, so a run records why each position was chosen.

**6 — Path builder.** Ordering, not geometry — the positions are already
fixed by stage 5. A nearest-neighbour tour with 2-opt improvement is more than
enough for a few hundred points, and `timing.py` already models what travel
costs, so the objective function is measurable rather than assumed.

**7 — Status trigger.** Watch frames as they arrive, detect a per-cell state
(the oscillation pattern), emit an event. This is exactly Faro's domain.

## The constraint that shapes everything: the interval

A 5-minute interval over 10 h is 120 timepoints, and **every position must be
visited within each interval**. That budget is now measurable rather than
guessed — the dashboard's **Throughput** tab times this microscope's stage
settling, camera readout, PFS lock and channel switching, then answers "how
many positions fit?".

```
per position = move + PFS lock + Σ channels(switch + expose + read)
```

Three things follow, and they are design constraints on plugins 5 and 6, not
afterthoughts:

- **Channels multiply.** Three channels is roughly three times the per-position
  cost, so the position count drops by about three.
- **The path matters more than it looks.** Well-to-well hops dominate once
  positions are scattered; clustering per well is usually worth more than a
  clever tour.
- **Leave slack.** A timelapse scheduled to 100% of its interval drifts later
  at every timepoint. The planner holds back 20% by default.

So the **position count is an output of the timing budget**, not an input:
plugin 5's quota should be set from what fits, and the dashboard shows both
numbers side by side.

## Interface consequence

Each plugin is a tab that (a) shows the current run document stage, (b) has a
*Run this stage* button, and (c) writes its output back. The tabs are ordered
as the pipeline, and a stage whose input is missing says which stage to run
first rather than failing.

That is already how the `/scope` tabs are laid out — Drive, Plate, Channels,
Throughput — and the remaining plugins extend the same pattern.

## Not yet decided

- **Where frames are written.** OME-Zarr is the ecosystem default and what
  Faro uses; ND2 is what the existing annotation tooling reads. Probably
  OME-Zarr for new acquisitions with a reader added to `io.py`.
- **Whether detection runs during the scan or after it.** After is simpler and
  loses nothing for a batch survey; during is needed only if the survey must
  adapt as it goes.
- **How plugin 7 edits a running sequence** — pause-and-requeue versus a
  custom engine. Worth deciding only when the first six work.
