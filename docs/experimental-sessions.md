# Experimental sessions — proposed data-collection strategy

**Status**: proposal. Awaiting sign-off before adding any
microscope-side macros.

## The problem this addresses

We want a cell detector that works at 10× (and ideally 4×). Annotating
at 10× is hard by eye and 4× is essentially impossible. But we have
plenty of 40× imagery and annotators can label that confidently.

We need a strategy to convert easy-to-label 40× ground truth into
training data at 10× and 4× — given that the objectives are not
parcentric, so naive coordinate transfer is off.

## Three session types

The lab should be running three kinds of sessions in parallel. They
serve different purposes and have very different time costs.

### Session type 1 — Bulk 40× collection (already happening, keep going)

What the lab is already doing: pick cells at 40× → time-lapse →
generate ND2 files. Don't change anything.

**Uses**:

- Single-cell classifier (single vs doublet, vs dividing later).
- Per-cell intensity analysis pipelines.
- Downsampled-40× as **pretraining** signal for low-mag detectors
  (with the caveat that real 4×/10× appearance differs — see
  `project_dataset_2026-05.md`).

**Annotation effort**: per-cell bboxes via the napari tool. Sparse in
time (1–3 frames per ND2).

### Session type 2 — Multi-mag annotation seed (the new addition)

**Goal**: produce triplets of (4×, 10×, 40×) images of the *same* FOV
so that 40× annotations can be transformed into 10× and 4× image space
via the calibration.

**Per-well procedure** (estimated ~15 min/well once the macro exists):

1. Centre stage on the well at 4×.
2. Capture a 4× whole-well scan (1–2 tiles).
3. Switch to 10×, capture a stitched scan of the same area (~10 tiles).
4. Switch to 40×, capture **one frame** at each of N candidate cell
   positions identified from the 10× scan.
5. At every objective change, stage coordinates are recorded and the
   parcentric calibration is applied — so a (stage_x, stage_y) recorded
   at 40× has a known image-space location in the 4× and 10× scans.

**Annotation flow downstream**:

- Annotate cells at 40× (cheap, high SNR, easy).
- A script transforms each 40× bbox into:
  - the corresponding 10× bbox in the 10× scan,
  - the corresponding 4× bbox in the 4× scan.
- Human reviews the propagated boxes at 10× and 4× and corrects them.
  (Reviewing is much faster than annotating from scratch.)

**Output**: per-well, three annotated images at three magnifications,
labels consistent across them by construction.

**Volume target**: ~20–50 wells over a few sessions. Doesn't need to be
huge — the *consistency* across mags is what makes this data special.

### Session type 3 — Calibration check (rare, periodic)

The image-registration-based calibration from
[calibration.md](calibration.md). Run when:

- A new scope or after any optical maintenance.
- Periodically (monthly?) to detect drift.
- When centring at 40× starts feeling off again.

**Time cost**: 2–5 min once the macro exists.

## What we can do without finished calibration

Calibration finishes faster than the dataset. But if you want to start
session type 2 *before* the calibration macro exists, we can do an
**image-registration-based annotation propagation** instead:

1. Acquire 40× + 10× + 4× of the same well in immediate succession
   (stage may have parcentric offset but it's the same offset for the
   whole session).
2. After acquisition, register the three images by content:
   downsample 40× to 10× scale, phase-correlate against the 10× image
   to find their relative offset. Repeat for 10× ↔ 4×.
3. Use that learned transform — not the (broken) stage coords — to
   propagate annotations.

This is essentially using Method A from `calibration.md` per session
instead of as a once-off calibration. More work per session, but
unblocks dataset collection today.

## Annotation propagation tooling

To make the multi-mag flow real, we'll need:

- A small script that takes `(40x_image, 10x_image, 4x_image)` + a
  calibration file (or the per-session image registration) and outputs
  the transform matrices between them.
- An extension to the napari annotation tool that opens the triplet
  side-by-side: annotate at 40×, see auto-propagated boxes at 10× and
  4×, edit/confirm.

Neither exists yet. Both are scoped after the calibration approach is
agreed and the annotation tool sees real user load.

## Open questions for you

1. Are you OK with **Method A (image-content registration)** as the
   primary calibration approach, or do you want to start with beads /
   commercial slide? My recommendation is A.
2. For session type 2, can the lab realistically run ~20–50 multi-mag
   wells over the next few weeks? If not, we lean harder on
   downsampled-40× and accept worse transfer.
3. Do you want me to start on the **calibration macro** (Python task
   that captures 4×/10×/40× without moving stage and computes offsets)
   now, or wait until you've confirmed `import nis` works on the v6.20
   install?
