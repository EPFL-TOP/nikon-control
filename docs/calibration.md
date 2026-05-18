# Objective cross-calibration — design proposal

**Status**: proposal. Awaiting sign-off before implementation.

## The actual problem

NIS already knows three things per objective from factory calibration:

1. **Pixel scale** (µm/px) — usually correct.
2. **Stage→image mapping** at a given objective — usually correct.
3. **Parcentric offset between objectives** — the part that is wrong on
   this scope. When the user switches from 10× to 40×, the optical
   centreline shifts by some (Δx, Δy) in stage coordinates. The
   software-assumed mapping treats the centreline as identical across
   objectives; that's the source of the centring drift the user sees.

Goal of the calibration: measure (Δx, Δy) for each ordered objective
pair `(low → high)` and persist it so the tile planner can correct for
it.

## Why not a custom 3D-printed probe

A custom probe would work, but it has friction:

- We need a feature visible and well-resolved at all three mags
  (4× FOV is ~hundreds of µm; 40× FOV is ~tens of µm). The feature has
  to be small enough to fit at 40× and crisp enough to localise at 4×.
- 3D-printed features at sub-mm scale are at the limit of FDM/SLA
  resolution. Anything we'd actually print probably isn't precise enough.
- We'd still need to derive a centroid algorithm per feature.

There are better, cheaper, more standard options.

## Methods (ranked)

### Method A — Image-content registration (recommended primary)

**Idea**: don't measure features; measure the parcentric offset directly
from any two images of the same sample at two magnifications.

Procedure (this becomes a JOBS Python task):

1. Centre the stage on any recognisable feature at the lowest mag (4×).
2. Without moving the stage, capture at 4×.
3. Switch objective to 10×, capture.
4. Switch objective to 40×, capture.
5. For each pair `(low → high)`:
   - Downsample the high-mag image to the low-mag pixel scale.
   - Phase-correlate (or run ECC alignment) against the low-mag image.
   - Recover a sub-pixel (Δx_px, Δy_px) offset.
   - Convert to micrometres using the high-mag pixel size.
6. Write `calibration.json`:

   ```json
   {
     "scope_id": "<some identifier>",
     "calibrated_at": "2026-05-18T15:00:00",
     "offsets_um": {
       "4x->10x": [dx, dy],
       "4x->40x": [dx, dy],
       "10x->40x": [dx, dy]
     },
     "method": "phase_correlation",
     "sample": "BF cells, well A1"
   }
   ```

**Pros**

- No special hardware.
- Runs on any structured sample — cells, debris, even a fingerprint.
- Cheap to repeat → users can recalibrate any time.
- Implementable today in Python (`scikit-image` has phase correlation,
  ECC, and feature-based registration ready to go).

**Cons**

- Assumes NIS' pixel-scale is correct. (Usually true; we can also
  validate it with the same images.)
- Live cells move during acquisition — use a fixed sample for the
  calibration run (PFA-fixed cells, or just a dust speck on slide).

**Precision estimate**: phase correlation reaches ~0.1 px in good cases.
At 40× with 0.16 µm/px that's ~16 nm. Vastly better than the centring
errors you see today.

### Method B — Single fluorescent bead (optional upgrade)

If Method A's precision is insufficient (it almost certainly isn't):

- Drop a single ~1 µm TetraSpeck bead on a slide.
- Image at each mag (use the fluo channel that matches the bead).
- Fit a 2D Gaussian to the bead in each image → sub-pixel centroid.
- Compute (centroid_high - centroid_low) → parcentric offset.

**Pros**: standard technique in super-resolution. ~50 nm precision.

**Cons**: bead procurement, careful positioning so the bead stays in
the small 40× FOV.

### Method C — Commercial calibration slide

A USAF resolution target (e.g. Thorlabs R1L3S1P, ~$200) or a stage
micrometer. Same workflow as A, but the feature is engineered for the
purpose, and it also validates pixel scale across mags.

**Pros**: stable, reproducible, vendor-traceable.

**Cons**: procurement, cost. Doesn't add anything over Method A for our
needs.

## Recommendation

Implement **Method A** as a JOBS Python task that:

- Captures at 4×, 10×, 40× without moving stage.
- Runs phase correlation pairwise.
- Writes `calibration.json` to a known location.
- Tail of the log file shows the computed offsets so the user can sanity
  check.

If we later find this precision insufficient (visible centring still off
at 40×), add Method B as an upgrade. We do **not** need a physical
probe.

## What the calibration buys us downstream

The tile planner (Phase 3 of the roadmap) takes a cell centroid in
**stage coordinates at 10×** and outputs a 40× stage position. Without
calibration, that's off by (Δx_{10→40}, Δy_{10→40}) — exactly the
centring drift the user wants to fix. With it, the 40× FOV lands where
the planner thinks it does.

Same logic for 4×→40× if we later confirm 4× detection is viable.
