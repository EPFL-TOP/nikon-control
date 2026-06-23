# Time-dependent ROIs — implementation notes

**Status**: implemented in schema v0.5. Sign-off on UX received
2026-06-23 (approach A, snap outside the range, ``t_start`` defaults to
the T at draw time).

## Problem

Most cells barely move during a time-lapse — one bbox covers the cell's
whole lifetime, which is what the current tool assumes. But:

- Debris drifts visibly frame to frame.
- Some cells migrate enough to leave a static bbox.

So we need annotations whose bbox can vary over time. A few options below.

## Proposal: keyframe model

Each annotation owns an ordered list of **keyframes** ``(t, bbox)``:

```python
@dataclass
class Keyframe:
    t: int
    bbox: list[float]   # [y0, x0, y1, x1]

@dataclass
class Annotation:
    label: str
    keyframes: list[Keyframe]      # at least one; sorted by t
    t_start: int = 0
    t_end: int | None = None
    t_deaths: list[int] = field(default_factory=list)
    z: int = 0
    notes: str = ""
    created: str = ...
```

- **Static cells** (~95 % of annotations): `keyframes = [Keyframe(t=t_start, bbox=...)]`.
  The single keyframe holds for every frame in `[t_start, t_end]`.
- **Drifting debris / migrating cells**: 2+ keyframes; the displayed bbox
  at any frame is **linearly interpolated** between the surrounding two.

Interpolation makes the user's annotation work tractable — they don't need
to redraw every frame for a slow drift, just a few sparse keyframes.

## UX

Two operations, both starting from a selected shape:

**1. Add a keyframe to the selected annotation at the current T.**

- Scrub to a frame where the bbox needs to be in a different place.
- The selected shape is duplicated as a keyframe at the current T,
  initially identical in shape.
- The user adjusts the new keyframe's bbox by dragging / resizing.
- Internally we record `(current_T, new_bbox)` as an additional
  keyframe for that annotation.

**2. Delete a keyframe.**

- Scrub to the keyframe T.
- Click "Delete keyframe at current T" — removes only that keyframe.
- The annotation must always retain at least one keyframe (the original
  drawn bbox).

Display:

- At any current T, each annotation's bbox is the interpolation between
  its surrounding keyframes (or the closest keyframe outside the range).
- Visibility (alpha 0/1) is still gated by `[t_start, t_end]`.
- Death labels still gated by `current_t >= t_death`.

## napari implementation sketch

Two layer architectures are plausible:

### (A) `ndim=2` shape layer with computed bboxes

- Keep the existing ndim=2 shape layer; one shape per annotation, always
  visible at every T.
- On every T change, recompute each shape's vertex coords from the
  keyframe list, and update `layer.data` in place.
- Editing in the GUI: detect when the user moves a shape and capture
  the new bbox as a keyframe at the current T.

**Pros**: bbox always present at every T; spatial continuity is obvious.
**Cons**: editing detection (mouse-up event) is napari-version-dependent
and a bit fragile; rebuilding `layer.data` on every T change has more
overhead.

### (B) `ndim=3` shape layer with one shape per keyframe

- Switch to ndim=3 — each shape is at a specific T.
- A "track id" property links the keyframes of one annotation.
- Drawing a new shape adds a keyframe; the user must click "link to
  selected" to attach the new shape to an existing annotation, else it
  starts a new one.
- Between keyframe T values, napari shows nothing; we add a *second*
  read-only shape layer of ndim=2 rendering the interpolated bboxes for
  display only (alpha 1 at non-keyframe frames, click-through off).

**Pros**: edits are first-class napari operations, no event-loop tricks.
**Cons**: two layers per class doubles the layer-list clutter; the
"link to selected" step is friction.

My lean is **(A)**. Less napari layer clutter, edits become "drag the
shape at the right T to make a new keyframe". The fragile bit (editing
detection) can be handled with a magicgui button — "Update keyframe at
current T" — which captures the current bbox shape. That keeps the
napari interactions standard.

## Workflow if (A)

1. User draws a box at some T. Annotation created with one keyframe at
   `t_start = current T` (or 0 — see open question).
2. The same bbox is shown across all T in `[t_start, t_end]`.
3. Scrub to a later T where the cell has moved.
4. Select the bbox, drag/resize it to the cell's new position.
5. Click **"Add keyframe at current T"** in the Lifecycle dock.
6. Internally: that bbox is recorded as a new keyframe at `current_T`.
   At intermediate T values, the displayed bbox interpolates between
   the previous keyframe and the new one.

Without step 5, the user's drag would either be a no-op (we don't capture
it) or it would override the existing keyframe. Step 5 is the explicit
"this is now a keyframe" gesture.

## JSON schema bump v0.5

```json
{
  "schema_version": "0.5",
  ...
  "annotations": [
    {
      "label": "debris",
      "keyframes": [
        {"t": 0,  "bbox": [10, 20, 100, 200]},
        {"t": 20, "bbox": [30, 25, 120, 205]},
        {"t": 60, "bbox": [80, 30, 170, 210]}
      ],
      "t_start": 0,
      "t_end": null,
      "t_deaths": [],
      "z": 0,
      "notes": "",
      "created": "..."
    }
  ]
}
```

v0.4 → v0.5 migration: existing `bbox` becomes `keyframes: [{t: t_start, bbox: bbox}]`.

## Locked-in answers

1. **Approach A** (ndim=2 shape layer, drag-and-keyframe).
2. **Snap** outside the keyframe range to the nearest end-keyframe.
3. **t_start = current_T** when a new bbox is drawn.

## Implementation summary (v0.5)

- ``Keyframe`` dataclass added; ``Annotation.keyframes`` is the source of
  truth. The legacy ``bbox`` attribute survives as a Python ``@property``
  returning the first keyframe's bbox for callers that don't yet know
  about the multi-keyframe model, but it's no longer stored in the JSON.
- napari stores the per-shape keyframe list as a JSON string in
  ``properties["keyframes_json"]`` — same workaround as ``t_deaths_json``
  because napari shape-layer properties are 1-D scalar arrays.
- ``_refresh_one_layer`` runs on every T change. For each shape it
  decodes the keyframes, calls ``_interpolate_bbox`` to compute the bbox
  at the current T, and overwrites ``layer.data[i]`` if it changed.
  Reentrancy is guarded by a module-local ``refreshing`` set so the
  ``layer.data = ...`` assignment doesn't re-trigger the data event.
- New shapes are detected in ``_on_data_change``: any shape whose
  ``keyframes_json`` is empty is treated as freshly drawn, its bbox
  read from ``layer.data[i]``, and seeded as the single keyframe at
  the current T (which also becomes ``t_start``).
- The Lifecycle dock has a new **⊞ ROI keyframes** row with two buttons:
  *Add @ current T* (captures the shape's current bbox as a keyframe)
  and *Drop @ current T* (removes the keyframe at the current T;
  refuses to leave a shape with zero keyframes).
- Schema migration v0.4 → v0.5: each annotation's old ``bbox`` becomes
  a single keyframe at ``t_start``. Round-tripping a v0.4 file through
  load/save produces an equivalent v0.5 file.
