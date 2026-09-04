# From annotations to a training set

## The simplified annotation file

The simple dashboard (`/simple`) writes one sidecar per ND2:
`<file>.simple.json`. It is deliberately flat — a label is just
*(frame, box, class)*, because that is exactly what a per-frame detector
sees:

```json
{
  "kind": "per-frame-boxes",
  "schema_version": "1.0",
  "source": "G:\\PROJECTS-02\\Samuel\\pos1.nd2",
  "image_shape": [200, 3, 1024, 1024],
  "axes": ["T", "C", "Y", "X"],
  "channels": ["BF", "mCherry", "GFP"],
  "bf_channel": 0,
  "n_frames": 20,
  "classes": ["single", "doublet", "debris", "unlabeled"],
  "boxes": [
    {"t": 0, "bbox": [102.0, 340.0, 158.0, 402.0], "label": "single",
     "z": 0, "score": 0.87, "auto": false},
    {"t": 0, "bbox": [500.0, 220.0, 570.0, 300.0], "label": "doublet",
     "z": 0, "score": null, "auto": false}
  ]
}
```

- `bbox` is `[y0, x0, y1, x1]` in **pixels of the full frame** (the same
  convention as the tracked schema).
- `bf_channel` says which channel the boxes were drawn on — an exporter needs
  it to render the matching plane.
- `label` is one of `single` / `doublet` / `debris`, or `unlabeled` for a
  detection nobody has classified yet. **`unlabeled` boxes are never
  training data.**
- `group` links the boxes of one physical cell across frames. It exists so
  the annotator classifies a cell **once** rather than once per frame
  (detections are grouped automatically by IoU). Training ignores it — the
  labels are per-frame — but the exporter passes it through as `cell` so
  correlated boxes remain identifiable.
- `origin` records which detector produced a box (`"cells"` / `"debris"`),
  or `null` for a hand-drawn one. Re-running one detector replaces only its
  own output, so the cell and debris passes no longer wipe each other.
- `auto` is `true` while a box is exactly as the detector produced it, and
  becomes `false` the moment a human classifies, moves, or resizes it. Two
  uses: re-running detection only replaces `auto` boxes (so it can't destroy
  curation work), and an export can optionally keep only human-touched boxes.
- There is **no tracking**: the same cell on frames 0 and 5 is two
  independent boxes, by design.

## Read the JSON at training time, or export a dataset first?

**Export first.** Write a script that turns `*.simple.json` + `*.nd2` into a
frozen dataset, and train from that — do not have the DataLoader open ND2s.
Reasons, in order of how much they bite:

1. **ND2 random access is the wrong shape for training.** The reader is slow
   for scattered reads and is not reliably safe across DataLoader worker
   processes; on Windows (spawn) it is worse. Every epoch would re-decode the
   same planes.
2. **The ND2s live on a network share** (`G:`). Training runs on a local
   NVIDIA box and then on the A100 cluster in Docker — neither should depend
   on a mount of the microscope share. An export is portable.
3. **Reproducibility.** Annotators keep working. If training reads the JSONs
   live, the dataset silently changes under you and two runs aren't
   comparable. A frozen, versioned export is what you compare against.
4. **Splits and class balance are export-time decisions** (see below) and
   want to be recorded once, not recomputed per run.

Keep the export a **build artifact**: the ND2s + JSONs stay the source of
truth, and the exporter is re-runnable to produce `v2`, `v3`, … as
annotation grows.

## Running it

```bat
:: 1. export a frozen dataset from the annotations
nikon-control-export --data-dir "G:\PROJECTS-02\Samuel" --out dataset-v1
::    useful flags: --frame-step 5   (cut near-duplicate frames)
::                  --verified-only  (drop auto-labelled boxes nobody touched)
::                  --dry-run        (report what would be exported)

:: 2. train, warm-starting from the existing 1-class cell model
nikon-control-train --dataset dataset-v1 --out cell_classes_model.pth ^
    --init-from "E:\PROJECTS-01\Clement\cell_detection_model.pth" ^
    --epochs 20 --batch-size 4
```

The exporter prints a summary and warns when the data is thin (a single
source file, so validation is meaningless; or few distinct cells). Training
reports `train_loss` plus **AP@0.5 per class** and `mAP@0.5` each epoch, and
keeps the best checkpoint by val mAP.

The checkpoint is written as `{"model_state_dict", "classes", ...}` — the
same shape the package already reads, and `CellDetector` infers the class
count from the box predictor, so the trained 3-class model loads back
without changes.

## Using a trained model to pre-classify (closing the loop)

Once `nikon-control-train` has produced a multi-class checkpoint, point the
dashboard at it (`--weights`, or pick it in the page) and **Detect cells**
both detects *and* identifies: each cell arrives with `single` / `doublet` /
`debris` already selected, so annotating becomes checking rather than
labelling. `CellDetector` reads the class names from the checkpoint, so
nothing else needs configuring; the original 1-class
`cell_detection_model.pth` still works and simply leaves every box
`unlabeled`.

One class is assigned per **cell** (a majority vote over the frames the
tracker linked), so a single odd frame cannot split a cell's identity.

**Predictions are not ground truth until a human accepts them.** A
pre-classified box keeps `auto: true`, which means *unverified*:

- fix any wrong class by tapping the box and picking the right one (that
  clears `auto` automatically), then press **✓ Accept predictions on this
  frame** (or for the whole file) to confirm the rest;
- the exporter counts unaccepted predictions in `manifest.json`
  (`unverified_boxes`) and **warns** when they make it into a dataset;
- `--verified-only` excludes them entirely.

This matters: training on the model's own unreviewed guesses entrenches its
mistakes instead of correcting them. The workflow is only a speed-up if
somebody actually looks.

## Overlapping boxes (duplicates)

A multi-class detector runs non-maximum suppression **per class**, so one
object can come back as `single` *and* as `doublet`; the cell and debris
passes can also both fire on the same object. Both dashboards resolve this.

Overlap is measured as **intersection ÷ area of the smaller box**, not IoU.
"Overlaps by more than 70%" is a statement about the smaller box, and only
this measure catches containment — a small `single` inside a big `doublet`
has ~100% overlap by this rule but an IoU well under 0.3, which an IoU rule
would miss entirely.

When two boxes overlap beyond the limit, one survives:

- **Boxes a human made, classified, moved or resized always win** over
  detector output, and are never deleted for one. Two overlapping *human*
  boxes are both kept — removing deliberate work is never automatic.
- Otherwise, by **On overlap keep the**: `bigger` (default, score breaks
  ties) or `higher-scoring` (area breaks ties). Bigger is the default
  because a doublet legitimately contains singles, so preferring area keeps
  the doublet rather than a confident `single` inside it.

It runs **automatically after each detection pass**, per frame, and
**⧈ Remove overlapping ROIs** re-runs it by hand (e.g. after adding boxes).
`/review` has the same controls plus a **"with overlapping boxes"** filter
and a whole-dataset sweep. The exporter counts what is left in
`manifest.json` (`overlapping_boxes`) and warns.

## Reviewing a dataset before training

`/review` opens an exported dataset (COCO + TIFFs) and steps through the
images **the model will actually be trained on**, so what you check is the
real input rather than something re-derived from it.

- The **“with unverified predictions”** filter walks only the images holding
  unaccepted model guesses — the highest-value pass. Boxes marked `?` are
  those.
- Tap a box, hit a class button to correct it (that also verifies it);
  add/delete boxes; reposition without dragging (🎯 click-to-place, arrows).
- **✓ Mark reviewed & next** tracks progress, stored on the COCO image
  entries so it survives reopening.
- **💾 Save** rewrites `annotations/*.json`, keeping the originals as
  `*.json.bak`.

⚠ **Re-running the export regenerates the dataset from the sidecars and
discards review edits.** Review is the last QC step before training; fixes
you want permanently belong in `/simple`.

## One rule for annotators

**Label *every* cell and every piece of debris on a frame you annotate.**
Only frames with at least one label are exported, and everything unboxed in
an exported frame is treated as background — so a cell you skipped actively
teaches the model to ignore cells. If a frame is too crowded or ambiguous to
finish, leave it entirely unlabelled (it is then skipped) rather than
half-labelled.

## Export format

Full-frame 16-bit TIFF + a COCO JSON:

```
dataset-v1/
  images/pos1_t000.tif        # full frame, 16-bit, BF channel
  images/pos1_t001.tif
  annotations/train.json      # COCO
  annotations/val.json
  manifest.json               # provenance: sources, counts, split, versions
```

- **Full frames, not crops.** A detector must learn to find cells against
  background; cropping to boxes throws away the negatives.
- **Keep 16-bit.** Don't collapse to 8-bit — the model's own preprocessing
  does percentile normalisation, and 8-bit would discard that headroom.
- **COCO** because the model is a torchvision `fasterrcnn_resnet50_fpn`;
  COCO boxes are `[x, y, w, h]`, so `[y0,x0,y1,x1]` -> `[x0, y0, x1-x0,
  y1-y0]`. Categories: `single`=1, `doublet`=2, `debris`=3 (0 is background).
- Skips `unlabeled`; `--verified-only` additionally drops `auto` boxes.
- Each annotation keeps `cell` (the group id) and `auto`, so correlated boxes
  can be grouped in later analysis or a stricter split.
- The `manifest.json` records source files, per-class counts, the split, the
  schema version, and the git commit — so a trained model can be traced back.

## Split by SOURCE FILE, not by frame

This matters more than anything else here. Because you annotate the first N
frames of the same movie, boxes on consecutive frames are **near-duplicates**
of each other. A random per-image split puts frame 3 in train and frame 4 in
val, so the model is validated on images it has effectively seen — validation
scores come out inflated and useless for model selection.

Split by **ND2 / position** (ideally by well or experiment), so no movie
appears in both train and val.

## A note on annotation effort

For the same reason, 20 frames of one position is worth far less than 20
positions × 1–2 frames: the extra frames are highly correlated, so the
*effective* dataset is much smaller than the box count suggests. Prefer
**breadth** — more ND2s, wells and conditions — over depth in one movie.

The "Annotate first N frames" spinner is the knob: set it low (e.g. 3–5) and
cover many files, rather than 20 frames of a few. If frame-to-frame
redundancy still looks high, an "annotate every k-th frame" option is a small
addition to the dashboard — say the word.
