"""Export ``*.simple.json`` annotations + ND2s into a frozen training set.

Why export instead of reading the JSON/ND2 during training: ND2 random
access is slow and not safe across DataLoader workers, the ND2s live on a
network share while training runs elsewhere (local GPU, then the A100 in
Docker), and a frozen dataset stays reproducible while annotators keep
working. See ``docs/training-export.md``.

Output layout::

    dataset/
      images/<stem>_t0000.tif     full frame, 16-bit, brightfield channel
      annotations/train.json      COCO
      annotations/val.json        COCO
      manifest.json               provenance + counts + the split

Two correctness rules are enforced here:

- **Only frames that have at least one classified box are exported.** A
  frame with no boxes is *unlabelled*, not empty — exporting it would teach
  the model that real cells are background.
- **The train/val split is by SOURCE FILE, never by frame.** Consecutive
  frames of one movie are near-duplicates, so a per-frame split leaks between
  train and val and inflates validation scores.

Run via ``nikon-control-export`` (see ``--help``).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .schema_simple import (
    SIMPLE_SUFFIX,
    TRAINING_CLASSES,
    SimpleAnnotationFile,
    load_simple,
)

# COCO category ids: 0 is background, so classes start at 1.
CATEGORY_IDS: dict[str, int] = {c: i + 1 for i, c in enumerate(TRAINING_CLASSES)}


def coco_categories() -> list[dict]:
    return [{"id": i, "name": name, "supercategory": "cell"}
            for name, i in CATEGORY_IDS.items()]


def split_by_source(stems: list[str], val_frac: float,
                    seed: int = 0) -> tuple[set[str], set[str]]:
    """Deterministically split SOURCE FILES (not frames) into train/val.

    Splitting by file is the whole point: frames from one movie are highly
    correlated, so they must never straddle the split. With very few files,
    at least one is still assigned to val (unless there is only one file, in
    which case val is empty and the caller warns).
    """
    import random

    ordered = sorted(stems)
    if len(ordered) <= 1:
        return set(ordered), set()
    rng = random.Random(seed)
    shuffled = list(ordered)
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_frac))
    n_val = min(n_val, len(shuffled) - 1)  # always leave something to train on
    return set(shuffled[n_val:]), set(shuffled[:n_val])


def frames_to_export(af: SimpleAnnotationFile, frame_step: int = 1,
                     verified_only: bool = False) -> dict[int, list]:
    """Frames that carry at least one usable label -> their boxes.

    ``frame_step`` subsamples frames (e.g. 5 = every 5th) to cut the
    near-duplicate redundancy of consecutive frames.
    """
    out: dict[int, list] = {}
    for b in af.training_boxes(verified_only=verified_only):
        if frame_step > 1 and b.t % frame_step:
            continue
        out.setdefault(b.t, []).append(b)
    return dict(sorted(out.items()))


def build_coco(records: list[dict]) -> dict:
    """Assemble a COCO dict from per-image records (pure).

    Each record: ``{"file_name", "height", "width", "source", "t", "boxes"}``
    where each box has ``bbox`` ``[y0,x0,y1,x1]``, ``label``, ``group``,
    ``score``, ``auto``.
    """
    images, annotations = [], []
    ann_id = 1
    for img_id, rec in enumerate(records, start=1):
        images.append({
            "id": img_id,
            "file_name": rec["file_name"],
            "height": rec["height"],
            "width": rec["width"],
            # provenance, so any label can be traced back to the microscope
            "source": rec["source"],
            "frame": rec["t"],
        })
        for b in rec["boxes"]:
            y0, x0, y1, x1 = (float(v) for v in b.bbox)
            w, h = x1 - x0, y1 - y0
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": CATEGORY_IDS[b.label],
                "bbox": [x0, y0, w, h],          # COCO is [x, y, w, h]
                "area": w * h,
                "iscrowd": 0,
                # which physical cell this came from — lets a future split or
                # analysis group correlated boxes
                "cell": b.group,
                "auto": b.auto,
                "score": b.score,
            })
            ann_id += 1
    return {
        "info": {"description": "nikon-control single/doublet/debris detection"},
        "images": images,
        "annotations": annotations,
        "categories": coco_categories(),
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _find_nd2(json_path: Path, af: SimpleAnnotationFile) -> Path | None:
    """Locate the ND2 for a sidecar.

    Prefers the sibling file (the annotation was written next to it), because
    ``af.source`` may be an absolute path from a different machine (e.g. a
    Windows ``G:\\`` share recorded on the server).
    """
    sibling = json_path.parent / (json_path.name[:-len(SIMPLE_SUFFIX)] + ".nd2")
    if sibling.exists():
        return sibling
    src = Path(af.source)
    return src if src.exists() else None


def export(data_dir: Path, out_dir: Path, *, val_frac: float = 0.2,
           seed: int = 0, frame_step: int = 1, verified_only: bool = False,
           dry_run: bool = False) -> dict:
    """Walk ``data_dir`` for sidecars and write a COCO dataset to ``out_dir``."""
    import numpy as np

    from .io import open_nd2

    sidecars = sorted(Path(data_dir).rglob("*" + SIMPLE_SUFFIX))
    if not sidecars:
        raise SystemExit(f"no *{SIMPLE_SUFFIX} files found under {data_dir}")

    img_dir = out_dir / "images"
    if not dry_run:
        img_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "annotations").mkdir(parents=True, exist_ok=True)

    per_source: dict[str, list[dict]] = {}
    skipped: list[str] = []
    box_counts: dict[str, int] = {c: 0 for c in TRAINING_CLASSES}
    cell_ids: dict[str, set] = {c: set() for c in TRAINING_CLASSES}

    for sc in sidecars:
        af = load_simple(sc)
        stem = sc.name[:-len(SIMPLE_SUFFIX)]
        frames = frames_to_export(af, frame_step, verified_only)
        if not frames:
            skipped.append(f"{sc.name}: no classified boxes")
            continue
        nd2_path = _find_nd2(sc, af)
        if nd2_path is None:
            skipped.append(f"{sc.name}: ND2 not found "
                           f"(looked for {stem}.nd2 and {af.source})")
            continue

        nd = None if dry_run else open_nd2(nd2_path)
        try:
            for t, boxes in frames.items():
                fname = f"{stem}_t{t:04d}.tif"
                if dry_run:
                    H, W = 0, 0
                else:
                    plane = np.asarray(nd["plane"](t, af.bf_channel))
                    H, W = int(plane.shape[-2]), int(plane.shape[-1])
                    import tifffile
                    tifffile.imwrite(img_dir / fname, plane)
                per_source.setdefault(stem, []).append({
                    "file_name": fname, "height": H, "width": W,
                    "source": str(nd2_path), "t": t, "boxes": boxes,
                })
                for b in boxes:
                    box_counts[b.label] += 1
                    cell_ids[b.label].add((stem, b.group or f"solo{id(b)}"))
        finally:
            if nd is not None:
                try:
                    nd["file"].close()
                except Exception:
                    pass

    if not per_source:
        raise SystemExit("nothing to export:\n  " + "\n  ".join(skipped))

    train_stems, val_stems = split_by_source(list(per_source), val_frac, seed)
    splits = {
        "train": [r for s in sorted(train_stems) for r in per_source[s]],
        "val": [r for s in sorted(val_stems) for r in per_source[s]],
    }

    manifest = {
        "created_from": str(data_dir),
        "git_commit": _git_commit(),
        "frame_step": frame_step,
        "verified_only": verified_only,
        "val_frac": val_frac,
        "seed": seed,
        "categories": coco_categories(),
        "split_by": "source file (never by frame — frames are correlated)",
        "split": {"train": sorted(train_stems), "val": sorted(val_stems)},
        "images": {k: len(v) for k, v in splits.items()},
        "boxes_per_class": box_counts,
        "cells_per_class": {c: len(v) for c, v in cell_ids.items()},
        "skipped": skipped,
    }

    if not dry_run:
        for name, recs in splits.items():
            (out_dir / "annotations" / f"{name}.json").write_text(
                json.dumps(build_coco(recs), indent=1)
            )
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(
        prog="nikon-control-export",
        description="Export *.simple.json annotations + ND2s to a COCO dataset.",
    )
    p.add_argument("--data-dir", required=True,
                   help="folder to search (recursively) for *.simple.json")
    p.add_argument("--out", required=True, help="dataset output folder")
    p.add_argument("--val-frac", type=float, default=0.2,
                   help="fraction of SOURCE FILES held out for validation")
    p.add_argument("--seed", type=int, default=0, help="split seed")
    p.add_argument("--frame-step", type=int, default=1,
                   help="keep every k-th frame (cuts near-duplicate frames)")
    p.add_argument("--verified-only", action="store_true",
                   help="drop auto-labelled boxes no human touched")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would be exported, write nothing")
    args = p.parse_args()

    m = export(Path(args.data_dir), Path(args.out), val_frac=args.val_frac,
               seed=args.seed, frame_step=args.frame_step,
               verified_only=args.verified_only, dry_run=args.dry_run)

    print(json.dumps(m, indent=2))
    print()
    print(f"images: {m['images']['train']} train / {m['images']['val']} val")
    print(f"boxes : {m['boxes_per_class']}")
    print(f"cells : {m['cells_per_class']}")
    if not m["split"]["val"]:
        print("\n⚠ only one source file — val is EMPTY. Annotate more ND2s "
              "before trusting any validation number.")
    n_cells = sum(m["cells_per_class"].values())
    if n_cells < 100:
        print(f"\n⚠ only {n_cells} distinct cells. Frames of the same cell are "
              "near-duplicates, so this is much less data than the box count "
              "suggests — prefer annotating MORE FILES over more frames.")
    if m["skipped"]:
        print("\nskipped:")
        for s in m["skipped"]:
            print("  -", s)


if __name__ == "__main__":
    main()
