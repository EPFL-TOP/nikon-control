"""Controller for the annotation REVIEW dashboard.

Operates directly on a ``nikon-control-export`` output (a COCO dataset), so
what gets reviewed is exactly what training will consume — no re-derivation,
no chance of reviewing something different from the real input.

As with the other dashboards, all correctness logic lives here (pure, no
GUI) and the view only wires widgets.

Editing model: boxes are addressed by ``"<split>:<coco annotation id>"``, and
edits mutate the loaded COCO payloads in place; ``save`` writes each split
back, keeping a one-off ``.bak`` of the original. Per-image ``reviewed``
flags are stored on the COCO image entries, so review progress survives
reopening the dataset.

Caveat the view surfaces prominently: re-running ``nikon-control-export``
REGENERATES the dataset from the ``*.simple.json`` sidecars and therefore
discards edits made here. Review is a final QC pass before training; lasting
fixes belong in the annotation dashboard.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..agreement import DEFAULT_MATCH_IOU, Box, compare, confusion, totals
from ..overlap import (
    DEFAULT_MAX_OVERLAP,
    PREFER_AREA,
    OverlapCandidate,
    count_overlaps,
    select_survivors,
)

SPLITS = ("train", "val")

# filters offered by the view
FILTER_ALL = "all images"
FILTER_UNVERIFIED = "with unverified predictions"
FILTER_UNREVIEWED = "not yet reviewed"
FILTER_REVIEWED = "reviewed"
FILTER_EMPTY = "with no boxes"
FILTER_OVERLAP = "with overlapping boxes"
FILTER_DISAGREE = "where the model disagrees"

# model predictions are cached next to the dataset so a review session
# doesn't have to re-run the model after every reload
PREDICTIONS_FILE = "predictions.json"


def load_splits(dataset_dir: str | Path) -> dict[str, dict]:
    """Read ``annotations/<split>.json`` for whichever splits exist."""
    d = Path(dataset_dir)
    ann = d / "annotations"
    out: dict[str, dict] = {}
    for name in SPLITS:
        p = ann / f"{name}.json"
        if p.exists():
            out[name] = json.loads(p.read_text())
    if not out:
        raise FileNotFoundError(
            f"no annotations/train.json or val.json under {d} — is this a "
            "nikon-control-export dataset?"
        )
    return out


def bbox_to_cwh(bbox: list[float]) -> tuple[float, float, float, float]:
    """COCO [x, y, w, h] -> (cx, cy, w, h) for a Bokeh Rect glyph."""
    x, y, w, h = (float(v) for v in bbox)
    return x + w / 2.0, y + h / 2.0, w, h


def cwh_to_bbox(cx: float, cy: float, w: float, h: float) -> list[float]:
    """(cx, cy, w, h) -> COCO [x, y, w, h]."""
    w, h = max(1.0, float(w)), max(1.0, float(h))
    return [float(cx) - w / 2.0, float(cy) - h / 2.0, w, h]


class ReviewState:
    def __init__(self, splits: dict[str, dict],
                 dataset_dir: str | Path | None = None):
        self.splits = splits
        self.dataset_dir = Path(dataset_dir) if dataset_dir else None
        self.dirty = False

        # categories (assumed consistent across splits — the exporter writes
        # the same list into both)
        cats: dict[int, str] = {}
        for payload in splits.values():
            for c in payload.get("categories", []):
                cats[int(c["id"])] = str(c["name"])
        self.cat_name = cats
        self.cat_id = {v: k for k, v in cats.items()}

        # flat, stable image order: split, then the file's own order
        self.images: list[tuple[str, dict]] = []
        self._anns: dict[str, dict] = {}
        self._by_image: dict[tuple[str, int], list[str]] = {}
        self._next_id = 1
        for split in SPLITS:
            payload = splits.get(split)
            if payload is None:
                continue
            for img in payload.get("images", []):
                self.images.append((split, img))
                self._by_image.setdefault((split, int(img["id"])), [])
            for a in payload.get("annotations", []):
                key = f"{split}:{a['id']}"
                self._anns[key] = a
                self._by_image.setdefault(
                    (split, int(a["image_id"])), []).append(key)
                self._next_id = max(self._next_id, int(a["id"]) + 1)

        self.current = 0
        self.filter = FILTER_ALL
        # file_name -> list of {"bbox":[y0,x0,y1,x1], "label":str, "score":f}
        self.predictions: dict[str, list[dict]] = {}
        self.prediction_meta: dict = {}
        self.match_iou = DEFAULT_MATCH_IOU
        self._agree_cache: dict[str, object] = {}

    # ---- classes -------------------------------------------------------
    @property
    def classes(self) -> list[str]:
        return [self.cat_name[i] for i in sorted(self.cat_name)]

    # ---- navigation / filtering ----------------------------------------
    def _has_unverified(self, idx: int) -> bool:
        return any(self._anns[k].get("auto") for k in self._keys(idx))

    def _matches(self, idx: int) -> bool:
        f = self.filter
        if f == FILTER_ALL:
            return True
        if f == FILTER_UNVERIFIED:
            return self._has_unverified(idx)
        if f == FILTER_UNREVIEWED:
            return not self.is_reviewed(idx)
        if f == FILTER_REVIEWED:
            return self.is_reviewed(idx)
        if f == FILTER_EMPTY:
            return not self._keys(idx)
        if f == FILTER_OVERLAP:
            return self.has_overlaps(idx)
        if f == FILTER_DISAGREE:
            ag = self.agreement(idx)
            return ag is not None and not ag.agrees
        # otherwise: a class name
        return any(self.cat_name.get(int(self._anns[k]["category_id"])) == f
                   for k in self._keys(idx))

    def visible(self) -> list[int]:
        """Indices matching the active filter (always non-empty in practice:
        an empty result leaves navigation where it is)."""
        return [i for i in range(len(self.images)) if self._matches(i)]

    def set_filter(self, name: str) -> int:
        """Apply a filter and jump to its first match. Returns the match
        count; the current image is kept if it still matches."""
        self.filter = name
        vis = self.visible()
        if vis and self.current not in vis:
            self.current = vis[0]
        return len(vis)

    def goto(self, idx: int) -> None:
        self.current = max(0, min(int(idx), len(self.images) - 1))

    def step(self, delta: int) -> None:
        """Move to the next/previous image that matches the filter."""
        vis = self.visible()
        if not vis:
            return
        if self.current in vis:
            pos = vis.index(self.current)
            self.current = vis[max(0, min(pos + delta, len(vis) - 1))]
        else:
            self.current = vis[0]

    # ---- image / box access --------------------------------------------
    def image(self, idx: int | None = None) -> dict:
        _, img = self.images[self.current if idx is None else idx]
        return img

    def split_of(self, idx: int | None = None) -> str:
        split, _ = self.images[self.current if idx is None else idx]
        return split

    def _keys(self, idx: int | None = None) -> list[str]:
        i = self.current if idx is None else idx
        split, img = self.images[i]
        return list(self._by_image.get((split, int(img["id"])), []))

    def boxes(self, idx: int | None = None) -> list[dict]:
        """CDS rows for one image — same contract as the other dashboards."""
        rows = []
        for n, key in enumerate(self._keys(idx), start=1):
            a = self._anns[key]
            cx, cy, w, h = bbox_to_cwh(a["bbox"])
            score = a.get("score")
            marks = []
            if score is not None:
                marks.append(f"{float(score):.2f}")
            if a.get("auto"):
                marks.append("?")  # unverified model prediction
            rows.append({
                "id": key,
                "num": n,
                "label": self.cat_name.get(int(a["category_id"]), "?"),
                "cx": cx, "cy": cy, "w": w, "h": h,
                "marker": " ".join(marks),
            })
        return rows

    def has(self, box_id: str) -> bool:
        return box_id in self._anns

    # ---- edits ----------------------------------------------------------
    def set_class(self, box_id: str, label: str) -> None:
        """Re-classify a box. Reviewing a prediction also verifies it."""
        if label not in self.cat_id:
            raise ValueError(f"unknown class {label!r}; have {self.classes}")
        a = self._anns[box_id]
        a["category_id"] = self.cat_id[label]
        a["auto"] = False  # a human just decided this
        self.dirty = True
        self._agree_cache.clear()  # the comparison is now stale

    def delete(self, box_id: str) -> None:
        a = self._anns.pop(box_id, None)
        if a is None:
            return
        split = box_id.split(":", 1)[0]
        key = (split, int(a["image_id"]))
        if box_id in self._by_image.get(key, []):
            self._by_image[key].remove(box_id)
        anns = self.splits[split]["annotations"]
        for i, other in enumerate(anns):
            if other is a:
                anns.pop(i)
                break
        self.dirty = True
        self._agree_cache.clear()

    def add_box(self, cx: float, cy: float, w: float, h: float, label: str,
                idx: int | None = None) -> str:
        """Add a box a human drew (so ``auto`` is False, ``score`` None)."""
        i = self.current if idx is None else idx
        split, img = self.images[i]
        ann = {
            "id": self._next_id,
            "image_id": int(img["id"]),
            "category_id": self.cat_id[label],
            "bbox": cwh_to_bbox(cx, cy, w, h),
            "area": max(1.0, float(w)) * max(1.0, float(h)),
            "iscrowd": 0,
            "cell": None,
            "auto": False,
            "score": None,
        }
        self._next_id += 1
        key = f"{split}:{ann['id']}"
        self._anns[key] = ann
        self._by_image.setdefault((split, int(img["id"])), []).append(key)
        self.splits[split]["annotations"].append(ann)
        self.dirty = True
        self._agree_cache.clear()
        return key

    def _set_geom(self, box_id: str, cx: float, cy: float,
                  w: float, h: float) -> None:
        a = self._anns[box_id]
        a["bbox"] = cwh_to_bbox(cx, cy, w, h)
        a["area"] = a["bbox"][2] * a["bbox"][3]
        a["auto"] = False
        self.dirty = True
        self._agree_cache.clear()

    def size_of(self, box_id: str) -> tuple[float, float]:
        _, _, w, h = bbox_to_cwh(self._anns[box_id]["bbox"])
        return w, h

    def center_of(self, box_id: str) -> tuple[float, float]:
        cx, cy, _, _ = bbox_to_cwh(self._anns[box_id]["bbox"])
        return cx, cy

    def move_to(self, box_id: str, cx: float, cy: float) -> None:
        w, h = self.size_of(box_id)
        self._set_geom(box_id, cx, cy, w, h)

    def nudge(self, box_id: str, dx: float, dy: float) -> None:
        cx, cy = self.center_of(box_id)
        self.move_to(box_id, cx + dx, cy + dy)

    def resize(self, box_id: str, w: float, h: float) -> None:
        cx, cy = self.center_of(box_id)
        self._set_geom(box_id, cx, cy, w, h)

    def scale(self, box_id: str, factor: float) -> None:
        cx, cy, w, h = bbox_to_cwh(self._anns[box_id]["bbox"])
        self._set_geom(box_id, cx, cy, w * factor, h * factor)

    # ---- overlap suppression ---------------------------------------------
    def _candidates(self, idx: int | None = None) -> list[OverlapCandidate]:
        out = []
        for key in self._keys(idx):
            a = self._anns[key]
            x, y, w, h = (float(v) for v in a["bbox"])
            out.append(OverlapCandidate(
                ref=key,
                bbox=[y, x, y + h, x + w],   # overlap helpers use y0,x0,y1,x1
                score=a.get("score"),
                protected=not a.get("auto"),
            ))
        return out

    def suppress_overlaps(self, max_overlap: float = DEFAULT_MAX_OVERLAP,
                          prefer: str = PREFER_AREA,
                          all_images: bool = False) -> int:
        """Drop duplicate overlapping boxes on this image (or every image).

        Human-made boxes are protected, exactly as in the annotation
        dashboard. Returns how many were removed.
        """
        targets = (range(len(self.images)) if all_images
                   else [self.current])
        removed = 0
        for i in targets:
            cands = self._candidates(i)
            if len(cands) < 2:
                continue
            _, dropped = select_survivors(cands, max_overlap, prefer)
            for key in dropped:
                self.delete(key)
                removed += 1
        return removed

    def overlap_count(self, max_overlap: float = DEFAULT_MAX_OVERLAP,
                      idx: int | None = None) -> int:
        return count_overlaps(self._candidates(idx), max_overlap)

    def has_overlaps(self, idx: int | None = None,
                     max_overlap: float = DEFAULT_MAX_OVERLAP) -> bool:
        return self.overlap_count(max_overlap, idx) > 0

    # ---- model predictions & disagreement --------------------------------
    def set_predictions(self, per_image: dict[str, list[dict]],
                        meta: dict | None = None) -> None:
        """Attach a model's predictions, keyed by image ``file_name``."""
        self.predictions = per_image
        self.prediction_meta = meta or {}
        self._agree_cache.clear()

    @property
    def has_predictions(self) -> bool:
        return bool(self.predictions)

    def predicted(self, idx: int | None = None) -> list[dict]:
        return list(self.predictions.get(str(self.image(idx)["file_name"]), []))

    def agreement(self, idx: int | None = None):
        """Compare annotations and predictions for one image (cached).

        Returns None when no predictions have been run.
        """
        if not self.predictions:
            return None
        i = self.current if idx is None else idx
        fname = str(self.image(i)["file_name"])
        if fname in self._agree_cache:
            return self._agree_cache[fname]
        gt = [
            Box(bbox=self._yxyx(self._anns[k]["bbox"]),
                label=self.cat_name.get(int(self._anns[k]["category_id"]), "?"))
            for k in self._keys(i)
        ]
        pred = [Box(bbox=list(p["bbox"]), label=str(p["label"]),
                    score=p.get("score")) for p in self.predictions.get(fname, [])]
        ag = compare(gt, pred, self.match_iou)
        self._agree_cache[fname] = ag
        return ag

    @staticmethod
    def _yxyx(coco_bbox) -> list[float]:
        x, y, w, h = (float(v) for v in coco_bbox)
        return [y, x, y + h, x + w]

    def matched_prediction_for(self, box_id: str) -> dict | None:
        """The prediction matched to one annotation, if any — so the view can
        offer 'use the model's class' for a mismatch."""
        ag = self.agreement()
        if ag is None:
            return None
        keys = self._keys()
        if box_id not in keys:
            return None
        gi = keys.index(box_id)
        for g, pi in ag.matched:
            if g == gi:
                return self.predicted()[pi]
        return None

    def disagreeing_indices(self) -> list[int]:
        """Image indices with any disagreement, worst first."""
        if not self.predictions:
            return []
        scored = []
        for i in range(len(self.images)):
            ag = self.agreement(i)
            if ag is not None and not ag.agrees:
                scored.append((ag.score, i))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [i for _, i in scored]

    def agreement_totals(self) -> dict[str, int]:
        if not self.predictions:
            return {}
        ags = [(str(img["file_name"]), self.agreement(i))
               for i, (_, img) in enumerate(self.images)]
        return totals([(n, a) for n, a in ags if a is not None])

    def agreement_confusion(self) -> dict[tuple[str, str], int]:
        if not self.predictions:
            return {}
        ags = [(str(img["file_name"]), self.agreement(i))
               for i, (_, img) in enumerate(self.images)]
        return confusion([(n, a) for n, a in ags if a is not None])

    def invalidate_agreement(self, idx: int | None = None) -> None:
        """Drop the cached comparison for an image after an edit."""
        i = self.current if idx is None else idx
        self._agree_cache.pop(str(self.image(i)["file_name"]), None)

    def save_predictions(self) -> Path | None:
        if self.dataset_dir is None or not self.predictions:
            return None
        path = self.dataset_dir / PREDICTIONS_FILE
        path.write_text(json.dumps({
            "meta": self.prediction_meta,
            "per_image": self.predictions,
        }, indent=1))
        return path

    def load_predictions(self) -> bool:
        """Load cached predictions if present. Returns whether any loaded."""
        if self.dataset_dir is None:
            return False
        path = self.dataset_dir / PREDICTIONS_FILE
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text())
        except Exception:
            return False
        self.set_predictions(payload.get("per_image", {}),
                             payload.get("meta", {}))
        return self.has_predictions

    # ---- review progress -------------------------------------------------
    def is_reviewed(self, idx: int | None = None) -> bool:
        return bool(self.image(idx).get("reviewed"))

    def mark_reviewed(self, reviewed: bool = True,
                      idx: int | None = None) -> None:
        self.image(idx)["reviewed"] = bool(reviewed)
        self.dirty = True

    def review_progress(self) -> tuple[int, int]:
        done = sum(1 for i in range(len(self.images)) if self.is_reviewed(i))
        return done, len(self.images)

    # ---- summaries -------------------------------------------------------
    def counts(self) -> dict[str, int]:
        c = {name: 0 for name in self.classes}
        for a in self._anns.values():
            name = self.cat_name.get(int(a["category_id"]))
            if name:
                c[name] = c.get(name, 0) + 1
        return c

    def unverified_count(self) -> int:
        return sum(1 for a in self._anns.values() if a.get("auto"))

    def split_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for split, _ in self.images:
            out[split] = out.get(split, 0) + 1
        return out

    # ---- persistence -----------------------------------------------------
    def save(self, backup: bool = True) -> list[Path]:
        """Write each split back. Returns the files written."""
        if self.dataset_dir is None:
            raise RuntimeError("this ReviewState has no dataset_dir to save to")
        written = []
        for split, payload in self.splits.items():
            path = self.dataset_dir / "annotations" / f"{split}.json"
            if backup:
                bak = path.with_suffix(".json.bak")
                if path.exists() and not bak.exists():
                    shutil.copy2(path, bak)  # one-off snapshot of the original
            path.write_text(json.dumps(payload, indent=1))
            written.append(path)
        self.dirty = False
        return written
