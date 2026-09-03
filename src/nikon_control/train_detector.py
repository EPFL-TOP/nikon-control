"""Train the single/doublet/debris detector on an exported COCO dataset.

Consumes what ``export_training.py`` produces and writes a checkpoint in the
same shape the rest of the package reads (``model_state_dict`` +
``classes``), so ``CellDetector`` can load it — it already infers the class
count from the box predictor.

Design choices worth knowing:

- **Preprocessing is imported from ``detector.normalize_plane``**, not
  reimplemented, so training sees exactly what inference will see
  (percentile normalisation of the 16-bit brightfield plane, replicated to
  3 channels). A mismatch here is the classic silent accuracy killer.
- **Warm start** (``--init-from``) loads the existing 1-class
  ``cell_detection_model.pth`` into the backbone/RPN and drops only the
  final box predictor, which has the wrong shape for 3 classes. That reuses
  everything the old model learned about finding cells.
- **AP@0.5 per class** is computed with a small built-in implementation, so
  there is no pycocotools dependency.

Run via ``nikon-control-train`` (see ``--help``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .detector import normalize_plane
from .export_training import CATEGORY_IDS

# id -> name, plus background at 0
CLASS_NAMES: list[str] = ["__background__"] + [
    name for name, _ in sorted(CATEGORY_IDS.items(), key=lambda kv: kv[1])
]


class CocoDetectionDataset:
    """Minimal COCO detection dataset over the exported 16-bit TIFFs."""

    def __init__(self, images_dir: Path, ann_file: Path, augment: bool = False):
        payload = json.loads(Path(ann_file).read_text())
        self.images_dir = Path(images_dir)
        self.images = payload["images"]
        self.augment = augment
        self.by_image: dict[int, list[dict]] = {}
        for a in payload["annotations"]:
            self.by_image.setdefault(a["image_id"], []).append(a)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        import tifffile
        import torch

        info = self.images[idx]
        plane = tifffile.imread(self.images_dir / info["file_name"])
        # identical preprocessing to inference
        img = normalize_plane(np.asarray(plane))
        anns = self.by_image.get(info["id"], [])
        boxes, labels = [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 1 or h <= 1:
                continue
            boxes.append([x, y, x + w, y + h])  # torchvision wants xyxy
            labels.append(a["category_id"])

        if self.augment and boxes and np.random.rand() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])  # horizontal flip
            W = img.shape[-1]
            boxes = [[W - x1, y0, W - x0, y1] for x0, y0, x1, y1 in boxes]

        tensor = torch.from_numpy(img)[None].repeat(3, 1, 1)
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([info["id"]]),
        }
        return tensor, target


def collate(batch):
    return tuple(zip(*batch))


def build_model(num_classes: int, init_from: str | None = None,
                pretrained_backbone: bool = True):
    """Faster R-CNN with ``num_classes`` (including background)."""
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    if init_from:
        # warm start from the existing cell model: same architecture, but its
        # predictor is 1-class so it must be rebuilt
        from .detector import load_checkpoint_state_dict
        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
        state = load_checkpoint_state_dict(torch.load(init_from,
                                                      map_location="cpu"))
        state = {k: v for k, v in state.items()
                 if not k.startswith("roi_heads.box_predictor.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        kept = len(state)
        print(f"warm start: loaded {kept} tensors from {init_from} "
              f"(box predictor reinitialised for {num_classes} classes)")
        if unexpected:
            print(f"  ignored {len(unexpected)} unexpected tensors")
    else:
        weights = "DEFAULT" if pretrained_backbone else None
        model = fasterrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def average_precision(preds: list[dict], gts: list[dict],
                      iou_thr: float = 0.5) -> dict[str, float]:
    """AP@iou per class + macro mAP (small, dependency-free implementation).

    ``preds``/``gts`` are per-image dicts of numpy arrays: boxes (xyxy),
    labels, and for preds scores.
    """
    out: dict[str, float] = {}
    aps = []
    for cid in sorted(CATEGORY_IDS.values()):
        name = CLASS_NAMES[cid]
        # gather this class across images
        scored = []          # (score, image_idx, box)
        n_gt = 0
        gt_by_img = {}
        for i, g in enumerate(gts):
            m = g["labels"] == cid
            gt_by_img[i] = g["boxes"][m]
            n_gt += int(m.sum())
        for i, p in enumerate(preds):
            m = p["labels"] == cid
            for box, sc in zip(p["boxes"][m], p["scores"][m]):
                scored.append((float(sc), i, box))
        if n_gt == 0:
            out[name] = float("nan")
            continue
        scored.sort(key=lambda x: -x[0])
        matched = {i: np.zeros(len(b), dtype=bool) for i, b in gt_by_img.items()}
        tp = np.zeros(len(scored))
        fp = np.zeros(len(scored))
        for k, (_, img_i, box) in enumerate(scored):
            gt = gt_by_img[img_i]
            if len(gt) == 0:
                fp[k] = 1
                continue
            x0 = np.maximum(box[0], gt[:, 0]); y0 = np.maximum(box[1], gt[:, 1])
            x1 = np.minimum(box[2], gt[:, 2]); y1 = np.minimum(box[3], gt[:, 3])
            inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
            area_b = (box[2] - box[0]) * (box[3] - box[1])
            area_g = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
            iou = inter / (area_b + area_g - inter + 1e-9)
            j = int(np.argmax(iou))
            if iou[j] >= iou_thr and not matched[img_i][j]:
                tp[k] = 1
                matched[img_i][j] = True
            else:
                fp[k] = 1
        ctp, cfp = np.cumsum(tp), np.cumsum(fp)
        recall = ctp / n_gt
        precision = ctp / np.maximum(ctp + cfp, 1e-9)
        # 101-point interpolated AP
        ap = 0.0
        for r in np.linspace(0, 1, 101):
            p_at_r = precision[recall >= r]
            ap += (p_at_r.max() if len(p_at_r) else 0.0) / 101
        out[name] = float(ap)
        aps.append(ap)
    out["mAP@0.5"] = float(np.mean(aps)) if aps else float("nan")
    return out


def evaluate(model, loader, device) -> dict[str, float]:
    """Run the model over the val loader and return AP@0.5 per class."""
    import torch

    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = [i.to(device) for i in imgs]
            outputs = model(imgs)
            for o, t in zip(outputs, targets):
                preds.append({
                    "boxes": o["boxes"].cpu().numpy(),
                    "labels": o["labels"].cpu().numpy(),
                    "scores": o["scores"].cpu().numpy(),
                })
                gts.append({
                    "boxes": t["boxes"].numpy(),
                    "labels": t["labels"].numpy(),
                })
    return average_precision(preds, gts)


def train(dataset: Path, out: Path, *, epochs: int = 20, batch_size: int = 2,
          lr: float = 5e-3, workers: int = 0, init_from: str | None = None,
          device: str | None = None) -> None:
    import torch
    from torch.utils.data import DataLoader

    from .detector import _resolve_device

    dev = _resolve_device(torch, device)
    print(f"device: {dev}")

    ann = dataset / "annotations"
    imgs = dataset / "images"
    train_ds = CocoDetectionDataset(imgs, ann / "train.json", augment=True)
    val_json = ann / "val.json"
    val_ds = CocoDetectionDataset(imgs, val_json) if val_json.exists() else None
    print(f"train images: {len(train_ds)}"
          + (f" | val images: {len(val_ds)}" if val_ds else " | NO val set"))
    if val_ds is not None and len(val_ds) == 0:
        print("⚠ val set is empty — annotate more ND2 FILES (the split is by "
              "file, deliberately). Validation numbers will be meaningless.")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=workers, collate_fn=collate)
    val_dl = (DataLoader(val_ds, batch_size=1, shuffle=False,
                         num_workers=workers, collate_fn=collate)
              if val_ds and len(val_ds) else None)

    num_classes = len(CLASS_NAMES)  # background + 3
    model = build_model(num_classes, init_from=init_from).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best = -1.0
    out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for imgs_b, targets in train_dl:
            imgs_b = [i.to(dev) for i in imgs_b]
            targets = [{k: v.to(dev) for k, v in t.items()} for t in targets]
            losses = model(imgs_b, targets)
            loss = sum(losses.values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss)
        sched.step()
        msg = f"epoch {epoch}/{epochs}  train_loss {total / max(1, len(train_dl)):.4f}"

        score = None
        if val_dl is not None:
            metrics = evaluate(model, val_dl, dev)
            score = metrics["mAP@0.5"]
            msg += "  " + "  ".join(
                f"{k} {v:.3f}" for k, v in metrics.items()
                if not (isinstance(v, float) and v != v)  # skip NaN
            )
        print(msg, flush=True)

        # checkpoint: best by val mAP, or last epoch when there is no val set
        is_best = score is not None and score > best
        if is_best:
            best = score
        if is_best or (val_dl is None and epoch == epochs):
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": CLASS_NAMES,
                "epoch": epoch,
                "val_mAP@0.5": score,
                "init_from": init_from,
            }, out)
            print(f"  saved {out}" + (f" (mAP {score:.3f})" if score else ""))
    print(f"done. best val mAP@0.5: {best:.3f}" if best >= 0 else "done.")


def main() -> None:
    p = argparse.ArgumentParser(
        prog="nikon-control-train",
        description="Train the single/doublet/debris detector on an exported "
                    "COCO dataset (see nikon-control-export).",
    )
    p.add_argument("--dataset", required=True, help="dataset folder from export")
    p.add_argument("--out", default="cell_classes_model.pth",
                   help="checkpoint path to write")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--workers", type=int, default=0,
                   help="DataLoader workers (0 is safest on Windows)")
    p.add_argument("--init-from", default=None,
                   help="warm start from an existing .pth (e.g. the current "
                        "1-class cell_detection_model.pth)")
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    args = p.parse_args()
    train(Path(args.dataset), Path(args.out), epochs=args.epochs,
          batch_size=args.batch_size, lr=args.lr, workers=args.workers,
          init_from=args.init_from, device=args.device)


if __name__ == "__main__":
    main()
