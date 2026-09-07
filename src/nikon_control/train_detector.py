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
import math
import time
from pathlib import Path

import numpy as np

from .detector import normalize_plane
from .export_training import CATEGORY_IDS

# id -> name, plus background at 0
CLASS_NAMES: list[str] = ["__background__"] + [
    name for name, _ in sorted(CATEGORY_IDS.items(), key=lambda kv: kv[1])
]


def dihedral(img: np.ndarray, boxes: list[list[float]], transpose: bool,
             hflip: bool, vflip: bool) -> tuple[np.ndarray, list[list[float]]]:
    """One of the 8 dihedral transforms of a plane and its xyxy boxes.

    Microscopy frames have no canonical orientation, so the full symmetry
    group is valid label-preserving augmentation — it multiplies 343 images
    into 8x as many distinct views for free, which is the cheapest lever
    against overfitting when the dataset is fixed.

    Box coordinates are continuous edges in ``[0, W]``, so a flip is
    ``x -> W - x`` (not ``W - 1 - x``) and the ``x0/x1`` pair swaps.
    """
    out = np.asarray(img)
    bx = [list(map(float, b)) for b in boxes]
    if transpose:                      # (y, x) -> (x, y)
        out = out.T
        bx = [[y0, x0, y1, x1] for x0, y0, x1, y1 in bx]
    H, W = out.shape[-2], out.shape[-1]
    if hflip:
        out = out[:, ::-1]
        bx = [[W - x1, y0, W - x0, y1] for x0, y0, x1, y1 in bx]
    if vflip:
        out = out[::-1, :]
        bx = [[x0, H - y1, x1, H - y0] for x0, y0, x1, y1 in bx]
    return np.ascontiguousarray(out), bx


def lr_at(it: int, total_iters: int, base_lr: float,
          warmup_iters: int) -> float:
    """Learning rate for one iteration: linear warmup, then cosine decay.

    Warmup matters specifically for the ``--init-from`` path: the 4-class box
    predictor is freshly initialised, so its early gradients are large and,
    at full LR, they damage the warm-started backbone. That is what produced
    the epoch-2 collapse (debris AP 0.78 -> 0.39) in the first runs.
    """
    if warmup_iters > 0 and it < warmup_iters:
        return base_lr * (0.1 + 0.9 * (it + 1) / warmup_iters)
    span = max(1, total_iters - warmup_iters)
    p = min(1.0, max(0.0, (it - warmup_iters) / span))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * p))


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

    def instance_counts(self) -> dict[int, int]:
        """Annotations per category id — exposes class imbalance/support."""
        out: dict[int, int] = {}
        for anns in self.by_image.values():
            for a in anns:
                cid = int(a["category_id"])
                out[cid] = out.get(cid, 0) + 1
        return out

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

        if self.augment and boxes:
            # full dihedral group (8 orientations) — valid because a
            # microscopy frame has no canonical up
            r = np.random.rand(3)
            img, boxes = dihedral(img, boxes, r[0] < 0.5, r[1] < 0.5,
                                  r[2] < 0.5)

        tensor = torch.from_numpy(img)[None].repeat(3, 1, 1)
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([info["id"]]),
        }
        return tensor, target


def collate(batch):
    return tuple(zip(*batch))


# ResNet stages, outermost first — torchvision's own ordering for deciding
# which to fine-tune.
_BACKBONE_STAGES = ["layer4", "layer3", "layer2", "layer1", "conv1"]


def freeze_backbone(model, trainable_layers: int) -> tuple[int, list[str]]:
    """Freeze all but the last ``trainable_layers`` ResNet stages.

    torchvision only applies ``trainable_backbone_layers`` when the model is
    built WITH pretrained weights; built with ``weights=None`` (which the
    warm-start path must do) it warns and silently trains all 5 stages. So we
    apply the same policy ourselves. Returns
    ``(trainable_param_count, stage_names_left_trainable)``.
    """
    keep = _BACKBONE_STAGES[:max(0, min(5, trainable_layers))]
    if trainable_layers >= 5:
        keep = keep + ["bn1"]
    body = model.backbone.body
    for name, param in body.named_parameters():
        param.requires_grad_(any(name.startswith(k) for k in keep))
    live = sorted({k for k in keep
                   for n, p in body.named_parameters()
                   if n.startswith(k) and p.requires_grad})
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_train, live


def freeze_batchnorm(model) -> int:
    """Replace the backbone's BatchNorm2d with FrozenBatchNorm2d.

    Every pretrained torchvision detector uses FrozenBatchNorm; only the
    ``weights=None`` path gets trainable BatchNorm. With a small batch size
    (4 here) live BatchNorm normalises on noisy batch statistics and keeps
    updating its running stats, which makes fine-tuning measurably less
    stable — a likely contributor to val mAP bouncing between epochs.
    Returns how many layers were converted.
    """
    import torch.nn as nn
    from torchvision.ops.misc import FrozenBatchNorm2d

    converted = 0

    def convert(module):
        nonlocal converted
        for name, child in list(module.named_children()):
            if isinstance(child, nn.BatchNorm2d):
                frozen = FrozenBatchNorm2d(child.num_features)
                with_no_grad = frozen.state_dict()
                del with_no_grad
                frozen.weight.data.copy_(child.weight.data)
                frozen.bias.data.copy_(child.bias.data)
                frozen.running_mean.data.copy_(child.running_mean.data)
                frozen.running_var.data.copy_(child.running_var.data)
                setattr(module, name, frozen)
                converted += 1
            else:
                convert(child)

    convert(model.backbone.body)
    return converted


def build_model(num_classes: int, init_from: str | None = None,
                pretrained_backbone: bool = True,
                trainable_layers: int = 3):
    """Faster R-CNN with ``num_classes`` (including background).

    ``trainable_layers`` is how many ResNet stages get gradients (0-5).
    torchvision defaults to **5** when constructed with ``weights=None`` —
    which the warm-start path does — so all 41M parameters were being
    fine-tuned on a few hundred images. 3 is torchvision's own default for
    fine-tuning: faster per step and markedly less prone to overfitting.
    """
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    if init_from:
        # warm start from the existing cell model: same architecture, but its
        # predictor is 1-class so it must be rebuilt
        from .detector import load_checkpoint_state_dict
        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None,
                                        trainable_backbone_layers=trainable_layers)
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
        model = fasterrcnn_resnet50_fpn(
            weights=weights, trainable_backbone_layers=trainable_layers)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Apply the freezing policy OURSELVES: torchvision ignores
    # trainable_backbone_layers when built without pretrained weights.
    n_bn = freeze_batchnorm(model)
    n_train, live = freeze_backbone(model, trainable_layers)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"model: {n_train/1e6:.1f}M of {n_all/1e6:.1f}M params trainable "
          f"(backbone stages trainable: {len(live)}/5"
          + (f" — {', '.join(live)}" if live else " — backbone fully frozen")
          + f"; {n_bn} BatchNorm layers frozen)")
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


def train(dataset: Path, out: Path, *, epochs: int = 20, batch_size: int = 4,
          lr: float = 2e-3, workers: int = 4, init_from: str | None = None,
          device: str | None = None, trainable_layers: int = 3,
          patience: int = 5, val_every: int = 1, amp: bool = True,
          warmup_frac: float = 0.5) -> None:
    import torch
    from torch.utils.data import DataLoader

    from .detector import _resolve_device

    dev = _resolve_device(torch, device)
    use_amp = bool(amp) and str(dev).startswith("cuda")
    print(f"device: {dev}" + ("  (mixed precision)" if use_amp else ""))

    ann = dataset / "annotations"
    imgs = dataset / "images"
    train_ds = CocoDetectionDataset(imgs, ann / "train.json", augment=True)
    val_json = ann / "val.json"
    val_ds = CocoDetectionDataset(imgs, val_json) if val_json.exists() else None
    print(f"train images: {len(train_ds)}"
          + (f" | val images: {len(val_ds)}" if val_ds else " | NO val set"))

    # Per-class support, because a macro mAP over 3 classes is dominated by
    # whichever class has fewest instances — that is usually why the metric
    # looks unstable from epoch to epoch.
    def _support(ds, name):
        if ds is None:
            return
        counts = ds.instance_counts()
        pretty = ", ".join(f"{CLASS_NAMES[c]} {counts.get(c, 0)}"
                           for c in sorted(CATEGORY_IDS.values()))
        print(f"{name} instances: {pretty}")
        thin = [CLASS_NAMES[c] for c in sorted(CATEGORY_IDS.values())
                if counts.get(c, 0) < 50]
        if thin and name == "val":
            print(f"  \u26a0 few val instances for {', '.join(thin)} — its AP "
                  "will swing by several points between epochs regardless of "
                  "what the model does. Annotate more of that class before "
                  "reading much into the number.")

    _support(train_ds, "train")
    _support(val_ds, "val")

    # A by-file split can land most of a rare class in val, starving training
    # of it. Worth flagging: it looks like a model weakness but is a split
    # artefact, fixable by re-exporting with a different --seed.
    if val_ds is not None and len(val_ds):
        tr_c, va_c = train_ds.instance_counts(), val_ds.instance_counts()
        for cid in sorted(CATEGORY_IDS.values()):
            tr, va = tr_c.get(cid, 0), va_c.get(cid, 0)
            if tr + va == 0:
                continue
            frac = va / (tr + va)
            if frac > 0.35 and tr < 300:
                print(f"  \u26a0 {CLASS_NAMES[cid]}: only {tr} train vs {va} "
                      f"val instances ({frac:.0%} of them are in val). The "
                      "by-file split put much of this class in the held-out "
                      "files, so training barely sees it — re-export with a "
                      "different --seed to rebalance.")
    if val_ds is not None and len(val_ds) == 0:
        print("\u26a0 val set is empty — annotate more ND2 FILES (the split is "
              "by file, deliberately).")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=workers, collate_fn=collate,
                          persistent_workers=workers > 0,
                          pin_memory=str(dev).startswith("cuda"))
    val_dl = (DataLoader(val_ds, batch_size=1, shuffle=False,
                         num_workers=workers, collate_fn=collate)
              if val_ds and len(val_ds) else None)

    num_classes = len(CLASS_NAMES)  # background + 3
    model = build_model(num_classes, init_from=init_from,
                        trainable_layers=trainable_layers).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    iters_per_epoch = max(1, len(train_dl))
    total_iters = epochs * iters_per_epoch
    warmup_iters = int(warmup_frac * iters_per_epoch)
    print(f"schedule: {total_iters} iters, {warmup_iters} warmup, "
          f"peak lr {lr:g}, cosine decay")

    best, best_epoch, since_best = -1.0, 0, 0
    out.parent.mkdir(parents=True, exist_ok=True)
    it = 0
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        total = 0.0
        for imgs_b, targets in train_dl:
            cur_lr = lr_at(it, total_iters, lr, warmup_iters)
            for g in opt.param_groups:
                g["lr"] = cur_lr
            imgs_b = [i.to(dev, non_blocking=True) for i in imgs_b]
            targets = [{k: v.to(dev, non_blocking=True) for k, v in t.items()}
                       for t in targets]
            with torch.amp.autocast("cuda", enabled=use_amp):
                losses = model(imgs_b, targets)
                loss = sum(losses.values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += float(loss.detach())
            it += 1
        dt = time.time() - t0
        msg = (f"epoch {epoch}/{epochs}  train_loss "
               f"{total / iters_per_epoch:.4f}  lr {cur_lr:.2e}  "
               f"{dt:.0f}s ({len(train_ds) / max(dt, 1e-9):.1f} img/s)")

        score = None
        due = val_dl is not None and (epoch % val_every == 0 or epoch == epochs)
        if due:
            metrics = evaluate(model, val_dl, dev)
            score = metrics["mAP@0.5"]
            msg += "  " + "  ".join(
                f"{k} {v:.3f}" for k, v in metrics.items()
                if not (isinstance(v, float) and v != v)  # skip NaN
            )
        print(msg, flush=True)

        if score is not None:
            if score > best:
                best, best_epoch, since_best = score, epoch, 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "classes": CLASS_NAMES,
                    "epoch": epoch,
                    "val_mAP@0.5": score,
                    "init_from": init_from,
                    "trainable_layers": trainable_layers,
                }, out)
                print(f"  saved {out} (mAP {score:.3f})")
            else:
                since_best += val_every
                if patience and since_best >= patience:
                    print(f"  early stop: no val improvement for "
                          f"{since_best} epoch(s); best was epoch "
                          f"{best_epoch} (mAP {best:.3f}). Training longer "
                          "only overfits — the loss keeps falling while val "
                          "does not.")
                    break
        elif val_dl is None and epoch == epochs:
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": CLASS_NAMES, "epoch": epoch,
                "val_mAP@0.5": None, "init_from": init_from,
                "trainable_layers": trainable_layers,
            }, out)
            print(f"  saved {out} (no val set)")

    if best >= 0:
        print(f"done. best val mAP@0.5: {best:.3f} at epoch {best_epoch}")
        if best_epoch <= 2:
            print("  note: the best epoch was the first or second — the warm "
                  "start is doing the work and there is little left to learn "
                  "from this much data. More ANNOTATED FILES will move this "
                  "number; more epochs will not.")
    else:
        print("done.")


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
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-3,
                   help="peak learning rate after warmup (default 2e-3; the "
                        "old 5e-3 with no warmup destabilised a warm start)")
    p.add_argument("--workers", type=int, default=4,
                   help="DataLoader workers. >0 overlaps TIFF reading and "
                        "percentile normalisation with GPU compute; use 0 if "
                        "you hit a Windows multiprocessing problem.")
    p.add_argument("--init-from", default=None,
                   help="warm start from an existing .pth (e.g. the current "
                        "1-class cell_detection_model.pth)")
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--trainable-layers", type=int, default=3,
                   choices=[0, 1, 2, 3, 4, 5],
                   help="ResNet stages to fine-tune (default 3). Lower = "
                        "faster and less overfitting on a small dataset; 5 "
                        "trains the whole backbone.")
    p.add_argument("--patience", type=int, default=5,
                   help="stop after this many epochs without val improvement "
                        "(0 disables)")
    p.add_argument("--val-every", type=int, default=1,
                   help="evaluate every N epochs (2 or 3 saves time once you "
                        "know it converges early)")
    p.add_argument("--no-amp", action="store_true",
                   help="disable mixed precision (on by default on CUDA)")
    args = p.parse_args()
    train(Path(args.dataset), Path(args.out), epochs=args.epochs,
          batch_size=args.batch_size, lr=args.lr, workers=args.workers,
          init_from=args.init_from, device=args.device,
          trainable_layers=args.trainable_layers, patience=args.patience,
          val_every=args.val_every, amp=not args.no_amp)


if __name__ == "__main__":
    main()
