"""Tests for the training pieces that can silently corrupt a run."""
import numpy as np
import pytest

from nikon_control.train_detector import CLASS_NAMES, dihedral, lr_at


def _frame_with_blob(H=40, W=60, y0=5, x0=10, y1=15, x1=30):
    """A frame that is zero except one bright rectangle, plus its xyxy box."""
    img = np.zeros((H, W), dtype=np.float32)
    img[y0:y1, x0:x1] = 1.0
    return img, [float(x0), float(y0), float(x1), float(y1)]


def _box_of_blob(img):
    """Recover the tight xyxy box around the bright pixels."""
    ys, xs = np.nonzero(img > 0.5)
    return [float(xs.min()), float(ys.min()),
            float(xs.max() + 1), float(ys.max() + 1)]


ALL_EIGHT = [(t, h, v) for t in (False, True)
             for h in (False, True) for v in (False, True)]


@pytest.mark.parametrize("transpose,hflip,vflip", ALL_EIGHT)
def test_every_dihedral_transform_keeps_the_box_on_the_object(
        transpose, hflip, vflip):
    """The whole point: if boxes don't follow the image, training learns
    garbage while the loss still looks fine."""
    img, box = _frame_with_blob()
    out, boxes = dihedral(img, [box], transpose, hflip, vflip)
    assert len(boxes) == 1
    assert _box_of_blob(out) == pytest.approx(boxes[0]), (
        transpose, hflip, vflip)


def test_identity_transform_changes_nothing():
    img, box = _frame_with_blob()
    out, boxes = dihedral(img, [box], False, False, False)
    assert np.array_equal(out, img)
    assert boxes == [box]


def test_transpose_swaps_the_shape():
    img, box = _frame_with_blob(H=40, W=60)
    out, _ = dihedral(img, [box], True, False, False)
    assert out.shape == (60, 40)


def test_flips_preserve_the_shape_and_box_size():
    img, box = _frame_with_blob()
    for h, v in ((True, False), (False, True), (True, True)):
        out, boxes = dihedral(img, [box], False, h, v)
        assert out.shape == img.shape
        w0, h0 = box[2] - box[0], box[3] - box[1]
        w1, h1 = boxes[0][2] - boxes[0][0], boxes[0][3] - boxes[0][1]
        assert (w1, h1) == (w0, h0)


def test_boxes_stay_inside_the_frame():
    img, box = _frame_with_blob()
    for t, h, v in ALL_EIGHT:
        out, boxes = dihedral(img, [box], t, h, v)
        H, W = out.shape
        x0, y0, x1, y1 = boxes[0]
        assert 0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H, (t, h, v)


def test_multiple_boxes_all_transform():
    img = np.zeros((40, 40), dtype=np.float32)
    boxes = [[1, 2, 5, 8], [20, 20, 30, 35]]
    _, out = dihedral(img, boxes, True, True, False)
    assert len(out) == 2
    assert out[0] != out[1]


def test_dihedral_does_not_mutate_its_inputs():
    img, box = _frame_with_blob()
    img_copy, boxes_in = img.copy(), [list(box)]
    dihedral(img, boxes_in, True, True, True)
    assert np.array_equal(img, img_copy)
    assert boxes_in == [list(box)]


# ---- LR schedule --------------------------------------------------------

def test_warmup_ramps_from_a_tenth_to_the_peak():
    lr, warm, total = 2e-3, 100, 1000
    assert lr_at(0, total, lr, warm) == pytest.approx(lr * (0.1 + 0.9 / 100))
    assert lr_at(warm - 1, total, lr, warm) == pytest.approx(lr)
    # monotonically increasing through warmup
    vals = [lr_at(i, total, lr, warm) for i in range(warm)]
    assert vals == sorted(vals)
    assert all(v <= lr + 1e-12 for v in vals)


def test_cosine_decays_to_zero_after_warmup():
    lr, warm, total = 2e-3, 100, 1000
    assert lr_at(warm, total, lr, warm) == pytest.approx(lr, rel=1e-3)
    mid = lr_at(warm + (total - warm) // 2, total, lr, warm)
    assert 0.4 * lr < mid < 0.6 * lr
    assert lr_at(total, total, lr, warm) == pytest.approx(0.0, abs=1e-12)


def test_lr_never_exceeds_the_peak_or_goes_negative():
    lr, warm, total = 5e-3, 50, 500
    vals = [lr_at(i, total, lr, warm) for i in range(total + 10)]
    assert max(vals) <= lr + 1e-12
    assert min(vals) >= 0.0


def test_no_warmup_is_pure_cosine():
    lr, total = 1e-3, 100
    assert lr_at(0, total, lr, 0) == pytest.approx(lr)
    assert lr_at(total, total, lr, 0) == pytest.approx(0.0, abs=1e-12)


def test_class_names_line_up_with_the_head():
    assert CLASS_NAMES[0] == "__background__"
    assert CLASS_NAMES[1:] == ["single", "doublet", "debris"]


# ---- backbone freezing (torchvision silently ignores the request) --------

@pytest.mark.parametrize("n,expected", [
    (0, []),
    (1, ["layer4"]),
    (2, ["layer3", "layer4"]),
    (3, ["layer2", "layer3", "layer4"]),
    (5, ["conv1", "layer1", "layer2", "layer3", "layer4"]),
])
def test_freeze_backbone_trains_exactly_the_outermost_stages(n, expected):
    """torchvision only honours trainable_backbone_layers when built with
    pretrained weights; the warm-start path must apply it itself."""
    from nikon_control.train_detector import build_model, freeze_backbone

    model = build_model(4, pretrained_backbone=False, trainable_layers=5)
    freeze_backbone(model, n)
    body = model.backbone.body
    live = [k for k in ["conv1", "layer1", "layer2", "layer3", "layer4"]
            if any(p.requires_grad for nm, p in body.named_parameters()
                   if nm.startswith(k))]
    assert live == expected


def test_freezing_fewer_stages_reduces_trainable_params_monotonically():
    from nikon_control.train_detector import build_model, freeze_backbone

    model = build_model(4, pretrained_backbone=False, trainable_layers=5)
    counts = []
    for n in (0, 1, 2, 3, 5):
        freeze_backbone(model, n)
        counts.append(sum(p.numel() for p in model.parameters()
                          if p.requires_grad))
    assert counts == sorted(counts)
    assert counts[0] < counts[-1]


def test_freeze_batchnorm_converts_and_preserves_weights():
    """Live BatchNorm at batch size 4 makes fine-tuning noisy; every
    pretrained torchvision detector uses FrozenBatchNorm instead."""
    import torch.nn as nn
    from torchvision.ops.misc import FrozenBatchNorm2d

    from nikon_control.train_detector import build_model, freeze_batchnorm

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Module()
            self.backbone.body = nn.Sequential(nn.BatchNorm2d(4),
                                               nn.Conv2d(4, 4, 1))

    m = Tiny()
    bn = m.backbone.body[0]
    bn.weight.data.fill_(2.0)
    bn.running_mean.data.fill_(3.0)
    n = freeze_batchnorm(m)
    assert n == 1
    frozen = m.backbone.body[0]
    assert isinstance(frozen, FrozenBatchNorm2d)
    assert float(frozen.weight[0]) == 2.0        # weights carried over
    assert float(frozen.running_mean[0]) == 3.0
