import numpy as np
import tifffile

from nikon_control.io import load_image


def test_load_tif_returns_2d_array(tmp_path):
    arr = (np.random.default_rng(0).random((128, 128)) * 255).astype(np.uint8)
    p = tmp_path / "x.tif"
    tifffile.imwrite(p, arr)

    loaded = load_image(p)

    assert loaded.shape == (128, 128)
    assert loaded.dtype == np.uint8


def test_load_tif_collapses_extra_dims(tmp_path):
    stack = np.zeros((3, 64, 64), dtype=np.uint16)
    stack[0] = 7
    p = tmp_path / "stack.tif"
    tifffile.imwrite(p, stack)

    loaded = load_image(p)

    assert loaded.shape == (64, 64)
    assert (loaded == 7).all()
