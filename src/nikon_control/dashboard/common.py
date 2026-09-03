"""Pieces shared by both annotation dashboards.

The full dashboard (``app.py``, tracked annotations) and the simplified one
(``simple_app.py``, independent per-frame boxes) differ only in their
*annotation* panel. Everything to do with **viewing** (the image figure,
contrast, the ND2 plane reader) and **finding a file** (the drive/folder
browser) lives here so a fix lands once for both.

The pure helpers import no GUI toolkit, so they are unit-testable; the
widget builder imports bokeh lazily inside the function.
"""
from __future__ import annotations

import os
import string
from pathlib import Path

import numpy as np

# ND2 reading lives in ``io.py``; re-exported here so both dashboards
# (and the training exporter) share one implementation.
from ..io import bf_channel_index, open_nd2, plane_extractor  # noqa: F401

# distinct line colours per class (cycled)
PALETTE = ["#ff3b30", "#ffcc00", "#34c759", "#00c7be", "#ff9500", "#af52de"]

# Placeholder for a category dropdown used as a *command* menu (pick a class
# -> apply to the selected box -> reset to this placeholder), so re-picking
# the same class for the next cell still fires on_change.
PICK_CATEGORY = "— set category —"

# Fallback model locations tried when --weights isn't given and no .pth sits
# in the data folder. Add site-specific defaults here.
DEFAULT_WEIGHTS = [
    r"E:\PROJECTS-01\Clement\cell_detection_model.pth",
]

# Fallback data folders the browser starts in when the launch folder has no
# ND2 files. Add site-specific defaults here.
DEFAULT_DATA_DIRS = [
    r"G:\PROJECTS-02\Samuel",
]


def list_drives() -> list[str]:
    """Available volumes to jump between: Windows drive letters, or the root
    and /Volumes mounts on macOS/Linux."""
    if os.name == "nt":
        return [f"{c}:\\" for c in string.ascii_uppercase
                if os.path.exists(f"{c}:\\")]
    vols = ["/"]
    v = Path("/Volumes")
    if v.exists():
        try:
            vols += [str(p) for p in sorted(v.iterdir()) if p.is_dir()]
        except Exception:
            pass
    return vols


def class_color(classes: list[str]) -> dict[str, str]:
    return {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(classes)}


def resolve_data_dir(data_dir: str | Path) -> Path:
    """Start the browser at a site-default folder when the launch folder has
    no ND2s (e.g. a bare ``--show`` launch from the cwd)."""
    data_dir = Path(data_dir)
    try:
        if any(data_dir.glob("*.nd2")):
            return data_dir
    except Exception:
        pass
    for cand in DEFAULT_DATA_DIRS:
        if Path(cand).is_dir():
            return Path(cand)
    return data_dir


def resolve_weights(weights_path: str) -> str:
    """Fall back to a site default model if none was given at launch."""
    if weights_path:
        return weights_path
    for cand in DEFAULT_WEIGHTS:
        if Path(cand).exists():
            return cand
    return ""


def scan_folder(path: str | Path) -> tuple[list[str], list[str], list[str]]:
    """List a folder for the in-page browser.

    Returns ``(subdirs, nd2_files, pth_files)`` as bare names, sorted
    case-insensitively. Raises OSError-ish on an unreadable folder — the
    caller reports it to the user.
    """
    d = Path(path).expanduser()
    entries = sorted(d.iterdir(), key=lambda x: x.name.lower())
    subs = [p.name for p in entries if p.is_dir() and not p.name.startswith(".")]
    nd2s = [p.name for p in entries if p.suffix.lower() == ".nd2"]
    pths = [p.name for p in entries if p.suffix.lower() == ".pth"]
    return subs, nd2s, pths


def contrast_bounds(plane: np.ndarray) -> tuple[float, float, float, float, float]:
    """Slider bounds + default window for a plane.

    Returns ``(mn, mx, lo, hi, step)``. The slider spans the full data
    min..max so the high end can be pushed all the way up for bright
    fluorescence; the default window trims only the 0.1% tails so a lone
    hot/dead pixel doesn't wreck it. ~500 steps keeps the handles smooth.
    """
    mn, mx = float(plane.min()), float(plane.max())
    if mx <= mn:
        mx = mn + 1.0
    lo, hi = (float(v) for v in np.percentile(plane, [0.1, 99.9]))
    if hi <= lo:
        hi = lo + 1.0
    step = max(1.0, (mx - mn) / 500.0)
    return mn, mx, lo, hi, step


def build_image_figure(width: int = 760, height: int = 760):
    """Build the shared image figure + box overlay.

    Returns ``(fig, img_src, img_r, box_src, rect_r, mapper)``.

    Tool setup is deliberate: **tap** selects a box and the default drag is
    **pan**, so a stray click can't start drawing a box that then sticks to
    the cursor. The Box-Edit tool stays in the toolbar for deliberate
    move/draw (Esc cancels a half-drawn box).
    """
    from bokeh.models import (
        BoxEditTool,
        ColumnDataSource,
        LabelSet,
        LinearColorMapper,
        PanTool,
        TapTool,
    )
    from bokeh.plotting import figure

    img_src = ColumnDataSource({"image": [np.zeros((2, 2), dtype=np.float32)]})
    box_src = ColumnDataSource(
        {"id": [], "num": [], "label": [], "cx": [], "cy": [], "w": [],
         "h": [], "marker": [], "color": [], "text": []}
    )
    mapper = LinearColorMapper(palette="Greys256", low=0, high=65535)
    fig = figure(width=width, height=height, match_aspect=True,
                 tools="pan,wheel_zoom,reset", title="(no file loaded)")
    img_r = fig.image(image="image", x=0, y=0, dw=1, dh=1, source=img_src,
                      color_mapper=mapper, level="image")
    rect_r = fig.rect(
        x="cx", y="cy", width="w", height="h", source=box_src,
        fill_alpha=0.0, line_color="color", line_width=3,
        nonselection_fill_alpha=0.0, nonselection_line_alpha=0.35,
        selection_fill_color="color", selection_fill_alpha=0.18,
        selection_line_color="white", selection_line_width=5,
    )
    fig.add_layout(
        LabelSet(
            x="cx", y="cy", text="text", source=box_src,
            text_color="white", text_font_size="10pt", text_font_style="bold",
            background_fill_color="black", background_fill_alpha=0.55,
            y_offset=12,
        )
    )
    box_tool = BoxEditTool(renderers=[rect_r], empty_value="")
    tap_tool = TapTool(renderers=[rect_r])
    fig.add_tools(box_tool, tap_tool)
    fig.toolbar.active_tap = tap_tool
    pan = fig.select_one(PanTool)
    if pan is not None:
        fig.toolbar.active_drag = pan
    return fig, img_src, img_r, box_src, rect_r, mapper
