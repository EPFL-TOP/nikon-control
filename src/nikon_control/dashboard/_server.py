"""Bokeh server entry point — executed by ``bokeh serve``.

Reads the data directory from the ``NIKON_CONTROL_DATA`` environment
variable (default: current directory) and builds the app document. Kept
separate from ``app.py`` so ``app.modify_doc`` stays import-safe for tests.
"""
import os

from bokeh.plotting import curdoc

from nikon_control.dashboard.app import modify_doc

modify_doc(
    curdoc(),
    os.environ.get("NIKON_CONTROL_DATA", "."),
    os.environ.get("NIKON_CONTROL_WEIGHTS", ""),
)
