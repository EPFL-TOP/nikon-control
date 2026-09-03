"""Route ``/annotate`` — the full dashboard (tracked annotations)."""
import os

from bokeh.plotting import curdoc

from nikon_control.dashboard.app import modify_doc

modify_doc(
    curdoc(),
    os.environ.get("NIKON_CONTROL_DATA", "."),
    os.environ.get("NIKON_CONTROL_WEIGHTS", ""),
)
