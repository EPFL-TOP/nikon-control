"""Route ``/scope`` — drive the microscope through Micro-Manager."""
import os

from bokeh.plotting import curdoc

from nikon_control.dashboard.scope_app import modify_doc

modify_doc(
    curdoc(),
    os.environ.get("NIKON_CONTROL_MM_CONFIG", ""),
    os.environ.get("NIKON_CONTROL_PLATE", ""),
)
