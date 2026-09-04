"""Route ``/review`` — review an exported training dataset."""
import os

from bokeh.plotting import curdoc

from nikon_control.dashboard.review_app import modify_doc

# Starts in NIKON_CONTROL_DATASET when set, else the data dir, else cwd.
modify_doc(
    curdoc(),
    os.environ.get("NIKON_CONTROL_DATASET")
    or os.environ.get("NIKON_CONTROL_DATA", "."),
)
