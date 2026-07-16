"""Bokeh annotation dashboard — the replacement for the napari tool.

Split into:

- ``state.py``  — pure, GUI-free controller (unit-tested). Owns the
  annotations, maps stable ids to them, and implements every mutation
  (add / move / delete box, category, lifecycle). No Bokeh import.
- ``app.py``    — thin Bokeh view wiring widgets to the controller.
- ``launch.py`` — ``nikon-control-dashboard`` console entry (runs
  ``bokeh serve``).

The controller is where the correctness lives, precisely because a live
GUI is hard to test; the view stays minimal.
"""
