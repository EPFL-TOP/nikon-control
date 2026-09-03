"""Bokeh server entry points.

One module per served app; ``bokeh serve`` turns each filename into a URL
route (``annotate.py`` -> ``/annotate``, ``simple.py`` -> ``/simple``), and
serves an index listing them at ``/``. Kept separate from the ``app`` /
``simple_app`` modules so ``modify_doc`` stays import-safe for tests.
"""
