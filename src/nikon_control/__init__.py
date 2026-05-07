"""nikon-control

Tooling to drive a Nikon microscope (NIS Elements + JOBS 6.20) for the
pipeline:

    10x BF scan  ->  detect cells  ->  plan 40x tile path  ->  40x time-lapse

Phase 0 (this scaffold) wires the loader and a placeholder BF detector so the
end-to-end data path can be validated before adding real segmentation, tile
planning, or a dashboard.
"""

__version__ = "0.0.1"
