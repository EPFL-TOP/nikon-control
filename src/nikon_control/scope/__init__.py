"""Direct microscope control via Micro-Manager / pymmcore-plus.

Separate from the annotation and training packages: nothing here is imported
by them, and nothing here needs a GPU. The hardware-touching parts import
pymmcore-plus lazily so the pure modules (``plate``) stay testable anywhere.
"""
