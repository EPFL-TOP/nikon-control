"""Verify that existing annotation files pick up new DEFAULT_CLASSES on open.

We don't import napari here — the merge logic in `open_for_annotation` is
hard to exercise without a GUI, so we simulate the same code path against
the persisted JSON: load an old file, simulate the merge, re-save, and
check the class list grew.
"""
from nikon_control.annotate import (
    DEFAULT_CLASSES,
    SCHEMA_VERSION,
    load,
    save,
)


def test_loaded_classes_can_be_extended(tmp_path):
    """An old JSON with three classes gains the new ones on re-save."""
    # Write a v0.5 file that predates fission_fusion having been added.
    old_payload = """{
  "source": "test.nd2",
  "schema_version": "0.5",
  "image_shape": [60, 3, 1024, 1024],
  "axes": ["T", "C", "Y", "X"],
  "channels": ["BF"],
  "classes": ["single", "doublet", "debris"],
  "annotator": "",
  "annotations": []
}
"""
    p = tmp_path / "old.annotations.json"
    p.write_text(old_payload)

    state = load(p)
    assert "fission_fusion" not in state.classes  # not in the file

    # Simulate the merge that open_for_annotation does on load
    for c in DEFAULT_CLASSES:
        if c not in state.classes:
            state.classes.append(c)

    save(state, p)
    reloaded = load(p)
    assert reloaded.schema_version == SCHEMA_VERSION
    assert "fission_fusion" in reloaded.classes
    # And the existing order is preserved
    assert reloaded.classes[:3] == ["single", "doublet", "debris"]
