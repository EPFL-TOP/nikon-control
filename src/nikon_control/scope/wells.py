"""Choose which wells to image — plugin 2 of the acquisition pipeline.

Pure selection logic over a registered plate: no hardware, no GUI. The
dashboard's plate map is a view onto this, and a script can drive the same
thing without a browser.

A selection is just a set of well names, but every convenient way of
expressing one — a row, a column, a rectangle between two corners, "every
other well" — is easy to get subtly wrong by hand, and a wrong well set is
only discovered hours into an overnight run. So the parsing and the
range logic live here with tests, and produce a ``useq.WellPlatePlan`` whose
``selected_wells`` the acquisition consumes directly.

Well names are the plate's own, taken from ``useq`` rather than re-derived:
``A1``…``H12`` on a 96-well plate, and a 1536-well plate's rows run past ``Z``
to ``AA``, which is exactly the kind of thing worth not re-implementing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# "A1", "h12", "AA3" — a row of letters then a column number.
_WELL = re.compile(r"^([A-Za-z]+)\s*(\d+)$")


@dataclass
class WellGrid:
    """The plate's well names laid out as rows x columns."""

    rows: list[str]                       # "A", "B", …
    columns: list[str]                    # "1", "2", …
    names: list[list[str]]                # names[r][c]

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_columns(self) -> int:
        return len(self.columns)

    def index(self, name: str) -> tuple[int, int] | None:
        key = normalise(name)
        for r, row in enumerate(self.names):
            for c, well in enumerate(row):
                if well == key:
                    return r, c
        return None

    def all_names(self) -> list[str]:
        return [n for row in self.names for n in row]

    def mask(self, selected) -> list[list[bool]]:
        chosen = {normalise(n) for n in selected}
        return [[n in chosen for n in row] for row in self.names]


def normalise(name: str) -> str:
    """``a1`` / ``A 1`` / ``A01`` -> ``A1``, so user input matches useq's names."""
    m = _WELL.match(str(name).strip())
    if not m:
        return str(name).strip().upper()
    return f"{m.group(1).upper()}{int(m.group(2))}"


def grid(plate: str) -> WellGrid:
    """The well names of a plate, from useq's own geometry."""
    import numpy as np
    from useq import WellPlatePlan

    plan = WellPlatePlan(plate=plate, a1_center_xy=(0.0, 0.0))
    names = np.asarray(plan.all_well_names)
    rows_cols = [[str(n) for n in row] for row in names]
    row_labels, col_labels = [], []
    for row in rows_cols:
        m = _WELL.match(row[0])
        row_labels.append(m.group(1) if m else row[0])
    for cell in rows_cols[0]:
        m = _WELL.match(cell)
        col_labels.append(m.group(2) if m else cell)
    return WellGrid(rows=row_labels, columns=col_labels, names=rows_cols)


@dataclass
class Selection:
    """Which wells are chosen, the plate they belong to, and in what order."""

    plate: str
    wells: set[str] = field(default_factory=set)
    # Serpentine by default: it is strictly less travel and never worse.
    serpentine_order: bool = True

    def __post_init__(self) -> None:
        self.wells = {normalise(w) for w in self.wells}

    # ------------------------------------------------------------- editing

    def toggle(self, name: str) -> "Selection":
        key = normalise(name)
        self.wells.discard(key) if key in self.wells else self.wells.add(key)
        return self

    def add(self, names) -> "Selection":
        self.wells |= {normalise(n) for n in names}
        return self

    def remove(self, names) -> "Selection":
        self.wells -= {normalise(n) for n in names}
        return self

    def clear(self) -> "Selection":
        self.wells.clear()
        return self

    def select_all(self) -> "Selection":
        return self.add(grid(self.plate).all_names())

    def row(self, label: str) -> list[str]:
        g = grid(self.plate)
        want = str(label).strip().upper()
        for i, r in enumerate(g.rows):
            if r == want:
                return list(g.names[i])
        return []

    def column(self, label) -> list[str]:
        g = grid(self.plate)
        want = str(label).strip()
        for i, c in enumerate(g.columns):
            if c == want:
                return [row[i] for row in g.names]
        return []

    def rectangle(self, corner_a: str, corner_b: str) -> list[str]:
        """Every well in the block spanned by two corners, in any order."""
        g = grid(self.plate)
        a, b = g.index(corner_a), g.index(corner_b)
        if a is None or b is None:
            return []
        r0, r1 = sorted((a[0], b[0]))
        c0, c1 = sorted((a[1], b[1]))
        return [g.names[r][c]
                for r in range(r0, r1 + 1)
                for c in range(c0, c1 + 1)]

    # ------------------------------------------------------------ ordering

    def ordered(self) -> list[str]:
        """Selected wells in plate (raster) order."""
        g = grid(self.plate)
        return [n for n in g.all_names() if n in self.wells]

    def visiting_order(self) -> list[str]:
        """The order a scan should actually visit — what everything downstream
        must use, so the choice reaches the plan and the saved file rather
        than only the screen."""
        return self.serpentine() if self.serpentine_order else self.ordered()

    def serpentine(self) -> list[str]:
        """Plate order, but alternate rows reversed.

        Halves the stage travel of a full-plate scan: a raster returns across
        the whole plate at the end of every row, a serpentine never does.
        """
        g = grid(self.plate)
        out: list[str] = []
        for i, row in enumerate(g.names):
            names = [n for n in row if n in self.wells]
            out.extend(reversed(names) if i % 2 else names)
        return out

    # ---------------------------------------------------------- the output

    def to_plan(self, calibration, **kwargs):
        """A ``useq.WellPlatePlan`` for these wells on a registered plate.

        ``selected_wells`` is useq's own format: a pair of index arrays, the
        same shape numpy fancy-indexing uses.
        """
        if calibration.plate != self.plate:
            raise ValueError(
                f"selection is for a {self.plate} plate but the calibration "
                f"is for a {calibration.plate} — register the plate you are "
                f"actually using."
            )
        g = grid(self.plate)
        rows, cols = [], []
        for name in self.visiting_order():
            idx = g.index(name)
            if idx is not None:
                rows.append(idx[0])
                cols.append(idx[1])
        return calibration.to_plan(selected_wells=(rows, cols), **kwargs)

    def describe(self) -> str:
        if not self.wells:
            return f"{self.plate}: no wells selected"
        names = self.visiting_order()
        shown = ", ".join(names[:8]) + (" …" if len(names) > 8 else "")
        order = "serpentine" if self.serpentine_order else "raster"
        return f"{self.plate}: {len(names)} well(s), {order} — {shown}"


def parse(plate: str, text: str) -> list[str]:
    """Read a human well list: ``A1, B2-B5, C*, *3, A1:D6``.

    Deliberately forgiving, because this is typed by hand:

    - ``A1``          one well
    - ``B2-B5``       a range along a row (or down a column)
    - ``A1:D6``       the rectangle between two corners
    - ``C*``          all of row C
    - ``*3``          all of column 3
    - ``all``         every well

    Unrecognised items are skipped rather than raising — the caller compares
    the count it asked for with the count it got.
    """
    g = grid(plate)
    sel = Selection(plate)
    out: list[str] = []
    for piece in re.split(r"[,\s]+", str(text).strip()):
        if not piece:
            continue
        low = piece.lower()
        if low == "all":
            out.extend(g.all_names())
        elif ":" in piece:
            a, b = piece.split(":", 1)
            out.extend(sel.rectangle(a, b))
        elif "-" in piece and not piece.startswith("-"):
            a, b = piece.split("-", 1)
            out.extend(sel.rectangle(a, b))
        elif piece.endswith("*"):
            out.extend(sel.row(piece[:-1]))
        elif piece.startswith("*"):
            out.extend(sel.column(piece[1:]))
        else:
            name = normalise(piece)
            if g.index(name) is not None:
                out.append(name)
    seen: set[str] = set()
    return [n for n in out if not (n in seen or seen.add(n))]


# --------------------------------------------------------------- persistence
# The selection lives in the same JSON as the plate calibration: one file per
# plate describes both where it is and which wells matter, and the
# acquisition reads one thing rather than correlating two.

def load_selection(path) -> Selection | None:
    """Read a selection from a plate file, if it carries one."""
    import json
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text())
    except (ValueError, OSError):
        return None
    plate = obj.get("plate")
    if not plate:
        return None
    return Selection(str(plate), set(obj.get("wells", [])),
                     serpentine_order=bool(obj.get("serpentine", True)))


def save_selection(selection: Selection, path) -> None:
    """Merge a selection into a plate file, leaving the calibration alone."""
    import json
    from pathlib import Path

    p = Path(path)
    obj = {}
    if p.exists():
        try:
            obj = json.loads(p.read_text())
        except (ValueError, OSError):
            obj = {}
    if obj.get("plate") and obj["plate"] != selection.plate:
        raise ValueError(
            f"{p.name} holds a {obj['plate']} calibration; refusing to write "
            f"a {selection.plate} selection over it."
        )
    obj.setdefault("plate", selection.plate)
    # Saved in visiting order, so the file records the route rather than
    # leaving every reader to re-derive it (and possibly differently).
    obj["wells"] = selection.visiting_order()
    obj["serpentine"] = selection.serpentine_order
    p.write_text(json.dumps(obj, indent=2))
