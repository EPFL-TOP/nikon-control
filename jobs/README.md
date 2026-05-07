# JOBS workflows

This folder will hold the JOBS workflow files (`.bin` / `.job`) and notes for
recreating them in NIS Elements 6.20.

## Phase 0 — manual capture

For Phase 0 there is no JOBS workflow. The loop is:

1. In NIS Elements, set the 10x objective and put one well in focus.
2. Acquire a single brightfield image.
3. Save it as `.nd2` (or export `.tif`) into the repo's `data/` folder.
4. From the repo root, run:

   ```
   python -m nikon_control.cli data/your_capture.nd2 --out out/overlay.png
   ```

   This validates that loading + the placeholder BF detector run end-to-end
   and produces an overlay PNG you can eyeball.

## Phase 1+ (planned)

- A JOBS workflow that does a **wellplate 10x BF scan** with stitching, saves
  the stitched mosaic to a known location, and emits a JSON sidecar with
  pixel size + stage origin (so Python can map pixels back to stage XY).
- A second JOBS workflow that consumes a Python-generated point list and
  runs the **40x time-lapse** over those points.

The Nikon-published example most relevant to the lo-mag detect → hi-mag
acquisition flow is at:
`../../nikon-microscopy/JOBS-examples/NIS_v5.42/15-Mouse_embryo/`.

## Watching JOBS Python output (no terminal)

JOBS Python tasks have **no attached console**. `print()` calls go into
NIS-Elements' own log file with a `PYTHON OUT:` prefix, which is awkward to
follow live.

Two practical options:

1. **NIS log viewer**: from a Python task, `import nis; nis.mac.OpenLogFile()`.
   Or use the menu — usually `View → Log` or `Tools → Log Viewer`.

2. **Tail our own log file** (recommended for development). Use
   `nikon_control.jobs_log.log(msg)` in JOBS Python tasks; it appends to a
   file we control with explicit flush:

   ```python
   import sys
   sys.path.append(r"C:\path\to\nikon-control\src")
   from nikon_control.jobs_log import log

   log("acquired tile %d" % i)
   ```

   The default path is `C:\temp\nikon_control.log` on Windows; override with
   the `NIKON_CONTROL_LOG` environment variable.

   In a separate PowerShell window, tail it:

   ```powershell
   Get-Content C:\temp\nikon_control.log -Wait
   ```

   That's your "terminal".

## Verifying which APIs v6.20 actually exposes

Before writing any real Phase 1 code, drop the snippet below into a Python
Script task on the microscope PC to discover what's available. The output
goes to `C:\temp\nikon_control.log` (tail it from PowerShell as above):

```python
import sys
sys.path.append(r"C:\path\to\nikon-control\src")
from nikon_control.jobs_log import log

try:
    import limjob
    log("limjob: OK")
    log("limjob attrs: %s" % sorted(a for a in dir(limjob) if not a.startswith("_")))
except Exception as e:
    log("limjob FAILED: %r" % e)

try:
    import nis
    log("nis: OK")
    macs = sorted(a for a in dir(nis.mac) if not a.startswith("_"))
    log("nis.mac count: %d" % len(macs))
    # macro names we expect to need:
    needed = ["StgMoveXY", "StgMoveTo", "PiezoXYMoveToXYPosition",
              "StgGetPosX", "StgGetPosY", "StgGetPosZ",
              "ObjectiveChange", "OptConfig",
              "PFS_On", "PFS_Off", "PFS_GetOffset", "PFS_SetOffset",
              "Capture", "CaptureND",
              "Jobs_GetJobrunFolder"]
    found = {m: any(m.lower() in x.lower() for x in macs) for m in needed}
    log("needed macro presence: %s" % found)
except Exception as e:
    log("nis FAILED: %r" % e)
```

This tells us in one task run:

- whether `limjob` and `nis` exist in v6.20,
- which of the macro functions we expect to need are actually present (or
  what they're called instead — the substring match catches near misses).
