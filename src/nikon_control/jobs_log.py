"""Logging helper for code that runs inside a JOBS Python task.

JOBS Python tasks have no attached console: ``print()`` output is routed into
the NIS-Elements log file with a ``PYTHON OUT:`` prefix, which is awkward to
follow live. ``log(msg)`` here appends a timestamped line to a file we control
(with explicit flush) and also calls ``print()`` so the line still ends up in
the NIS log.

Usage on the Windows microscope PC:

1. Set the log path before running the JOB::

       setx NIKON_CONTROL_LOG "C:\\temp\\nikon_control.log"

2. Tail it from a PowerShell window::

       Get-Content C:\\temp\\nikon_control.log -Wait

3. From a JOBS Python task::

       import sys
       sys.path.append(r"C:\\path\\to\\nikon-control\\src")
       from nikon_control.jobs_log import log

       log("acquired tile %d" % i)
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

_DEFAULT_LOG_PATH = (
    Path(r"C:\temp\nikon_control.log")
    if sys.platform == "win32"
    else Path("/tmp/nikon_control.log")
)


def log_path() -> Path:
    return Path(os.environ.get("NIKON_CONTROL_LOG", str(_DEFAULT_LOG_PATH)))


def log(msg: object) -> None:
    """Append a timestamped line to the log file and to stdout."""
    line = f"{datetime.now().isoformat(timespec='milliseconds')}  {msg}"
    p = log_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
    print(line)
