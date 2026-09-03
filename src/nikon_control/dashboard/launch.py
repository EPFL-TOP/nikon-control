"""``nikon-control-dashboard`` — launch the Bokeh annotation dashboards.

Wraps ``bokeh serve`` so users get a single command. TWO dashboards are
served by the one process, sharing the viewer and file browser:

- ``/annotate`` — the full dashboard: tracked annotations with keyframes and
  lifecycle (birth / end / deaths / class changes).
- ``/simple`` — the simplified dashboard: independent per-frame boxes over
  the first N frames, three classes (single / doublet / debris), written to
  ``<file>.simple.json``. This is the one for building training data.

``/`` lists both. The data directory (folder of ND2 files + their sidecar
JSONs) reaches the apps via the ``NIKON_CONTROL_DATA`` environment variable.

Examples::

    nikon-control-dashboard --data-dir Z:/experiments/2026-07 --show
    nikon-control-dashboard --data-dir . --port 5006 \\
        --allow-websocket-origin server.epfl.ch:5006
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(prog="nikon-control-dashboard")
    p.add_argument("--data-dir", default=".",
                   help="folder containing .nd2 files (default: cwd)")
    p.add_argument("--weights", default="",
                   help="path to cell_detection_model.pth (enables the "
                        "in-dashboard Detect button)")
    p.add_argument("--port", type=int, default=5006)
    p.add_argument("--address", default=None,
                   help="bind address. Omit for localhost (fine when users "
                        "RDP into this machine); use 0.0.0.0 to accept "
                        "connections from other machines' browsers.")
    p.add_argument("--show", action="store_true",
                   help="open a browser tab automatically")
    p.add_argument("--allow-websocket-origin", action="append", default=[],
                   help="host:port allowed to connect (repeatable; needed "
                        "when reaching the server from another machine)")
    args = p.parse_args()

    apps_dir = Path(__file__).parent / "apps"
    server_scripts = [str(apps_dir / "annotate.py"), str(apps_dir / "simple.py")]
    env = dict(os.environ)
    env["NIKON_CONTROL_DATA"] = str(Path(args.data_dir).resolve())
    if args.weights:
        env["NIKON_CONTROL_WEIGHTS"] = str(Path(args.weights).resolve())

    cmd = [sys.executable, "-m", "bokeh", "serve", *server_scripts,
           "--port", str(args.port)]
    if args.address:
        cmd += ["--address", args.address]
    if args.show:
        cmd.append("--show")
    for origin in args.allow_websocket_origin:
        cmd += ["--allow-websocket-origin", origin]

    host = args.address or "localhost"
    print("data dir:", env["NIKON_CONTROL_DATA"])
    print(f"  full dashboard  : http://{host}:{args.port}/annotate")
    print(f"  simple dashboard: http://{host}:{args.port}/simple")
    print("running:", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    main()
