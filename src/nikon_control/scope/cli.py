"""``nikon-control-scope`` — bring up and inspect the microscope.

Subcommands follow the order a rig is actually brought up:

    nikon-control-scope adapters                 what this machine can drive
    nikon-control-scope devices NikonTi2         what one adapter offers
    nikon-control-scope probe NikonTi2 TIXYDrive try to connect one device
    nikon-control-scope config MMConfig.cfg      inspect an existing config
    nikon-control-scope plate ...                register a plate on the stage
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import discover
from .plate import WellRef, calibrate, suggested_refs


def _cmd_adapters(args) -> int:
    install = discover.mm_install()
    print(f"Micro-Manager: {install or 'NOT FOUND'}")
    core = discover.new_core()
    adapters = discover.available_adapters(core)
    print(f"{len(adapters)} device adapter(s) installed\n")
    for a in adapters:
        mark = ""
        if a in discover.NIKON_ADAPTERS:
            mark = f"   <- {discover.NIKON_ADAPTERS[a]}"
        elif a in discover.CAMERA_ADAPTERS:
            mark = f"   <- {discover.CAMERA_ADAPTERS[a]}"
        print(f"  {a}{mark}")
    print("\nNikon readiness:")
    for note in discover.nikon_readiness(core):
        print(f"  - {note}")
    return 0


def _cmd_devices(args) -> int:
    entries = discover.adapter_devices(args.adapter)
    if not entries:
        print(f"no devices found for adapter {args.adapter!r} "
              "(is it installed? run: nikon-control-scope adapters)")
        return 1
    print(f"{args.adapter}: {len(entries)} device(s)\n")
    width = max(len(e.name) for e in entries)
    for e in entries:
        print(f"  {e.name:<{width}}  {e.type:<14} {e.description}")
    return 0


def _cmd_probe(args) -> int:
    targets = ([(args.adapter, args.device)] if args.device else
               [(args.adapter, e.name)
                for e in discover.adapter_devices(args.adapter)])
    if not targets:
        print(f"nothing to probe for {args.adapter!r}")
        return 1
    failures = 0
    for lib, name in targets:
        res = discover.probe(lib, name, read_properties=args.properties)
        print(res.describe())
        if not res.ok:
            failures += 1
        elif args.properties and res.properties:
            for k, v in sorted(res.properties.items()):
                print(f"     {k} = {v}")
    print(f"\n{len(targets) - failures}/{len(targets)} connected")
    return 1 if failures and args.device else 0


def _cmd_config(args) -> int:
    core = discover.new_core()
    try:
        discover.load_config(args.path, core)
    except Exception as exc:
        print(f"could not load {args.path}: {exc}")
        return 1
    devices = discover.loaded_devices(core)
    print(f"{args.path}: {len(devices)} device(s)\n")
    w = max((len(d.label) for d in devices), default=5)
    for d in devices:
        role = f"  [{d.role}]" if d.role else ""
        print(f"  {d.label:<{w}}  {d.type:<14} {d.library}/{d.name}{role}")
    roles = discover.core_roles(core)
    missing = [r for r, v in roles.items() if not v]
    print("\ncore roles:")
    for r, v in roles.items():
        print(f"  {r:<10} {v or '— not set —'}")
    if missing:
        print(f"\nnot configured: {', '.join(missing)}")
    return 0


def _cmd_plate(args) -> int:
    if args.suggest:
        a, b, c = suggested_refs(args.plate)
        print(f"For a {args.plate} plate, centre these three wells and record "
              f"the stage position of each:\n  {a}   {b}   {c}\n"
              "Corners are worth the travel — adjacent wells barely constrain "
              "the rotation.\nThen:\n"
              f"  nikon-control-scope plate --plate {args.plate} "
              f"--well {a} X Y --well {b} X Y --well {c} X Y")
        return 0
    if not args.well:
        print("give at least one --well NAME X Y (or --suggest)")
        return 1
    refs = [WellRef(name=w[0], x=float(w[1]), y=float(w[2])) for w in args.well]
    try:
        cal = calibrate(args.plate, refs)
    except ValueError as exc:
        print(f"calibration refused: {exc}")
        return 1
    print(cal.describe())
    if not cal.rotation_estimated:
        print("  note: one well only — rotation assumed 0. Add a far corner "
              "to measure it.")
    if cal.residual_um > 50:
        print(f"  warning: residual {cal.residual_um:.0f} µm is large for "
              "hand-centred wells; re-check the well identities.")
    if args.json:
        Path(args.json).write_text(json.dumps({
            "plate": cal.plate,
            "a1_center_xy": list(cal.a1_center_xy),
            "rotation": cal.rotation,
            "residual_um": cal.residual_um,
            "scale_error": cal.scale_error,
            "n_refs": cal.n_refs,
        }, indent=2))
        print(f"  written to {args.json}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(
        prog="nikon-control-scope",
        description="Bring up and inspect the microscope through Micro-Manager.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("adapters", help="list installed device adapters").set_defaults(
        func=_cmd_adapters)

    d = sub.add_parser("devices", help="list the devices one adapter offers")
    d.add_argument("adapter")
    d.set_defaults(func=_cmd_devices)

    pr = sub.add_parser("probe", help="load and initialise a device to see if "
                                      "it is really connected")
    pr.add_argument("adapter")
    pr.add_argument("device", nargs="?", help="omit to probe every device "
                                              "the adapter offers")
    pr.add_argument("--properties", action="store_true",
                    help="print the device's properties when it connects")
    pr.set_defaults(func=_cmd_probe)

    c = sub.add_parser("config", help="inspect a Micro-Manager .cfg")
    c.add_argument("path")
    c.set_defaults(func=_cmd_config)

    pl = sub.add_parser("plate", help="register a well plate against the stage")
    pl.add_argument("--plate", default="96-well",
                    help="plate type (default 96-well)")
    pl.add_argument("--well", nargs=3, action="append", metavar=("NAME", "X", "Y"),
                    help="a well you centred, and the stage position there; "
                         "repeatable")
    pl.add_argument("--suggest", action="store_true",
                    help="print which wells to use and stop")
    pl.add_argument("--json", help="write the calibration to this file")
    pl.set_defaults(func=_cmd_plate)

    args = p.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
