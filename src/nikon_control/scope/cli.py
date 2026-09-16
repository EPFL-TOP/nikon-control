"""``nikon-control-scope`` — bring up and inspect the microscope.

Subcommands follow the order a rig is actually brought up:

    nikon-control-scope adapters                 what this machine can drive
    nikon-control-scope stand                    which Nikon stand is here
    nikon-control-scope devices NikonTi2         what one adapter offers
    nikon-control-scope probe NikonTi2 TIXYDrive try to connect one device
    nikon-control-scope build --out MMConfig.cfg  write a config for what
                                                  is really attached
    nikon-control-scope config MMConfig.cfg      inspect an existing config
    nikon-control-scope channel --name BF        capture the current
                                                 illumination as a channel
    nikon-control-scope plate ...                register a plate on the stage

Both stand generations are supported: the Ti2-E via ``NikonTi2`` and the
older Ti-E via ``NikonTI``. Nothing above :mod:`.stand` names a device
directly — the commands report *roles*, so the same code serves both.
"""
from __future__ import annotations

import argparse
import difflib
from pathlib import Path

from . import (channels as channels_mod, config_build, discover,
               plate as plate_mod, stand as stand_mod)
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


def _print_roles(roles: dict[str, str], devices=None) -> None:
    print("\n  roles:")
    for line in stand_mod.describe_roles(roles):
        print(f"    {line}")
    if devices is None:
        return
    # A role with several candidates is where a silent wrong pick hides —
    # four devices on a Ti2 are typed XYStage and only one is the stage.
    ambiguous = stand_mod.ambiguous_roles(devices)
    if ambiguous:
        print("\n  chosen from several candidates "
              "(override in the .cfg if wrong):")
        for role, names in ambiguous.items():
            chosen = roles.get(role, "")
            others = ", ".join(n for n in names if n != chosen)
            print(f"    {stand_mod.ROLE_LABELS[role]:<18} {chosen}"
                  f"   (also: {others})")


def _cmd_devices(args) -> int:
    core = discover.new_core()
    scan = discover.scan_adapter(args.adapter, core)
    if not scan.installed:
        near = difflib.get_close_matches(
            args.adapter, discover.available_adapters(core), n=3, cutoff=0.6)
        print(f"no adapter named {args.adapter!r} is installed.")
        if near:
            print(f"did you mean: {', '.join(near)}?")
        print("run: nikon-control-scope adapters")
        return 1

    if not scan.devices:
        # Installed but empty is a real diagnosis, not "not found" — and what
        # it means depends on the stand.
        print(f"{args.adapter}: installed, but offers no devices.")
        st = stand_mod.stand_for_adapter(args.adapter)
        if st:
            print(f"  {st.empty_list_meaning}")
            mm = discover.mm_install() or "the Micro-Manager folder"
            found = discover.find_driver(st)
            if not found.satisfied:
                st_status = discover.StandStatus(stand=st, installed=True,
                                                 driver=found)
                for line in st_status.diagnosis(mm)[1:]:
                    print(f"  {line}")
        if scan.error:
            print(f"  error: {scan.error}")
        return 1

    print(f"{args.adapter}: {len(scan.devices)} device(s)\n")
    width = max(len(e.name) for e in scan.devices)
    for e in scan.devices:
        print(f"  {e.name:<{width}}  {e.type:<14} {e.description}")
    st = stand_mod.stand_for_adapter(args.adapter)
    if st:
        _print_roles(stand_mod.resolve_roles(scan.devices), scan.devices)
        if not st.dynamic:
            print(f"\n  {st.empty_list_meaning}")
    return 0


def _cmd_stand(args) -> int:
    core = discover.new_core()
    mm = discover.mm_install()
    print(f"Micro-Manager: {mm or 'NOT FOUND'}\n")
    statuses = discover.stand_status(core, mm)
    usable = 0
    for st in statuses:
        head = f"{st.stand.adapter}  ({st.stand.label})"
        print(head)
        print("-" * len(head))
        for line in st.diagnosis(mm):
            print(f"  {line}")
        if st.devices:
            _print_roles(st.roles, st.devices)
        if st.usable:
            usable += 1
        if args.deep and st.installed and not st.devices:
            print("\n  deep check (asking Windows why the adapter will not "
                  "load):")
            for line in discover.deep_check(st.stand, mm):
                print(f"    {line}")
        if args.notes:
            print("\n  notes:")
            for n in st.stand.notes:
                print(f"    - {n}")
        print()

    if args.notes:
        print("both stands:")
        for n in discover.SHARED_NOTES:
            print(f"  - {n}")
        print()

    ready = [s for s in statuses if s.usable]
    if ready:
        s = ready[0]
        xy = s.roles.get("xystage", "<xy stage>")
        print(f"Next: probe one device to prove the stand is really talking:\n"
              f"  nikon-control-scope probe {s.stand.adapter} {xy} --properties")
    else:
        print("No stand is ready yet. Fix the driver findings above, then "
              "re-run this command.")
    return 0 if usable else 1


def _cmd_fix_driver(args) -> int:
    st = stand_mod.STANDS_BY_KEY[args.stand]
    mm = discover.mm_install()
    ok, msg = discover.install_driver(st, mm, args.source)
    print(msg)
    if not ok:
        return 1
    print("\nre-checking:")
    for line in discover.deep_check(st, mm):
        print(f"  {line}")
    print("\nthen: nikon-control-scope stand")
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


def _cmd_build(args) -> int:
    mm = discover.mm_install()
    core = discover.new_core()
    print(f"Micro-Manager: {mm or 'NOT FOUND'}")
    print("Connecting devices one at a time — this powers up the stand and "
          "may take a moment.\n")

    result = config_build.build(
        core,
        stand_adapter=args.adapter,
        camera_adapter=args.camera_adapter,
        camera_device=args.camera_device,
        skip=set(args.skip or ()),
    )
    for line in result.summary():
        print(line)
    for note in result.notes:
        print(f"\n! {note}")

    missing = stand_mod.missing_roles(result.roles)
    if result.devices:
        from .discover import DeviceEntry
        entries = [DeviceEntry(library=d.library, name=d.label, type=d.type)
                   for d in result.devices]
        _print_roles(result.roles, entries)
    if missing:
        print(f"\n  not resolved: {', '.join(missing)}")

    if not result.devices:
        gui = discover.gui_launcher(mm)
        if gui:
            print(f"\nNothing connected. This install does have the "
                  f"Micro-Manager GUI ({gui}); its Hardware Configuration "
                  f"Wizard is the fallback when a device needs pre-init "
                  f"properties this command cannot guess.")
        return 1

    text = config_build.to_text(result, core)
    if args.dry_run:
        print("\n--- would write ---")
        print(text)
        return 0

    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"\n{out} already exists — pass --force to overwrite.")
        return 1
    out.write_text(text)
    print(f"\nwrote {out}")
    print(f"check it with:  nikon-control-scope config {out}")
    print(f"then connect the /scope dashboard to it, or:\n"
          f"  nikon-control-dashboard --mm-config {out.resolve()}")
    return 0


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

    # Core roles cover camera/XY/focus/autofocus/shutter; the stand roles add
    # the ones MMCore has no slot for (PFS offset, nosepiece, light path) and
    # are labelled the same way on both stand generations.
    labelled = [discover.DeviceEntry(library=d.library, name=d.label,
                                     type=d.type, description=d.description)
                for d in devices]
    _print_roles(stand_mod.resolve_roles(labelled), labelled)

    groups = channels_mod.groups(_scope_from(core))
    if groups:
        print("\nconfig groups (channels):")
        for g, presets in groups.items():
            print(f"  {g}: {', '.join(presets) or '(no presets)'}")
    else:
        print("\nno config groups defined — see `nikon-control-scope channel`")

    if args.properties:
        scope = _scope_from(core)
        print("\nproperties:")
        for d in devices:
            props = scope.properties(d.label)
            if not props:
                continue
            print(f"\n  {d.label}")
            for info in props:
                print(f"    {info.describe()}")
    return 0


def _scope_from(core):
    from .control import Scope

    return Scope(core)


def _cmd_channel(args) -> int:
    from .control import Scope

    if not channels_mod.valid_name(args.name or ""):
        if not args.list:
            print("give a --name for the preset (no commas)")
            return 1
    try:
        scope = Scope.from_config(args.config)
    except Exception as exc:
        print(f"could not load {args.config}: {exc}")
        return 1

    existing = channels_mod.read_presets(args.config, args.group)
    if args.list:
        if not existing:
            print(f"no presets in group {args.group!r} in {args.config}")
            return 0
        print(f"group {args.group!r}:")
        for preset in existing.values():
            print(f"  {preset.describe()}")
        return 0

    preset = channels_mod.capture(scope, args.name,
                                  with_exposure=args.with_exposure)
    if not preset.settings:
        print("nothing to capture — this configuration has no shutters, "
              "filter turrets or light path devices.")
        return 1
    print(f"captured from the microscope's current state:\n  "
          f"{preset.describe()}")
    if args.dry_run:
        print("\n--- would add ---")
        print("\n".join(preset.lines(args.group)))
        return 0
    channels_mod.append_to_config(args.config, preset, args.group)
    verb = "updated" if args.name in existing else "added"
    print(f"\n{verb} preset {args.name!r} in group {args.group!r} "
          f"-> {args.config}")
    print("reload the dashboard to see it in the Channel menu")
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
    if args.show:
        try:
            cal = plate_mod.load(args.show)
        except Exception as exc:
            print(f"could not read {args.show}: {exc}")
            return 1
        print(cal.describe())
        lay = plate_mod.layout(cal)
        print(f"  {len(lay.names)} wells, {lay.rows}x{lay.columns}, "
              f"{lay.well_width_um / 1000:.1f} mm each")
        print(f"  A1 {lay.x[0]:.0f}, {lay.y[0]:.0f} µm   "
              f"{lay.names[-1]} {lay.x[-1]:.0f}, {lay.y[-1]:.0f} µm")
        return 0
    if not args.well:
        print("give at least one --well NAME X Y (or --suggest, or --show FILE)")
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
        plate_mod.save(cal, args.json)
        print(f"  written to {args.json} — the /scope dashboard reads this "
              f"file, and so does `plate --show`")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(
        prog="nikon-control-scope",
        description="Bring up and inspect the microscope through Micro-Manager.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("adapters", help="list installed device adapters").set_defaults(
        func=_cmd_adapters)

    st = sub.add_parser("stand", help="which Nikon stand (Ti2 or Ti) this "
                                      "machine can drive, and its roles")
    st.add_argument("--notes", action="store_true",
                    help="print each stand's known gotchas too")
    st.add_argument("--deep", action="store_true",
                    help="for a stand whose adapter offers no devices, load "
                         "the DLLs directly and report the operating "
                         "system's own error (Windows only)")
    st.set_defaults(func=_cmd_stand)

    fd = sub.add_parser("fix-driver",
                        help="copy a stand's vendor DLL into the "
                             "Micro-Manager folder, where the adapter looks")
    fd.add_argument("--stand", default="ti2", choices=["ti2", "ti"],
                    help="which stand's driver (default ti2)")
    fd.add_argument("--from", dest="source", default=None,
                    help="copy from this path instead of the vendor default")
    fd.set_defaults(func=_cmd_fix_driver)

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

    b = sub.add_parser("build", help="write a Micro-Manager .cfg containing "
                                     "the devices that actually connect")
    b.add_argument("--out", default="MMConfig.cfg",
                   help="file to write (default MMConfig.cfg)")
    b.add_argument("--adapter", default=None,
                   help="stand adapter to build from (default: the installed "
                        "Nikon one)")
    b.add_argument("--camera-adapter", default=None,
                   help="camera adapter, e.g. HamamatsuHam / PVCAM / "
                        "AndorSDK3 (default: try each installed one)")
    b.add_argument("--camera-device", default=None,
                   help="specific camera device name within that adapter")
    b.add_argument("--skip", action="append", default=[],
                   help="device name to leave out; repeatable")
    b.add_argument("--dry-run", action="store_true",
                   help="print the configuration instead of writing it")
    b.add_argument("--force", action="store_true",
                   help="overwrite an existing file")
    b.set_defaults(func=_cmd_build)

    c = sub.add_parser("config", help="inspect a Micro-Manager .cfg")
    c.add_argument("path")
    c.add_argument("--properties", action="store_true",
                   help="also dump every device property, with its limits "
                        "and allowed values — this is how to find the real "
                        "name of e.g. the lamp intensity")
    c.set_defaults(func=_cmd_config)

    ch = sub.add_parser("channel",
                        help="capture the microscope's current illumination "
                             "as a named channel preset")
    ch.add_argument("--config", default="MMConfig.cfg",
                    help="the .cfg to read and write (default MMConfig.cfg)")
    ch.add_argument("--name", help="preset name, e.g. BF or GFP")
    ch.add_argument("--group", default=channels_mod.DEFAULT_GROUP,
                    help=f"config group (default {channels_mod.DEFAULT_GROUP})")
    ch.add_argument("--with-exposure", action="store_true",
                    help="capture the camera exposure into the preset too")
    ch.add_argument("--list", action="store_true",
                    help="list the presets already defined and stop")
    ch.add_argument("--dry-run", action="store_true",
                    help="print the lines instead of writing them")
    ch.set_defaults(func=_cmd_channel)

    pl = sub.add_parser("plate", help="register a well plate against the stage")
    pl.add_argument("--plate", default="96-well",
                    help="plate type (default 96-well)")
    pl.add_argument("--well", nargs=3, action="append", metavar=("NAME", "X", "Y"),
                    help="a well you centred, and the stage position there; "
                         "repeatable")
    pl.add_argument("--suggest", action="store_true",
                    help="print which wells to use and stop")
    pl.add_argument("--json", help="write the calibration to this file")
    pl.add_argument("--show", metavar="FILE",
                    help="print a saved calibration and the wells it implies")
    pl.set_defaults(func=_cmd_plate)

    args = p.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
