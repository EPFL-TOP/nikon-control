# Deploying the annotation tool on a Windows server

Target setup: a Windows machine that several lab members RDP into to run
the annotation tool against ND2 files on a shared drive.

## Prerequisites

- Windows 10 / 11 / Server with RDP enabled.
- Python **3.11** (recommended) or 3.12.
  - **Avoid 3.13** on Windows for now — some of napari's transitive
    dependencies still lag on 3.13 wheels.
  - **Must be installed for all users**, not per-user. A per-user Python
    (default location: `C:\Users\<you>\AppData\Local\...`) is not readable
    by other Windows users due to ACLs on `AppData\Local`, so a venv
    created from it will fail for every other annotator with
    `No Python at '"C:\Users\<you>\AppData\...'`.
  - Install from <https://www.python.org/downloads/windows/> with
    **"Install Python 3.11 for all users"** TICKED, or install Miniconda
    "for all users" so it lands in `C:\ProgramData\miniconda3\` instead of
    `C:\Users\<you>\AppData\`.
- Git for Windows: <https://git-scm.com/download/win>.
- Microsoft Visual C++ Redistributable (usually preinstalled; needed by
  some scientific wheels).

## Install (one-time, by an admin)

Open PowerShell **as the admin who'll own the install** (not necessarily
Administrator — just the account that has write access to the install
location). The example below uses `C:\Tools\nikon-control` but any
all-users-readable folder works.

```powershell
cd C:\Tools
git clone <your-fork-or-mirror-url> nikon-control
cd nikon-control

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[annotate]"
```

If `Activate.ps1` is blocked by execution policy, run once (current user
only):

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Quick smoke test from the same PowerShell window:

```powershell
nikon-control-annotate --help
```

Should print the CLI usage. If it does, napari + nd2 are wired correctly.

## Launcher for end users

Most annotators will not want to touch PowerShell. Create a batch file
`C:\Tools\nikon-control\annotate.bat`:

```bat
@echo off
cd /d "C:\Tools\nikon-control"
call .venv\Scripts\activate.bat
nikon-control-annotate %*
if errorlevel 1 pause
```

Then either:

- **Drag & drop**: put a shortcut to `annotate.bat` on the desktop. A user
  drags an ND2 file onto the shortcut → napari opens with that file
  loaded.
- **Right-click menu**: register `annotate.bat` as the "Open with…" target
  for `.nd2` files.
- **Explicit invocation**: from any Command Prompt:
  `C:\Tools\nikon-control\annotate.bat Z:\path\to\file.nd2`.

## Where annotations land

By default the JSON file is written next to the ND2:

```
Z:\experiments\2026-05-12\positionA.nd2
Z:\experiments\2026-05-12\positionA.annotations.json
```

If the ND2 directory is read-only for the annotator, pass `-o`:

```
annotate.bat Z:\readonly\file.nd2 -o D:\my_annotations\file.annotations.json
```

We may want to grow a separate config for "ND2 root" and "annotations
root" so annotators don't have to specify `-o` every time — flag this if
it becomes friction.

## Per-user vs shared venv

The setup above is **one shared venv** at `C:\Tools\nikon-control\.venv`,
which any user with read access to the folder can run.

- **Pro**: one install to maintain, one launcher path.
- **Con**: any user with write access to that folder can `pip install`
  things into the shared venv. If that's a concern, make the folder
  read-only for the annotators (admin updates only) or give each user
  their own venv at `%USERPROFILE%\nikon-control\.venv`.

For a small lab the shared-venv pattern is fine.

## Updating

When this repo is updated:

```powershell
cd C:\Tools\nikon-control
git pull
.\.venv\Scripts\Activate.ps1
pip install -e ".[annotate]"   # only needed if dependencies changed
```

## Common issues

- **`No Python at '"C:\Users\<owner>\AppData\Local\...\python.exe'`** when
  another user tries to launch: the venv was created from a per-user
  Python install. Re-do prerequisites with a system-wide Python (see
  Prerequisites), then recreate the venv:

  ```powershell
  cd C:\Tools\nikon-control
  Remove-Item -Recurse -Force .venv
  py -3.11 -m venv .venv
  .\.venv\Scripts\Activate.ps1
  python -m pip install --upgrade pip
  pip install -e ".[annotate]"
  ```

- **`Activate.ps1` blocked**: see `Set-ExecutionPolicy` line above.
- **`napari` opens but is black / OpenGL errors**: rare on real GPUs;
  common on Windows Server with no graphics driver. Try
  `set QT_OPENGL=software` before launching napari, or install Mesa's
  software OpenGL DLL alongside the python.exe.
- **`nd2.ND2File` import error about a missing DLL**: install the
  Microsoft Visual C++ Redistributable
  (<https://aka.ms/vs/17/release/vc_redist.x64.exe>) and reboot.
- **ND2 files on a UNC path don't open**: map the share to a drive
  letter (`net use Z: \\server\share`) and use `Z:\...`.
- **High-DPI scaling looks awful**: in the shortcut's
  Properties → Compatibility → "Change high DPI settings…", enable
  "High DPI scaling override" → "System".

## What's deliberately *not* automated yet

- Multi-file batch annotation (a "queue" view in napari). Speak up if
  this becomes a bottleneck.
- Per-annotator identity tracking in the JSON. The `annotator` field is
  empty by default; we can wire it to `%USERNAME%` if you want
  attribution.
- Centralised label aggregation across all annotators. Not needed until
  we start training.
