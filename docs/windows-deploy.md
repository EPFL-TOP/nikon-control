# Deploying the annotation tool on a Windows server

Target setup: a Windows machine that several lab members RDP into to run
the annotation tool against ND2 files on a shared drive.

> The current tool is the **Bokeh dashboard** (`nikon-control-dashboard`).
> The napari `nikon-control-annotate` tool below is retired/legacy. Jump to
> [Running the dashboard](#running-the-dashboard) if it's already installed.

## Clean install from scratch (new server)

Follow these top to bottom on a fresh Windows machine. It uses **Miniconda**
(the same env used for the GPU torch build) and installs the current Bokeh
dashboard. Do it once, as an admin; annotators then just open a browser.

Throughout, the conda env is called `single-cells` and the code lives in
`C:\Tools\nikon-control`. Change these if you like, but keep them consistent.

### 1. Install the prerequisites

Install these first (all "for all users" so every RDP account can use them):

1. **Miniconda — for all users.** Download from
   <https://www.anaconda.com/download/success> (Miniconda). During setup pick
   **"Install for all users"** so it lands in `C:\ProgramData\miniconda3\`,
   **not** `C:\Users\<you>\AppData\Local\` — a per-user install under
   `AppData\Local` is unreadable by other Windows accounts and breaks
   multi-user use.
2. **Git for Windows** — <https://git-scm.com/download/win>.
3. **Microsoft Visual C++ Redistributable x64** —
   <https://aka.ms/vs/17/release/vc_redist.x64.exe>, then reboot. Required by
   torch (`c10.dll`) and the ND2 reader; installing it now avoids the
   `WinError 1114` DLL failure later.
4. **NVIDIA driver** (only if the box has an NVIDIA GPU) — a recent
   Studio/Game-Ready driver from <https://www.nvidia.com/download/index.aspx>.
   Verify with `nvidia-smi` in a terminal. No CUDA *toolkit* is needed — the
   torch wheel bundles its own CUDA runtime.

Open a fresh **Anaconda Prompt** (Start menu) for the rest so `conda` is on
PATH.

### 2. Get the code

```bat
cd C:\Tools
git clone <your-fork-or-mirror-url> nikon-control
cd nikon-control
```

### 3. Create the environment and install the tool

```bat
conda create -y -n single-cells python=3.11
conda activate single-cells
python -m pip install --upgrade pip
pip install -e ".[dashboard]"
```

Python **3.11** is recommended (3.12 is fine; avoid 3.13 on Windows).

### 4. Enable the GPU (recommended — skip if CPU-only)

`pip install` pulls the **CPU-only** torch on Windows, so detection would run
slowly. Replace it with a CUDA wheel matching your driver:

```bat
conda activate single-cells
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

That last line must print `True <your GPU>`. Use `cu124` for very new drivers
or `cu118` for older ones. (More detail + troubleshooting in
[GPU acceleration](#gpu-acceleration-make-detection-fast).)

### 5. Put the detection model in place

Copy `cell_detection_model.pth` to a readable location, e.g.
`C:\Tools\nikon-control\cell_detection_model.pth` or a shared drive. You'll
point `--weights` at it (or drop it in the data folder for auto-discovery).

### 6. First run and smoke test

```bat
conda activate single-cells
nikon-control-dashboard --data-dir "G:\PROJECTS-02\Samuel" ^
    --weights "C:\Tools\nikon-control\cell_detection_model.pth" --show
```

A browser opens at <http://localhost:5006>. Pick an ND2 from the dropdown,
scrub frames, then click **Detect cells** — the status line should say
"Detected N tracks on **CUDA**". If it says CPU, revisit step 4; if detection
errors, see the troubleshooting below.

If the `nikon-control-dashboard` command isn't found, the module form always
works: `python -m nikon_control.dashboard.launch --show`.

### 7. Make it available to everyone, always-on

For multiple annotators and survival across logoffs/reboots, run it as a
Windows service and (for off-server users) bind it to the network — see
[Multiple users on the Windows Server](#multiple-users-on-the-windows-server).
That section's NSSM command is the recommended way to keep it running.

You're done. Day-to-day, annotators just open the URL — no terminal needed.

---

## Running the dashboard

In a conda env (or the venv), with the package installed
(`pip install -e ".[dashboard]"`):

```bat
conda activate single-cells
nikon-control-dashboard --data-dir "Z:\experiments\2026-07" ^
    --weights "C:\Tools\nikon-control\cell_detection_model.pth" --show
```

- `--data-dir` — folder of `.nd2` files (their `.annotations.json` sidecars
  are written alongside). Pick the file from the dropdown in the browser.
- `--weights` — path to `cell_detection_model.pth` (enables the in-app
  *Detect cells* button; also auto-discovered if a `.pth` sits in the data
  folder).
- `--show` — opens a browser tab automatically.

**Two dashboards are served by the one process**, sharing the same viewer
and file browser:

| URL | Dashboard | Use it for |
|---|---|---|
| <http://localhost:5006/annotate> | **Full** — tracked annotations: keyframes, birth/end/deaths, class changes over time | following individual cells through a movie |
| <http://localhost:5006/simple> | **Simple** — per-frame boxes, first N frames, 3 classes (single/doublet/debris) | building detector **training data** |
| <http://localhost:5006/review> | **Review** — steps through an exported dataset and fixes labels | QC before training |

<http://localhost:5006/> lists all three. `--dataset <folder>` makes
`/review` open an exported dataset at startup. The simple one is the one to point
annotators at for training data — it has no tracking or lifecycle controls,
and writes a separate `<file>.simple.json` sidecar that cannot be confused
with the full dashboard's `<file>.annotations.json`.

`--data-dir` and `--weights` only set the *starting* folder/model —
annotators without terminal access can navigate to any folder, ND2, and
`.pth` from the in-page file browser: a **Drive / volume** dropdown to
switch between drive letters (C:, E:, G:, network mounts), plus the Folder
field + Up / Refresh / Subfolders dropdown. So a fixed launch like
`nikon-control-dashboard --show` is enough; users browse from there.

Site defaults are baked in (edit `_DEFAULT_DATA_DIRS` / `_DEFAULT_WEIGHTS`
in `src/nikon_control/dashboard/app.py`): the browser starts in
`G:\PROJECTS-02\Samuel` when the launch folder has no ND2s, and the model
defaults to `E:\PROJECTS-01\Clement\cell_detection_model.pth`.

If the `nikon-control-dashboard` command isn't found (console script not on
PATH), the module form always works:

```bat
python -m nikon_control.dashboard.launch --data-dir "Z:\experiments\2026-07" --show
```

**Reaching it from another machine** (not the server console): add the
server's hostname so Bokeh accepts the websocket, and browse to it:

```bat
nikon-control-dashboard --data-dir "Z:\exp" --port 5006 ^
    --allow-websocket-origin myserver.epfl.ch:5006
```

then open `http://myserver.epfl.ch:5006` in a browser. Multiple users can
each open the URL — every browser tab is its own independent session.

### GPU acceleration (make detection fast)

The dashboard shows the device it used, e.g. *"Detected N tracks on
**CUDA**"* or *"…on **CPU** (slow)…"* in the status line after `Detect
cells`. If it says CPU, detection is running on the processor — on Windows
`pip install torch` installs the **CPU-only** build by default, so the
NVIDIA GPU is never used.

To install the CUDA build (in the same conda env), replace torch with a
CUDA wheel:

```bat
conda activate single-cells
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

(Use `cu124` instead of `cu121` for very new drivers; `cu118` for older
ones — any that your driver supports.) Verify:

```bat
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

It should print `True <your GPU>`. Then the dashboard's Detect will report
**CUDA** and run ~10–50 ms/frame instead of ~0.5–0.7 s/frame. Debris
detection is CPU-based (classical image processing) and unaffected.

#### If detection throws a CUDA error at runtime

Symptom: *"CUDA error: unknown error … kernel errors might be asynchronously
reported…"*. The dashboard now **falls back to CPU automatically** (you'll
see "⚠ GPU error — retrying on CPU"), so annotation isn't blocked — but the
GPU is worth fixing. Checks, in order:

1. **Is the server running as a Windows service (NSSM)?** Services run in
   session 0, which often has **no GPU access**. Either run the dashboard
   from a normal (interactive) login of an account that can use the GPU, or
   configure the service to run as that account / grant it GPU access. This
   is the most common cause of "unknown error" on a server.
2. `nvidia-smi` in the same environment — confirms the driver sees the GPU
   and it isn't out of memory (another process hogging it).
3. Driver vs CUDA wheel: a very old driver can't run a newer `cuXXX` wheel —
   install a wheel your driver supports (`cu118` is the safest older option).
4. For a real stack trace, set `CUDA_LAUNCH_BLOCKING=1` before launching and
   reproduce; the error then points at the actual failing op.
5. Restart the dashboard after any of the above (the detector is cached per
   session).

#### `WinError 1114` / "c10.dll … initialization routine failed"

torch itself won't load — detection never starts. Causes, in order:

1. **Wrong environment.** If the error path is
   `...\miniconda3\Lib\site-packages\torch` (the conda **base** env) but you
   installed torch into `single-cells`, you launched with the wrong Python.
   `conda activate single-cells` first, then launch — or use that env's
   python explicitly:
   `C:\Users\<you>\AppData\Local\miniconda3\envs\single-cells\python.exe -m nikon_control.dashboard.launch --show`.
   Confirm with `python -c "import torch; print(torch.__file__)"` — the path
   must be inside `envs\single-cells`.
2. **Missing Visual C++ Redistributable.** `c10.dll` depends on the
   VC++ 2015–2022 x64 runtime. Install
   <https://aka.ms/vs/17/release/vc_redist.x64.exe> and reboot.
3. **Dirty torch install** (e.g. CUDA wheel over a CPU wheel). Clean reinstall
   in the right env:

   ```bat
   conda activate single-cells
   pip uninstall -y torch torchvision
   pip cache purge
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```

   That last line must succeed *at the terminal* before the dashboard will
   work — it isolates the problem from anything in this app.

### Multiple users on the Windows Server

The dashboard is a **web server**: you run **one** server process, and every
user connects to it with a **browser** — each browser tab is its own
independent session (its own file, ROIs, and undo state; the model/GPU are
shared). You do *not* run one per user.

**How users connect:**

- Users who **RDP into this same server**: each opens a browser in their RDP
  session to `http://localhost:5006`. `localhost` is per-machine, so all
  sessions on the server reach the one process. No extra flags needed.
- Users on **their own machines** (not RDP): start the server bound to the
  network and allow their origin, then they browse to the server's hostname:

  ```bat
  nikon-control-dashboard --address 0.0.0.0 --port 5006 ^
      --allow-websocket-origin myserver.epfl.ch:5006
  ```

  then they open `http://myserver.epfl.ch:5006`. (Open port 5006 in the
  firewall.)

**Keep it running independent of your login (important).** On Windows,
processes started in an interactive session are **killed when that user logs
off**. So if you launch it from your account and log off, it dies for
everyone. Options, most robust first:

1. **Run it as a Windows service** with [NSSM](https://nssm.cc) so it starts
   at boot and survives any logoff:

   ```bat
   nssm install NikonDashboard "C:\ProgramData\miniconda3\envs\single-cells\python.exe" ^
       "-m" "nikon_control.dashboard.launch" "--data-dir" "G:\PROJECTS-02\Samuel" ^
       "--weights" "E:\PROJECTS-01\Clement\cell_detection_model.pth" "--port" "5006"
   nssm start NikonDashboard
   ```

   (Use the full path to the env's `python.exe`. Set the service to run as an
   account that can read the data/model drives.) Now anyone can use
   `http://localhost:5006` at any time without you being logged in.

2. **Disconnect, don't log off.** If you start it in your RDP session and
   click *Disconnect* (not *Sign out*), the session and its process stay
   alive and other RDP users can still reach `localhost:5006`. Fragile — a
   reboot or accidental sign-out stops it — so prefer the service.

**Concurrency is safe:** two people editing *different* ND2s are fully
independent. If two open the *same* file, the one who saves second is warned
that the file changed on disk and is refused rather than silently
overwriting the other's work (reload, then save). Annotations are written
next to each ND2 as `<file>.annotations.json`.

## Prerequisites (both tools)

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
pip install -e ".[dashboard]"     # the current Bokeh tool + detection
# pip install -e ".[annotate]"    # only if you still need the legacy napari tool
```

(Installing into a conda env instead of a venv is fine — `conda activate
<env>` then the same `pip install -e ".[dashboard]"`.)

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

## Launcher for the legacy napari tool

*(Skip this if you're using the Bokeh dashboard above.)* Most annotators
will not want to touch PowerShell. Create a batch file
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
