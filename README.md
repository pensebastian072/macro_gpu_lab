# macro_gpu_lab

<!-- one-tap-install -->
[![Download ZIP](https://img.shields.io/badge/Download-ZIP-2ea44f?style=for-the-badge&logo=github)](https://github.com/pensebastian072/macro_gpu_lab/archive/refs/heads/main.zip)

**Run it on your computer in 3 steps:** 1) [download the ZIP](https://github.com/pensebastian072/macro_gpu_lab/archive/refs/heads/main.zip) · 2) unzip it · 3) double-click **`install.bat`** (Windows) or run **`./install.sh`** (macOS/Linux).
The dashboard opens in your browser at `http://127.0.0.1:8102` - it runs only on your machine. Next time use `start.bat` / `./start.sh`.
For the full research stack (large downloads) use `install.bat --full` / `./install.sh --full`.
<!-- one-tap-install -->

GPU research framework over the HQ macro brain. Assembles a multi-year daily
panel for the 22-asset macro universe, engineers cross-asset relationship
features, trains models to predict forward returns (and later to detect
relationship/regime shifts), and validates every model through the canonical
overfit gate (PBO / Deflated Sharpe / purged CV). Output is an advisory shadow
flag-file — nothing here touches a broker or a live hot path.

## Rules (non-negotiable)

- Research / advisory / paper only. No live execution, ever.
- A model stays SHADOW until it clears the gate: PBO < 0.5 AND Deflated Sharpe
  > 0 on **every** horizon. Promotion is an env flip, never automatic.
- Output is a fail-safe flag-file. Any HQ reader is off hot path, neutral on
  stale/missing.
- A correct gate usually FAILS the first model. That is the honest result.

## Setup

```powershell
pwsh scripts/setup_venv.ps1
```

Then use `.venv\Scripts\python.exe` explicitly (PATH `python` is the broken
Store stub on this box).

## Run

```powershell
.venv\Scripts\python.exe -m macro_gpu_lab.gpu      # CUDA smoke test
.venv\Scripts\python.exe -m macro_gpu_lab.data     # build the daily panel
.venv\Scripts\python.exe -m macro_gpu_lab.train --once
.venv\Scripts\python.exe -m macro_gpu_lab.models.shift_detector --surprise   # calm-eruption research
```

### Viewer

```bash
pip install flask
python -m ui.app        # http://127.0.0.1:8102
```

A read-only page over the gate verdicts and the per-asset leans: 22 assets grouped
by the macro force each maps to, with `p_up` at 5d and 21d, direction and
conviction.

**It works on a fresh clone.** The repo ships a committed snapshot of the flag at
`ui/snapshot/macro_gpu_state.json`, so the real output renders with no market
data, API key or GPU. If you have run the bench, the viewer prefers your live
`journal/flags/macro_gpu_state.json`. The header says which, and how old it is.

Staleness is **recomputed from `as_of`**, not read from the `stale` field in the
file — that field records what was true when the flag was written, which is a
different question from whether it is old now. The shipped snapshot is a month
old and the page says so.

The leans are display only. Neither model has cleared the gate, so nothing here
sizes, gates or vetoes anything. Binds `127.0.0.1` only, no POST route.

### Surprise-shift signal -> webhook (advisory, exact RF)

```powershell
# set once (notification sink only -- Discord/Slack/generic JSON; NEVER a broker)
$env:MACRO_GPU_WEBHOOK_URL = "https://discord.com/api/webhooks/..."
.venv\Scripts\python.exe -m macro_gpu_lab.models.shift_export --post
```

Writes `journal/flags/tv_signal.json` (authoritative live signal) and POSTs it to
the webhook. Posts only when a calm-state eruption setup is live or the signal
changed (`--always` to force). The exact RF signal goes over the webhook; the
Pine study is a fallback approximation. SHADOW/unvalidated (DSR<0), ~2016+ regime
-- paper/manual use.

### Schedule it daily (Task Scheduler)

```powershell
# 1. store the webhook token OUT of git (secrets/ is gitignored):
#    create  secrets\webhook_url.txt  with just the URL on one line
# 2. register the elevated daily job (UAC prompt; runs hidden via wscript):
powershell -ExecutionPolicy Bypass -File scripts\register_daily_export.ps1 -At 17:30
#    remove it with:  ... register_daily_export.ps1 -Unregister
```

The job runs `scripts\_run_shift_export.vbs` -> `run_shift_export.ps1` which
refreshes the panel, posts the signal, and logs to `journal\runs\`. It launches
`python.exe` hidden (NOT `pythonw.exe` -- Norton blocks that in the venv).

## Reuse map

- Overfit gate ported from `copper_brain/copper_brain/validate.py`.
- RF baseline + non-overlapping OOS pattern from `copper_brain/copper_brain/model.py`.
- Feature math mirrors `hq-trading-system/analytics/macro_engine.py`.
- Flag-file fail-safe contract from `copper_brain/copper_brain/publish.py`.
- GPU/venv scaffold mirrors `Kronos`.
