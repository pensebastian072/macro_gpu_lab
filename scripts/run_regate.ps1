# Weekly worker: RE-GATE every model on fresh data.
#
# The daily MacroGpuLab-ShiftExport task refreshes the panel and posts the signal,
# but it never re-runs the gate. Consequence, caught on 2026-07-30: this repo's
# scorecards all carried a single date, 2026-07-04, so a 26-day-old FAIL read exactly
# like a current one, and the published daily signal had never been graded at all.
# This task closes that gap.
#
# ASCII-only (PowerShell 5.1 cp1252). Called by _run_regate.vbs (hidden).
$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
$py = Join-Path $repo ".venv\Scripts\python.exe"
# The task launches from system32 -- put the package on the path so `-m` resolves.
Set-Location $repo
$env:PYTHONPATH = $repo

$runs = Join-Path $repo "journal\runs"
New-Item -ItemType Directory -Force -Path $runs | Out-Null
$log = Join-Path $runs ("regate_" + (Get-Date -Format "yyyyMMdd") + ".log")

# Deliberately NO webhook secret is loaded here. Re-gating writes scorecards and the
# shadow flag; it must never post anything.

"[{0}] regate start" -f (Get-Date -Format s) | Out-File -FilePath $log -Append -Encoding utf8

# Run through cmd so >> / 2>&1 are native (avoids PowerShell wrapping python's
# stderr in NativeCommandError records).
# --rebuild refetches the panel first so the gate sees current data, not a stale one.
& cmd.exe /c "`"$py`" -m macro_gpu_lab.train --once --rebuild >> `"$log`" 2>&1"
& cmd.exe /c "`"$py`" -m macro_gpu_lab.models.shift_detector >> `"$log`" 2>&1"
& cmd.exe /c "`"$py`" -m macro_gpu_lab.models.shift_detector --surprise >> `"$log`" 2>&1"

"[{0}] regate done" -f (Get-Date -Format s) | Out-File -FilePath $log -Append -Encoding utf8
