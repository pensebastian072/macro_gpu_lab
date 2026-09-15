# Daily worker: refresh panel + post the surprise-shift signal to the webhook.
# ASCII-only (PowerShell 5.1 cp1252). Called by _run_shift_export.vbs (hidden).
$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
$py = Join-Path $repo ".venv\Scripts\python.exe"
# The task launches from system32 -- put the package on the path so `-m` resolves.
Set-Location $repo
$env:PYTHONPATH = $repo

$runs = Join-Path $repo "journal\runs"
New-Item -ItemType Directory -Force -Path $runs | Out-Null
$log = Join-Path $runs ("shift_export_" + (Get-Date -Format "yyyyMMdd") + ".log")

# Load the webhook secret from a local gitignored file if the env var is unset.
# Keeps the token out of the Task Scheduler definition and out of git.
if (-not $env:MACRO_GPU_WEBHOOK_URL) {
    $secretFile = Join-Path $repo "secrets\webhook_url.txt"
    if (Test-Path $secretFile) {
        $env:MACRO_GPU_WEBHOOK_URL = (Get-Content $secretFile -Raw).Trim()
    }
}

# PREFLIGHT: with $py missing, the `cmd /c ... >> log 2>&1` calls below just append
# "The system cannot find the path specified." and keep going, and ErrorActionPreference
# is Continue - so this task reported result=0 while doing nothing every night from
# 2026-09-01 to 09-08 (repo moved to D: without its venv rebuilt). Fail loudly instead.
if (-not (Test-Path $py)) {
    "[{0}] FATAL: interpreter missing at {1} - venv not built. Rebuild with perf_probe\migrationebuild_torch_venvs.ps1" -f (Get-Date -Format s), $py |
        Out-File -FilePath $log -Append -Encoding utf8
    exit 1
}

"[{0}] run start" -f (Get-Date -Format s) | Out-File -FilePath $log -Append -Encoding utf8
# Run through cmd so >> / 2>&1 are native (avoids PowerShell wrapping python's
# stderr in NativeCommandError records).
& cmd.exe /c "`"$py`" -m macro_gpu_lab.data >> `"$log`" 2>&1"
& cmd.exe /c "`"$py`" -m macro_gpu_lab.models.shift_export --post >> `"$log`" 2>&1"
"[{0}] run done" -f (Get-Date -Format s) | Out-File -FilePath $log -Append -Encoding utf8
