# Register a WEEKLY Task Scheduler job that re-gates every model on fresh data.
#
# Sibling to MacroGpuLab-ShiftExport, deliberately NOT a modification of it: the
# daily export must keep running daily, and the gate only needs a weekly refresh.
# Runs as the current user at normal (Limited) run level -- like the export, this
# needs no admin, so registration needs no UAC. ASCII-only. Runs wscript ->
# _run_regate.vbs (hidden).
#
# Why this exists: on 2026-07-30 every scorecard in this repo carried a single date,
# 2026-07-04, because the daily task refreshes the panel and posts the signal but
# never re-runs the gate. A 26-day-old FAIL read exactly like a current one.
#
#   powershell -ExecutionPolicy Bypass -File scripts\register_weekly_regate.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\register_weekly_regate.ps1 -At 18:30 -Day Saturday
#   powershell -ExecutionPolicy Bypass -File scripts\register_weekly_regate.ps1 -Unregister
param(
    [string]$At = "18:30",          # local time, well after the daily 17:30 export
    [string]$Day = "Saturday",      # weekend: the GPU is free and no session competes
    [switch]$Unregister
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$taskName = "MacroGpuLab-WeeklyRegate"

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Output "unregistered $taskName"
    return
}

$vbs = Join-Path $repo "scripts\_run_regate.vbs"
if (-not (Test-Path $vbs)) { throw "missing $vbs" }

$action  = New-ScheduledTaskAction -Execute "wscript.exe" -Argument "`"$vbs`""
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $Day -At $At
# 90 minutes: a full re-gate trains RF + torch + both shift variants and refetches
# the panel. The measured run on 2026-07-30 took well under 15 minutes, so this is
# headroom rather than an expectation.
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
                -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 90)

# No -Principal: runs as the registering user, Limited run level, only when
# logged on -- no elevation required.
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Settings $settings -Force | Out-Null
Write-Output "registered '$taskName' weekly on $Day at $At"
Write-Output "run now to test: Start-ScheduledTask -TaskName $taskName"
Write-Output "log: journal\runs\regate_<YYYYMMDD>.log"
