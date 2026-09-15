# Register a daily Task Scheduler job that posts the surprise-shift signal.
# Runs as the current user at normal (Limited) run level -- the export needs no
# admin -- so registration needs no UAC. ASCII-only. Runs wscript ->
# _run_shift_export.vbs (hidden).
#
#   powershell -ExecutionPolicy Bypass -File scripts\register_daily_export.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\register_daily_export.ps1 -At 17:30
#   powershell -ExecutionPolicy Bypass -File scripts\register_daily_export.ps1 -Unregister
param(
    [string]$At = "17:30",          # local time, after the US cash close
    [switch]$Unregister
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$taskName = "MacroGpuLab-ShiftExport"

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Output "unregistered $taskName"
    return
}

$vbs = Join-Path $repo "scripts\_run_shift_export.vbs"
if (-not (Test-Path $vbs)) { throw "missing $vbs" }

$action   = New-ScheduledTaskAction -Execute "wscript.exe" -Argument "`"$vbs`""
$trigger  = New-ScheduledTaskTrigger -Daily -At $At
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
                -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 20)

# No -Principal: runs as the registering user, Limited run level, only when
# logged on -- no elevation required.
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Settings $settings -Force | Out-Null
Write-Output "registered '$taskName' daily at $At"
Write-Output "run now to test: Start-ScheduledTask -TaskName $taskName"
Write-Output "log: journal\runs\shift_export_<YYYYMMDD>.log"
