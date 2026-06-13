# ─────────────────────────────────────────────────────────────────────────────
# CryptoTracker Pro — Task Scheduler Setup
# Run this ONCE (as Administrator) to register the daily prediction job.
# After that, Windows will run it automatically every day at 8:00 AM.
# ─────────────────────────────────────────────────────────────────────────────

$taskName   = "CryptoTrackerPro-DailyPrediction"
$scriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$batFile    = Join-Path $scriptDir "run_prediction.bat"

# ── Check the .bat file exists ────────────────────────────────────────────────
if (-not (Test-Path $batFile)) {
    Write-Error "run_prediction.bat not found at: $batFile"
    exit 1
}

# ── Remove existing task if present ──────────────────────────────────────────
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Host "Removed existing task: $taskName"
}

# ── Build trigger: daily at 08:00 AM ─────────────────────────────────────────
$trigger = New-ScheduledTaskTrigger -Daily -At "08:00AM"

# ── Build action: run the batch file ─────────────────────────────────────────
$action = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/c `"$batFile`""

# ── Settings: run even if on battery, don't stop if idle ─────────────────────
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10)

# ── Principal: run as current user ───────────────────────────────────────────
$principal = New-ScheduledTaskPrincipal `
    -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive `
    -RunLevel Highest

# ── Register ─────────────────────────────────────────────────────────────────
Register-ScheduledTask `
    -TaskName  $taskName `
    -Trigger   $trigger `
    -Action    $action `
    -Settings  $settings `
    -Principal $principal `
    -Force | Out-Null

Write-Host ""
Write-Host "✅ Task registered: '$taskName'"
Write-Host "   Runs daily at 08:00 AM using: $batFile"
Write-Host "   Logs saved to: $(Join-Path $scriptDir 'prediction_log.txt')"
Write-Host ""
Write-Host "To verify, open Task Scheduler and look for '$taskName'"
Write-Host "To run it now for a test: Start-ScheduledTask -TaskName '$taskName'"
