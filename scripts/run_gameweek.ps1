# Runs the Step 6 pipeline for the latest finished FPL gameweek and appends a log.
#
# Behaviour:
#   - does nothing if the latest finished gameweek is already published (--if-new);
#   - never replaces published output with an older gameweek or fewer fixtures/rows
#     (the pipeline's regression guard), and writes nothing when a validity check fails;
#   - exit code 0 = published or nothing to do, 1 = checks failed / error (see the log).
#
# Schedule it with Task Scheduler (see README, "Step 6") or run it by hand:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_gameweek.ps1

$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo ".venv\Scripts\python.exe"
$logDir = Join-Path $repo "logs"
$log = Join-Path $logDir "pipeline.log"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Write-Log([string]$message) {
    $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $message
    Add-Content -Path $log -Value $line -Encoding UTF8
}

Set-Location $repo
$env:PYTHONUTF8 = "1"
Write-Log "run_gameweek start"
if (-not (Test-Path $python)) {
    Write-Log "ERROR: venv python not found at $python"
    exit 1
}

# merge stderr into stdout inside cmd so PowerShell does not wrap log lines in NativeCommandError records
$output = & cmd /c "`"$python`" -m src.pipeline.gameweek --if-new 2>&1"
$code = $LASTEXITCODE
foreach ($line in $output) { Write-Log ("  " + ($line | Out-String).TrimEnd()) }
Write-Log "run_gameweek end (exit $code)"
exit $code
