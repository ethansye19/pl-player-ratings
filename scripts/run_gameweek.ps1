# Runs the Step 6 pipeline for the latest finished FPL gameweek and appends a log.
#
# Behaviour:
#   - does nothing if the latest finished gameweek is already published (--if-new);
#   - never replaces published output with an older gameweek or fewer fixtures/rows
#     (the pipeline's regression guard), and writes nothing when a validity check fails;
#   - exit code 0 = published or nothing to do, 1 = checks failed / error (see the log);
#   - on any non-zero exit (checks failed, a Python crash with a traceback, the venv Python
#     missing) it shows a Windows notification with the first failure line and the log path.
#     Exit 0 stays silent. The log is written before the notification is attempted, and a
#     notification that cannot be shown is only logged: it never changes the exit code.
#     If the main Windows notifications switch is off (Settings > System > Notifications, i.e.
#     ToastEnabled = 0 under HKCU\Software\Microsoft\Windows\CurrentVersion\PushNotifications),
#     Windows would drop the notification silently, so the log says "not displayed: Windows
#     notifications are switched off" instead of "notification shown".
#     RATING_SYSTEM_NOTIFY=log writes the notification to the log instead of showing it, and
#     RATING_SYSTEM_NOTIFY=fail simulates a broken notification API; RATING_SYSTEM_PUSH_KEY
#     points the switch check at another registry key (all three used by the tests).
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

function Get-FailureLine($lines, [int]$code) {
    # the first listed check failure, else the exception line of a traceback, else the last line of output
    $text = @($lines | ForEach-Object { ($_ | Out-String).Trim() } | Where-Object { $_ })
    $check = $text | Where-Object { $_ -match '^-\s+\S' -or $_ -match 'check failed:' } | Select-Object -First 1
    if ($check) { return (($check -replace '^-\s+', '') -replace '^.*check failed:\s*', '') }
    $exception = $text | Where-Object { $_ -match '^[A-Za-z_][\w.]*(Error|Exception|Exit|Interrupt)\b' } | Select-Object -Last 1
    if ($exception) { return $exception }
    if ($text.Count -gt 0) { return $text[-1] }
    return "exit code $code with no output"
}

function Test-WindowsNotificationsOff {
    # the main switch in Settings > System > Notifications; a missing value means on (the Windows default)
    $key = if ($env:RATING_SYSTEM_PUSH_KEY) { $env:RATING_SYSTEM_PUSH_KEY } else { 'HKCU:\Software\Microsoft\Windows\CurrentVersion\PushNotifications' }
    $value = (Get-ItemProperty -Path $key -Name ToastEnabled -ErrorAction SilentlyContinue).ToastEnabled
    return ($null -ne $value -and [int]$value -eq 0)
}

function Send-FailureNotification([int]$code, [string]$detail) {
    # never throws: whatever happens here, the caller still exits with $code
    try {
        $title = "Rating pipeline failed (exit $code)"
        if ($detail.Length -gt 240) { $detail = $detail.Substring(0, 237) + "..." }
        if ($env:RATING_SYSTEM_NOTIFY -eq "log") {
            Write-Log "notification (not shown, RATING_SYSTEM_NOTIFY=log): $title | $detail | $log"
            return
        }
        if ($env:RATING_SYSTEM_NOTIFY -eq "fail") { throw "simulated notification failure (RATING_SYSTEM_NOTIFY=fail)" }
        if (Test-WindowsNotificationsOff) {
            Write-Log "notification not displayed: Windows notifications are switched off (Settings > System > Notifications): $title | $detail"
            return
        }
        [void][Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime]
        [void][Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime]
        $esc = { param($s) [System.Security.SecurityElement]::Escape($s) }
        $xml = New-Object Windows.Data.Xml.Dom.XmlDocument
        $xml.LoadXml(("<toast><visual><binding template=""ToastGeneric""><text>{0}</text><text>{1}</text><text>Log: {2}</text></binding></visual></toast>" -f
            (& $esc $title), (& $esc $detail), (& $esc $log)))
        # Windows PowerShell's own app id: toasts from an unregistered id are dropped silently
        $appId = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe'
        [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($appId).Show([Windows.UI.Notifications.ToastNotification]::new($xml))
        Write-Log "notification shown: $title | $detail"
    } catch {
        try { Write-Log ("notification could not be shown: {0}" -f $_.Exception.Message) } catch { }
    }
}

Set-Location $repo
$env:PYTHONUTF8 = "1"
Write-Log "run_gameweek start"
if (-not (Test-Path $python)) {
    $message = "venv python not found at $python"
    Write-Log "ERROR: $message"
    Write-Log "run_gameweek end (exit 1)"
    Send-FailureNotification 1 $message
    exit 1
}

# merge stderr into stdout inside cmd so PowerShell does not wrap log lines in NativeCommandError records
$output = & cmd /c "`"$python`" -m src.pipeline.gameweek --if-new 2>&1"
$code = $LASTEXITCODE
foreach ($line in $output) { Write-Log ("  " + ($line | Out-String).TrimEnd()) }
Write-Log "run_gameweek end (exit $code)"
if ($code -ne 0) {
    $detail = "exit code $code"
    try { $detail = Get-FailureLine $output $code } catch { }
    Send-FailureNotification $code $detail
}
exit $code
