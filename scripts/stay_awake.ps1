# Keeps Windows from idle-sleeping while a long job runs (like a video player does); changes no settings.
# Ends when the given log file contains the finish marker, or after -Hours.
#   powershell -NoProfile -File scripts\stay_awake.ps1 -Log raw\overnight_log.txt -Until "=== done" -Hours 12
param([string]$Log = "raw\overnight_log.txt", [string]$Until = "=== done", [double]$Hours = 12)
Add-Type -Namespace Win32 -Name Power -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);
'@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]"0x00000001"
$end = (Get-Date).AddHours($Hours)
[Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null
while ((Get-Date) -lt $end) {
    Start-Sleep -Seconds 60  # first a minute: the job may still be starting a fresh log
    if ((Test-Path $Log) -and (Select-String -Path $Log -SimpleMatch $Until -Quiet)) { break }
}
[Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
