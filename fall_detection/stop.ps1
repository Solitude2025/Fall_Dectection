$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'start.ps1')
try {
    $event = [Threading.EventWaitHandle]::OpenExisting((Get-ProjectStopEventName))
} catch [Threading.WaitHandleCannotBeOpenedException] {
    Write-Host 'No start.bat website session is running.'
    exit 0
}
try {
    $event.Set() | Out-Null
    Write-Host 'Stop requested. The start.bat window will close and stop its services.'
} finally { $event.Dispose() }
