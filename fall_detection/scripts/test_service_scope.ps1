param([switch]$Worker)
$ErrorActionPreference = 'Stop'
$project = Split-Path $PSScriptRoot -Parent
$testDirectory = Join-Path $project 'app_data\launcher_scope_test'
New-Item -ItemType Directory -Path $testDirectory -Force | Out-Null
$shell = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'

if ($Worker) {
    Add-Type -Path (Join-Path $PSScriptRoot 'ServiceProcessScope.cs')
    $scope = New-Object ServiceProcessScope
    # The sleeping child and its grandchild must both die on owner termination.
    $child = $scope.Start($shell, '-NoProfile -ExecutionPolicy Bypass -File "' + (Join-Path $PSScriptRoot 'test_service_scope_child.ps1') + '"',
        $project, (Join-Path $testDirectory 'worker.stdout.log'), (Join-Path $testDirectory 'worker.stderr.log'))
    Write-Output $child.Id
    while ($true) { Start-Sleep -Seconds 1 }
}

$owner = Start-Process $shell -ArgumentList ('-NoProfile -ExecutionPolicy Bypass -File "' + $PSCommandPath + '" -Worker') `
    -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $testDirectory 'owner.stdout.log') `
    -RedirectStandardError (Join-Path $testDirectory 'owner.stderr.log')
try {
    $deadline = [DateTime]::UtcNow.AddSeconds(15)
    do {
        $child = Get-CimInstance Win32_Process -Filter "ParentProcessId = $($owner.Id)" | Where-Object Name -eq 'powershell.exe' | Select-Object -First 1
        if ($child) { $grandchild = Get-CimInstance Win32_Process -Filter "ParentProcessId = $($child.ProcessId)" | Select-Object -First 1 }
        if ($grandchild) { break }
        Start-Sleep -Milliseconds 100
    } while ([DateTime]::UtcNow -lt $deadline)
    if (-not $grandchild) { throw 'Worker descendants did not start' }
    # Intentionally kill ONLY the owner: no tree-kill and no finally cleanup.
    Stop-Process -Id $owner.Id
    $deadline = [DateTime]::UtcNow.AddSeconds(5)
    do {
        $survivors = @(Get-Process -Id $child.ProcessId,$grandchild.ProcessId -ErrorAction SilentlyContinue)
        if (-not $survivors.Count) { break }
        Start-Sleep -Milliseconds 100
    } while ([DateTime]::UtcNow -lt $deadline)
    if ($survivors.Count) { throw 'Child processes survived owner termination' }
    Write-Host 'PASS: Force-closing the owner stops its worker and grandchild automatically.'
} finally {
    if (-not $owner.HasExited) { Stop-Process -Id $owner.Id -ErrorAction SilentlyContinue }
}
