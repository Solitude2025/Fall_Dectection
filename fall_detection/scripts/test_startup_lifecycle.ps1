$ErrorActionPreference = 'Stop'
$project = Split-Path $PSScriptRoot -Parent
. (Join-Path $project 'start.ps1')
$testDirectory = Join-Path $project 'app_data\launcher_scope_test'
New-Item -ItemType Directory -Path $testDirectory -Force | Out-Null
if ((Test-LocalPort 8001) -or (Test-LocalPort 8080)) { throw 'Stop existing services before lifecycle verification.' }

foreach ($method in @('close-cmd', 'stop-bat')) {
    $launcher = Start-Process $env:ComSpec -ArgumentList ('/d /c ""' + (Join-Path $project 'start.bat') + '" -NoBrowser"') `
        -WorkingDirectory $project -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $testDirectory "$method.stdout.log") `
        -RedirectStandardError (Join-Path $testDirectory "$method.stderr.log")
    try {
        $deadline = [DateTime]::UtcNow.AddSeconds(150)
        do {
            if ((Test-ProjectService 8001 'python-inference') -and (Test-ProjectService 8080 'fall-detection-backend')) { break }
            if ($launcher.HasExited) { throw "Launcher exited early: $method" }
            Start-Sleep -Seconds 1
        } while ([DateTime]::UtcNow -lt $deadline)
        if (-not (Test-ProjectService 8080 'fall-detection-backend')) { throw 'Backend did not become ready' }
        if ($launcher.HasExited) { throw 'CMD did not remain open after startup' }
        $processes = @(Get-CimInstance Win32_Process)
        $owned = @($launcher.Id)
        do {
            $children = @($processes | Where-Object { $_.ParentProcessId -in $owned -and $_.ProcessId -notin $owned })
            $owned += @($children | ForEach-Object { [int]$_.ProcessId })
        } while ($children.Count)
        if ($method -eq 'close-cmd') {
            # Kill only CMD, not its tree: this reproduces an owner disappearing.
            Stop-Process -Id $launcher.Id
        } else {
            & (Join-Path $project 'stop.bat')
        }
        $deadline = [DateTime]::UtcNow.AddSeconds(15)
        do {
            $survivors = @(Get-Process -Id $owned -ErrorAction SilentlyContinue)
            if (-not $survivors.Count -and -not (Test-LocalPort 8001) -and -not (Test-LocalPort 8080)) { break }
            Start-Sleep -Milliseconds 200
        } while ([DateTime]::UtcNow -lt $deadline)
        if ($survivors.Count -or (Test-LocalPort 8001) -or (Test-LocalPort 8080)) { throw "Services survived $method" }
        Write-Host "PASS: $method closes CMD, Python, Maven, Java and both service ports."
    } finally {
        if (-not $launcher.HasExited) { Stop-Process -Id $launcher.Id -ErrorAction SilentlyContinue }
    }
}
