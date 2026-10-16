param(
    [switch]$NoBrowser,
    [ValidateRange(1, 600)][int]$StartupTimeoutSeconds = 120
)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$logDirectory = Join-Path $projectRoot 'app_data\logs'

function Test-ProjectService([int]$Port, [string]$Service) {
    try {
        $health = Invoke-RestMethod "http://127.0.0.1:$Port/api/health" -TimeoutSec 2
        return $health.status -eq 'UP' -and $health.service -eq $Service
    } catch {
        return $false
    }
}

function Test-LocalPort([int]$Port) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $connection = $client.BeginConnect('127.0.0.1', $Port, $null, $null)
        if (-not $connection.AsyncWaitHandle.WaitOne(300)) { return $false }
        $client.EndConnect($connection)
        return $true
    } catch {
        return $false
    } finally {
        $client.Dispose()
    }
}

function Invoke-ProjectDocker([string[]]$DockerArguments, [int]$TimeoutSeconds) {
    $process = Start-Process -FilePath 'docker.exe' -ArgumentList $DockerArguments `
        -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $logDirectory 'docker.stdout.log') `
        -RedirectStandardError (Join-Path $logDirectory 'docker.stderr.log')
    if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
        Stop-Process -Id $process.Id -ErrorAction SilentlyContinue
        return $false
    }
    $process.Refresh()
    return $process.ExitCode -eq 0
}

function Select-ProjectDatabase {
    if (Test-LocalPort 3306) { return $false }
    if (Get-Command docker.exe -ErrorAction SilentlyContinue) {
        Write-Host 'Checking Docker/MySQL...'
        if (Invoke-ProjectDocker @('info') 10) {
            if (Invoke-ProjectDocker @('compose', 'up', '-d', 'mysql') 45) {
                $deadline = [DateTime]::UtcNow.AddSeconds(30)
                do {
                    if (Test-LocalPort 3306) { return $false }
                    Start-Sleep -Seconds 1
                } while ([DateTime]::UtcNow -lt $deadline)
            }
        }
    }
    Write-Host 'MySQL is unavailable; using the existing file-based H2 database.'
    Write-Host 'H2 and MySQL have separate event histories; no data is migrated.'
    return $true
}

function Start-ProjectService([string]$Name, [string]$Script, [switch]$UseH2) {
    $arguments = '-NoLogo -NoProfile -ExecutionPolicy Bypass -File "{0}"' -f (Join-Path $projectRoot $Script)
    if ($UseH2) { $arguments += ' -UseH2' }
    $shell = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
    return $script:serviceScope.Start($shell, $arguments, $projectRoot,
        (Join-Path $logDirectory "$Name.stdout.log"), (Join-Path $logDirectory "$Name.stderr.log"))
}

function Get-ProjectStopEventName {
    $hash = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [Text.Encoding]::UTF8.GetBytes($projectRoot.ToLowerInvariant())
        return 'Local\FallDetection-' + ([BitConverter]::ToString($hash.ComputeHash($bytes))).Replace('-', '') + '-Stop'
    } finally { $hash.Dispose() }
}

function Start-Project {
    New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
    # Prevent double-clicking again while the first launch is still warming up.
    $lockPath = Join-Path $logDirectory 'startup.lock'
    try {
        $startupLock = [IO.File]::Open($lockPath, 'OpenOrCreate', 'ReadWrite', 'None')
    } catch [IO.IOException] {
        throw 'The website already has a start.bat window. Use that window; close it to stop the services.'
    }
    $stopEvent = $null
    $script:serviceScope = $null
    try {
        if (-not ('ServiceProcessScope' -as [type])) {
            Add-Type -Path (Join-Path $projectRoot 'scripts\ServiceProcessScope.cs')
        }
        $script:serviceScope = New-Object ServiceProcessScope
        $stopEvent = New-Object Threading.EventWaitHandle($false, [Threading.EventResetMode]::ManualReset, (Get-ProjectStopEventName))
        # Also detect the CMD owner exiting when only that process is killed.
        $parentId = (Get-CimInstance Win32_Process -Filter "ProcessId = $PID").ParentProcessId
        $owner = Get-Process -Id $parentId -ErrorAction SilentlyContinue
        $services = @(
            @{ Name = 'inference'; Port = 8001; Service = 'python-inference'; Script = 'start_inference.ps1' },
            @{ Name = 'backend'; Port = 8080; Service = 'fall-detection-backend'; Script = 'start_backend.ps1' }
        )
        # An unrelated listener is not proof that this project's service is up.
        foreach ($service in $services) {
            $service.Ready = Test-ProjectService $service.Port $service.Service
            if ($service.Ready) {
                throw "A previous unmanaged service is running on port $($service.Port). Stop the old services first, then run start.bat again."
            }
            if (-not $service.Ready -and (Test-LocalPort $service.Port)) {
                throw "Port $($service.Port) is occupied but the project health check failed. Check the existing process; it has not been stopped."
            }
        }
        foreach ($service in $services) {
            $useH2 = $false
            if ($service.Name -eq 'backend') { $useH2 = Select-ProjectDatabase }
            Write-Host "Starting $($service.Name) on port $($service.Port)..."
            $service.Process = Start-ProjectService $service.Name $service.Script -UseH2:$useH2
        }
        Write-Host 'Waiting for both services to become ready...'
        $deadline = [DateTime]::UtcNow.AddSeconds($StartupTimeoutSeconds)
        do {
            if ($stopEvent.WaitOne(0) -or ($owner -and $owner.HasExited)) { return }
            $allReady = $true
            foreach ($service in $services) {
                $service.Ready = Test-ProjectService $service.Port $service.Service
                if ($service.Ready) { continue }
                $allReady = $false
                if ($service.Process -and $service.Process.HasExited) {
                    throw "$($service.Name) exited before becoming ready. See $logDirectory\$($service.Name).stderr.log and .stdout.log."
                }
            }
            if ($allReady) {
                Write-Host 'Ready: http://localhost:8080'
                Write-Host "Logs: $logDirectory"
                Write-Host 'Keep this CMD window open while using the website.'
                Write-Host 'Close the window or press Ctrl+C to stop all website services.'
                if (-not $NoBrowser) { Start-Process 'http://localhost:8080' | Out-Null }
                while (-not $stopEvent.WaitOne(1000)) {
                    if ($owner -and $owner.HasExited) { break }
                    foreach ($service in $services) {
                        if ($service.Process.HasExited) {
                            throw "$($service.Name) stopped unexpectedly. All website services will stop; see $logDirectory."
                        }
                    }
                }
                return
            }
            Start-Sleep -Seconds 1
        } while ([DateTime]::UtcNow -lt $deadline)
        throw "Startup timed out. Services will be stopped; see $logDirectory before retrying."
    } finally {
        if ($script:serviceScope) { $script:serviceScope.Dispose() }
        if ($stopEvent) { $stopEvent.Dispose() }
        $startupLock.Dispose()
        Write-Host 'Website services stopped.'
    }
}

# Dot-sourcing exposes functions for isolated launcher regression tests.
if ($MyInvocation.InvocationName -ne '.') {
    try {
        Start-Project
        exit 0
    } catch {
        Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
}
