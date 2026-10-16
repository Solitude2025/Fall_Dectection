$ErrorActionPreference = 'Stop'
$launcherPath = Join-Path (Split-Path $PSScriptRoot -Parent) 'start.ps1'

function Assert-True($Condition, [string]$Message) {
    if (-not $Condition) { throw "FAIL: $Message" }
    Write-Host "PASS: $Message"
}

# Parse every entry point with Windows PowerShell's own parser.
foreach ($name in @('start.ps1', 'stop.ps1', 'start_backend.ps1', 'start_inference.ps1')) {
    $parseErrors = $null
    $tokens = $null
    $path = Join-Path (Split-Path $PSScriptRoot -Parent) $name
    $null = [Management.Automation.Language.Parser]::ParseFile($path, [ref]$tokens, [ref]$parseErrors)
    Assert-True ($parseErrors.Count -eq 0) "$name parses"
}

. $launcherPath
function Test-LocalPort { return $true }
Assert-True (-not (Select-ProjectDatabase)) 'An existing MySQL listener selects MySQL'

. $launcherPath
function Test-LocalPort { return $false }
function Invoke-ProjectDocker { return $false }
Assert-True (Select-ProjectDatabase) 'Unavailable Docker falls back to H2'

. $launcherPath
$script:portChecks = 0
$script:dockerCalls = @()
function Test-LocalPort {
    $script:portChecks++
    return $script:portChecks -gt 1
}
function Get-Command { return @{ Name = 'docker.exe' } }
function Invoke-ProjectDocker([string[]]$DockerArguments, [int]$TimeoutSeconds) {
    $script:dockerCalls += ($DockerArguments -join ' ')
    return $true
}
Assert-True (-not (Select-ProjectDatabase)) 'MySQL becoming ready after Docker selects MySQL without a CMD jump'
Assert-True (($script:dockerCalls -join '|') -eq 'info|compose up -d mysql') 'Docker receives the expected commands'

. $launcherPath
function Get-Command { return @{ Name = 'docker.exe' } }
function Test-LocalPort { return $false }
function Invoke-ProjectDocker([string[]]$DockerArguments, [int]$TimeoutSeconds) {
    return $DockerArguments[0] -eq 'info'
}
Assert-True (Select-ProjectDatabase) 'Docker Compose failure falls back to H2'

. $launcherPath
$NoBrowser = $true
function Test-ProjectService { return $true }
function Start-ProjectService { throw 'Healthy services must not be relaunched' }
$rejected = $false
try { Start-Project } catch { $rejected = $_.Exception.Message -like 'A previous unmanaged service*' }
Assert-True $rejected 'Unmanaged services are rejected instead of escaping the launch window lifetime'

. $launcherPath
$NoBrowser = $true
$script:launches = @()
$script:healthChecks = 0
function Test-ProjectService {
    $script:healthChecks++
    return $script:healthChecks -gt 2
}
function Test-LocalPort { return $false }
function Select-ProjectDatabase { return $true }
function Start-ProjectService([string]$Name, [string]$Script, [switch]$UseH2) {
    $script:launches += "$Name/$UseH2"
    if ($Name -eq 'backend') {
        $event = [Threading.EventWaitHandle]::OpenExisting((Get-ProjectStopEventName))
        $event.Set() | Out-Null
        $event.Dispose()
    }
    return @{ HasExited = $false }
}
Start-Project
Assert-True (($script:launches -join '|') -eq 'inference/False|backend/True') 'Cold startup launches both services and passes H2 only to backend'

. $launcherPath
function Test-ProjectService { return $false }
function Test-LocalPort { return $true }
function Start-ProjectService { throw 'Port conflict must not launch anything' }
$rejected = $false
try { Start-Project } catch { $rejected = $_.Exception.Message -like 'Port 8001 is occupied*' }
Assert-True $rejected 'Unrelated port listener is rejected'

. $launcherPath
function Test-ProjectService { return $false }
function Test-LocalPort { return $false }
function Select-ProjectDatabase { return $true }
function Start-ProjectService { return @{ HasExited = $true } }
$rejected = $false
try { Start-Project } catch { $rejected = $_.Exception.Message -like 'inference exited before becoming ready*' }
Assert-True $rejected 'Child process failure is reported instead of claiming success'

Write-Host 'All startup regression tests passed.'
