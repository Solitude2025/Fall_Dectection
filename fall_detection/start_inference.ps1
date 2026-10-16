$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    Write-Host 'Starting Python camera inference service on http://localhost:8001'
    $python = 'python'
    foreach ($candidate in @('.venv\Scripts\python.exe', 'venv\Scripts\python.exe')) {
        $localPython = Join-Path $PSScriptRoot $candidate
        if (Test-Path $localPython) { $python = $localPython; break }
    }
    & $python -m services.inference_service --camera-index 0
    if ($LASTEXITCODE -ne 0) { throw "Python inference service exited with code $LASTEXITCODE." }
} finally {
    Pop-Location
}
