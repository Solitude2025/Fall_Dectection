param(
    [switch]$UseH2
)

$ErrorActionPreference = 'Stop'
$jdk = 'C:\Program Files\BellSoft\LibericaJDK-17'
if (Test-Path (Join-Path $jdk 'bin\java.exe')) {
    $env:JAVA_HOME = $jdk
    $env:Path = "$jdk\bin;$env:Path"
}

if ($UseH2) {
    $env:SPRING_PROFILES_ACTIVE = 'h2'
    Write-Host 'Using the file-based H2 database because MySQL is not running.'
} else {
    Remove-Item Env:SPRING_PROFILES_ACTIVE -ErrorAction SilentlyContinue
    Write-Host 'Using MySQL.'
}

$backend = Join-Path $PSScriptRoot 'backend'
if (-not (Get-Command mvn -ErrorAction SilentlyContinue)) {
    throw 'Maven was not found. Install Maven and reopen the launcher.'
}
if (-not $env:JAVA_HOME -or -not (Test-Path (Join-Path $env:JAVA_HOME 'bin\javac.exe'))) {
    throw 'JDK 17+ was not found. Set JAVA_HOME to a JDK (not a JRE).'
}
Push-Location $backend
try {
    Write-Host 'Starting Spring Boot backend on http://localhost:8080'
    Write-Host 'If Maven is not installed, install JDK 17 + Maven first.'
    mvn spring-boot:run
    if ($LASTEXITCODE -ne 0) { throw "Spring Boot/Maven exited with code $LASTEXITCODE." }
} finally {
    Pop-Location
}
