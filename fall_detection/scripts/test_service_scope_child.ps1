$shell = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
Start-Process $shell -ArgumentList '-NoProfile -Command "Start-Sleep -Seconds 120"' -WindowStyle Hidden | Out-Null
Start-Sleep -Seconds 120
