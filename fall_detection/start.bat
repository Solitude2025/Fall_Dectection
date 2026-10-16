@echo off
setlocal EnableExtensions
title Fall Detection - close this window to stop
rem Keep orchestration outside CMD parenthesized blocks (GOTO breaks them).
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" (
    echo.
    echo Startup failed. See the message above and app_data\logs.
    pause
)
exit /b %RESULT%
