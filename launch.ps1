$ErrorActionPreference = 'SilentlyContinue'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$backendDir = Join-Path $root "backend"
$appExe = Join-Path $root "frontend\release\win-unpacked\IRIS.exe"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " IRIS: Satellite Intelligence Desktop" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# 1. Check if backend on port 8000 is running
$conn = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if (-not $conn) {
    Write-Host "Starting Python AI backend (127.0.0.1:8000)..." -ForegroundColor Yellow
    Start-Process -FilePath $python -ArgumentList "-m uvicorn main:app --host 127.0.0.1 --port 8000" -WorkingDirectory $backendDir -WindowStyle Hidden
    
    # Wait until port 8000 is listening (up to 15 seconds)
    $attempts = 0
    while ($attempts -lt 15) {
        Start-Sleep -Seconds 1
        $conn = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
        if ($conn) { break }
        $attempts++
    }
    if ($conn) {
        Write-Host "Backend started successfully." -ForegroundColor Green
    } else {
        Write-Host "Warning: Backend took longer than expected to start." -ForegroundColor Yellow
    }
} else {
    Write-Host "AI backend is active." -ForegroundColor Green
}

# 2. Launch IRIS Desktop Application
Write-Host "Launching IRIS Desktop Window..." -ForegroundColor Cyan
if (Test-Path $appExe) {
    Start-Process -FilePath $appExe -WorkingDirectory (Split-Path -Parent $appExe)
} else {
    Write-Host "Executable not found at: $appExe" -ForegroundColor Red
    Read-Host "Press Enter to exit"
}
