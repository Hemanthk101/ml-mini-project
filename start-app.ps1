$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
& powershell.exe -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'run-gpu.ps1') app.py @args
