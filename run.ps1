# Invoke the isolated runtime; forward all arguments to Python.
$ErrorActionPreference = 'Stop'
$taskPython = Join-Path $PSScriptRoot '.runtime/python37/python.exe'
if (!(Test-Path -LiteralPath $taskPython)) { throw 'Run .\setup.ps1 first.' }
Push-Location $PSScriptRoot
try { & $taskPython @args; if ($LASTEXITCODE -ne 0) { throw "Python exited with code $LASTEXITCODE" } }
finally { Pop-Location }
