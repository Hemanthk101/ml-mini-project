$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    if (!(Test-Path -LiteralPath '.runtime/python37/python.exe')) {
        & "$PSScriptRoot/setup.ps1"
    }
    & '.runtime/python37/python.exe' -m pip install zstandard==0.19.0
    if ($LASTEXITCODE -ne 0) { throw 'GPU installer dependency failed' }
    & '.runtime/python37/python.exe' download_gpu.py
    if ($LASTEXITCODE -ne 0) { throw 'CUDA/cuDNN download failed' }
    & "$PSScriptRoot/run-gpu.ps1" gpu_check.py
}
finally { Pop-Location }
