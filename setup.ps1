$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $taskRuntime = Join-Path $PSScriptRoot '.runtime/python37'
    if (!(Test-Path -LiteralPath "$taskRuntime/python.exe")) {
        New-Item -ItemType Directory -Force -Path $taskRuntime | Out-Null
        Invoke-WebRequest 'https://www.python.org/ftp/python/3.7.9/python-3.7.9-embed-amd64.zip' -OutFile '.runtime/python37.zip'
        Expand-Archive -LiteralPath '.runtime/python37.zip' -DestinationPath $taskRuntime -Force
    }
    Set-Content -LiteralPath "$taskRuntime/python37._pth" -Encoding ASCII -Value @('python37.zip', '.', $PSScriptRoot, 'import site')
    if (!(Test-Path -LiteralPath "$taskRuntime/Lib/site-packages/pip")) {
        Invoke-WebRequest 'https://bootstrap.pypa.io/pip/3.7/get-pip.py' -OutFile '.runtime/get-pip.py'
        & "$taskRuntime/python.exe" '.runtime/get-pip.py'
        if ($LASTEXITCODE -ne 0) { throw 'pip bootstrap failed' }
    }
    $taskRequirements = 'requirements-mini.txt'
    if (Test-Path -LiteralPath 'requirements-lock.txt') { $taskRequirements = 'requirements-lock.txt' }
    & "$taskRuntime/python.exe" -m pip install -r $taskRequirements
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
    & "$taskRuntime/python.exe" -c "import tensorflow as tf; print('TensorFlow',tf.__version__); assert hasattr(tf,'contrib')"
    if ($LASTEXITCODE -ne 0) { throw 'TensorFlow 1.x verification failed' }
}
finally { Pop-Location }
