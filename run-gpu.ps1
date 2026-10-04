$ErrorActionPreference = 'Stop'
$taskOldPath = $env:PATH
$taskOldRequire = $env:TACOTRON_REQUIRE_GPU
$taskOldGrowth = $env:TF_FORCE_GPU_ALLOW_GROWTH
$taskOldTune = $env:TF_CUDNN_USE_AUTOTUNE
try {
    $taskGpuBin = Join-Path $PSScriptRoot '.runtime/gpu/Library/bin'
    if (!(Test-Path -LiteralPath "$taskGpuBin/cudnn64_7.dll")) { throw 'Run .\setup-gpu.ps1 first.' }
    $env:PATH = "$taskGpuBin;$taskOldPath"
    $env:TACOTRON_REQUIRE_GPU = '1'
    $env:TF_FORCE_GPU_ALLOW_GROWTH = 'true'
    $env:TF_CUDNN_USE_AUTOTUNE = '0'
    & "$PSScriptRoot/run.ps1" @args
}
finally {
    $env:PATH = $taskOldPath
    $env:TACOTRON_REQUIRE_GPU = $taskOldRequire
    $env:TF_FORCE_GPU_ALLOW_GROWTH = $taskOldGrowth
    $env:TF_CUDNN_USE_AUTOTUNE = $taskOldTune
}
