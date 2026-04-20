param(
    [switch]$StatusOnly,
    [switch]$SkipGpt,
    [switch]$SkipVision,
    [switch]$SkipEmbed,
    [switch]$SkipSpeech,
    [switch]$SkipOcr,
    [switch]$SkipDetect
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$LogRoot = Join-Path $env:USERPROFILE "demo-stack-logs"
$NvidiaSmi = Join-Path $env:WINDIR "System32\nvidia-smi.exe"
New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null

function Write-Section {
    param([string]$Message)
    Write-Host ""
    Write-Host "== $Message =="
}

function Test-PortListening {
    param([int]$Port)
    return $null -ne (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue | Select-Object -First 1)
}

function Test-HealthUrl {
    param([string]$Url)
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -Method Get -TimeoutSec 5
        return $response.StatusCode -ge 200 -and $response.StatusCode -lt 500
    } catch {
        return $false
    }
}

function Start-ManagedProcess {
    param($Service)

    $stdoutLog = Join-Path $LogRoot "$($Service.Slug).out.log"
    $stderrLog = Join-Path $LogRoot "$($Service.Slug).err.log"

    Write-Host "starting $($Service.Name) on port $($Service.Port)"
    Start-Process `
        -FilePath $Service.FilePath `
        -ArgumentList $Service.ArgumentList `
        -WorkingDirectory $Service.WorkingDirectory `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog | Out-Null
}

function Wait-ForHealth {
    param($Service)

    for ($attempt = 1; $attempt -le $Service.ReadyAttempts; $attempt++) {
        if (Test-HealthUrl -Url $Service.HealthUrl) {
            Write-Host "healthy  $($Service.Name) -> $($Service.HealthUrl)"
            return $true
        }
        Start-Sleep -Seconds 2
    }

    Write-Warning "health check failed for $($Service.Name) after start: $($Service.HealthUrl)"
    return $false
}

function Ensure-ServiceState {
    param($Service)

    $listening = Test-PortListening -Port $Service.Port
    $healthy = Test-HealthUrl -Url $Service.HealthUrl

    if ($listening -and $healthy) {
        Write-Host "healthy  $($Service.Name) port $($Service.Port)"
        return $true
    }

    if ($StatusOnly) {
        if ($listening) {
            Write-Warning "$($Service.Name) is listening on $($Service.Port) but health failed"
        } else {
            Write-Warning "$($Service.Name) is not listening on $($Service.Port)"
        }
        return $false
    }

    if ($listening -and -not $healthy) {
        Write-Warning "$($Service.Name) already owns port $($Service.Port) but health is failing; not replacing automatically"
        return $false
    }

    Start-ManagedProcess -Service $Service
    return Wait-ForHealth -Service $Service
}

$services = @()

if (-not $SkipGpt) {
    $services += [pscustomobject]@{
        Name = "gpt-oss-20b"
        Slug = "gpt-oss-20b"
        Port = 8013
        FilePath = "C:\Users\kodep\llama.cpp\build\bin\llama-server.exe"
        ArgumentList = '-m models\gpt-oss-20b-mxfp4.gguf -a gpt-oss-20b -c 131072 -fa on -ngl 99 --jinja --port 8013 --host 0.0.0.0'
        WorkingDirectory = "C:\Users\kodep\llama.cpp"
        HealthUrl = "http://127.0.0.1:8013/v1/models"
        ReadyAttempts = 30
    }
}

if (-not $SkipVision) {
    $services += [pscustomobject]@{
        Name = "qwen3-vl"
        Slug = "qwen3-vl"
        Port = 8082
        FilePath = "C:\Users\kodep\llama.cpp\build\bin\llama-server.exe"
        ArgumentList = '-m C:\Users\kodep\models\Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf --mmproj C:\Users\kodep\models\mmproj-Qwen3VL-30B-A3B-Instruct-F16.gguf -np 1 -ngl 99 --host 0.0.0.0 --port 8082'
        WorkingDirectory = "C:\Users\kodep\llama.cpp"
        HealthUrl = "http://127.0.0.1:8082/v1/models"
        ReadyAttempts = 30
    }
}

if (-not $SkipEmbed) {
    $services += [pscustomobject]@{
        Name = "qwen3-embedding"
        Slug = "qwen3-embedding"
        Port = 8083
        FilePath = "C:\Users\kodep\llama.cpp\build\bin\llama-server.exe"
        ArgumentList = '-m C:\Users\kodep\models\Qwen3-Embedding-4B-Q4_K_M.gguf --embedding --host 0.0.0.0 --port 8083'
        WorkingDirectory = "C:\Users\kodep\llama.cpp"
        HealthUrl = "http://127.0.0.1:8083/v1/models"
        ReadyAttempts = 20
    }
}

if (-not $SkipSpeech) {
    $services += [pscustomobject]@{
        Name = "stt-server"
        Slug = "stt-server"
        Port = 9001
        FilePath = "py"
        ArgumentList = '-3.11 C:\Users\kodep\voice-lab\servers\stt_server.py --port 9001 --device cuda'
        WorkingDirectory = "C:\Users\kodep\voice-lab"
        HealthUrl = "http://127.0.0.1:9001/health"
        ReadyAttempts = 20
    }
    $services += [pscustomobject]@{
        Name = "tts-server"
        Slug = "tts-server"
        Port = 9002
        FilePath = "py"
        ArgumentList = '-3.11 C:\Users\kodep\voice-lab\servers\tts_server.py --port 9002 --voice C:\Users\kodep\voice-lab\models\en_US-libritts_r-medium.onnx'
        WorkingDirectory = "C:\Users\kodep\voice-lab"
        HealthUrl = "http://127.0.0.1:9002/health"
        ReadyAttempts = 20
    }
}

if (-not $SkipOcr) {
    $services += [pscustomobject]@{
        Name = "ocr-server"
        Slug = "ocr-server"
        Port = 9003
        FilePath = "py"
        ArgumentList = '-3.11 C:\Users\kodep\ocr_server.py'
        WorkingDirectory = "C:\Users\kodep"
        HealthUrl = "http://127.0.0.1:9003/health"
        ReadyAttempts = 20
    }
}

if (-not $SkipDetect) {
    $services += [pscustomobject]@{
        Name = "detect-server"
        Slug = "detect-server"
        Port = 9004
        FilePath = "py"
        ArgumentList = '-3.11 C:\Users\kodep\detect_server.py'
        WorkingDirectory = "C:\Users\kodep"
        HealthUrl = "http://127.0.0.1:9004/health"
        ReadyAttempts = 20
    }
}

Write-Section "Windows Demo Stack"
foreach ($service in $services) {
    Ensure-ServiceState -Service $service | Out-Null
}

Write-Section "Windows Status"
$services |
    ForEach-Object {
        [pscustomobject]@{
            Name = $_.Name
            Port = $_.Port
            Listening = Test-PortListening -Port $_.Port
            Healthy = Test-HealthUrl -Url $_.HealthUrl
        }
    } |
    Format-Table -AutoSize

Write-Section "GPU"
if (Test-Path $NvidiaSmi) {
    & $NvidiaSmi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
} else {
    Write-Host "nvidia-smi.exe not found at $NvidiaSmi"
}
