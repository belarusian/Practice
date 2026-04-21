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

function Import-LabEnvFile {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        return
    }

    foreach ($line in Get-Content -Path $Path) {
        $trimmed = $line.Trim()
        if ([string]::IsNullOrWhiteSpace($trimmed) -or $trimmed.StartsWith("#")) {
            continue
        }

        $separator = $trimmed.IndexOf("=")
        if ($separator -le 0) {
            continue
        }

        $name = $trimmed.Substring(0, $separator).Trim()
        $value = $trimmed.Substring($separator + 1).Trim()

        if (-not ($name -match "^[A-Za-z_][A-Za-z0-9_]*$")) {
            continue
        }

        if (
            ($value.StartsWith("'") -and $value.EndsWith("'")) -or
            ($value.StartsWith('"') -and $value.EndsWith('"'))
        ) {
            $value = $value.Substring(1, $value.Length - 2)
        }

        if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($name))) {
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

function Get-LabEnv {
    param(
        [string]$Name,
        [string]$Default
    )
    $value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($value)) {
        return $Default
    }
    return $value
}

Import-LabEnvFile -Path (Join-Path $PSScriptRoot "lab.env")

$LlamaRoot = Get-LabEnv -Name "SUNNY_LLAMA_ROOT" -Default "C:\ml-lab\llama.cpp"
$ModelsRoot = Get-LabEnv -Name "SUNNY_MODELS_ROOT" -Default "C:\ml-lab\models"
$VoiceLabRoot = Get-LabEnv -Name "SUNNY_VOICE_LAB_ROOT" -Default "C:\ml-lab\voice-lab"
$WindowsScriptsRoot = Get-LabEnv -Name "SUNNY_WINDOWS_SCRIPTS_ROOT" -Default "C:\ml-lab"
$LlamaServerExe = Join-Path $LlamaRoot "build\bin\llama-server.exe"
$GptModelPath = Join-Path $LlamaRoot "models\gpt-oss-20b-mxfp4.gguf"
$VisionModelPath = Join-Path $ModelsRoot "Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf"
$VisionMmprojPath = Join-Path $ModelsRoot "mmproj-Qwen3VL-30B-A3B-Instruct-F16.gguf"
$EmbeddingModelPath = Join-Path $ModelsRoot "Qwen3-Embedding-4B-Q4_K_M.gguf"
$SttServerPath = Join-Path $VoiceLabRoot "servers\stt_server.py"
$TtsServerPath = Join-Path $VoiceLabRoot "servers\tts_server.py"
$TtsVoicePath = Join-Path $VoiceLabRoot "models\en_US-libritts_r-medium.onnx"
$OcrServerPath = Join-Path $WindowsScriptsRoot "ocr_server.py"
$DetectServerPath = Join-Path $WindowsScriptsRoot "detect_server.py"

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
        FilePath = $LlamaServerExe
        ArgumentList = "-m `"$GptModelPath`" -a gpt-oss-20b -c 131072 -fa on -ngl 99 --jinja --port 8013 --host 0.0.0.0"
        WorkingDirectory = $LlamaRoot
        HealthUrl = "http://127.0.0.1:8013/v1/models"
        ReadyAttempts = 30
    }
}

if (-not $SkipVision) {
    $services += [pscustomobject]@{
        Name = "qwen3-vl"
        Slug = "qwen3-vl"
        Port = 8082
        FilePath = $LlamaServerExe
        ArgumentList = "-m `"$VisionModelPath`" --mmproj `"$VisionMmprojPath`" -np 1 -ngl 99 --host 0.0.0.0 --port 8082"
        WorkingDirectory = $LlamaRoot
        HealthUrl = "http://127.0.0.1:8082/v1/models"
        ReadyAttempts = 30
    }
}

if (-not $SkipEmbed) {
    $services += [pscustomobject]@{
        Name = "qwen3-embedding"
        Slug = "qwen3-embedding"
        Port = 8083
        FilePath = $LlamaServerExe
        ArgumentList = "-m `"$EmbeddingModelPath`" --embedding --host 0.0.0.0 --port 8083"
        WorkingDirectory = $LlamaRoot
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
        ArgumentList = "-3.11 `"$SttServerPath`" --port 9001 --device cuda"
        WorkingDirectory = $VoiceLabRoot
        HealthUrl = "http://127.0.0.1:9001/health"
        ReadyAttempts = 20
    }
    $services += [pscustomobject]@{
        Name = "tts-server"
        Slug = "tts-server"
        Port = 9002
        FilePath = "py"
        ArgumentList = "-3.11 `"$TtsServerPath`" --port 9002 --voice `"$TtsVoicePath`""
        WorkingDirectory = $VoiceLabRoot
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
        ArgumentList = "-3.11 `"$OcrServerPath`""
        WorkingDirectory = $WindowsScriptsRoot
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
        ArgumentList = "-3.11 `"$DetectServerPath`""
        WorkingDirectory = $WindowsScriptsRoot
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
