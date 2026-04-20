param(
    [switch]$DryRun,
    [switch]$RestoreDemo
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$NvidiaSmi = Join-Path $env:WINDIR "System32\nvidia-smi.exe"

function Write-Section {
    param([string]$Message)
    Write-Host ""
    Write-Host "== $Message =="
}

function Get-DemoGpuServices {
    @(
        [pscustomobject]@{ Name = "gpt-oss-20b"; Port = 8013 },
        [pscustomobject]@{ Name = "qwen3-vl"; Port = 8082 },
        [pscustomobject]@{ Name = "qwen3-embedding"; Port = 8083 },
        [pscustomobject]@{ Name = "stt-server"; Port = 9001 },
        [pscustomobject]@{ Name = "tts-server"; Port = 9002 },
        [pscustomobject]@{ Name = "ocr-server"; Port = 9003 },
        [pscustomobject]@{ Name = "detect-server"; Port = 9004 }
    )
}

function Get-ListeningProcessIds {
    param([int]$Port)

    @(
        Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty OwningProcess -Unique
    ) | Where-Object { $_ -ne $null }
}

function Stop-DemoGpuServices {
    param([switch]$DryRunMode)

    $results = @()
    foreach ($service in Get-DemoGpuServices) {
        $pids = @(Get-ListeningProcessIds -Port $service.Port)

        if ($pids.Count -eq 0) {
            $results += [pscustomobject]@{
                Name = $service.Name
                Port = $service.Port
                Action = "already-stopped"
                Pids = ""
            }
            continue
        }

        if ($DryRunMode) {
            $results += [pscustomobject]@{
                Name = $service.Name
                Port = $service.Port
                Action = "would-stop"
                Pids = ($pids -join ",")
            }
            continue
        }

        Stop-Process -Id $pids -Force
        Start-Sleep -Seconds 1

        $stillListening = @(Get-ListeningProcessIds -Port $service.Port)
        $results += [pscustomobject]@{
            Name = $service.Name
            Port = $service.Port
            Action = if ($stillListening.Count -eq 0) { "stopped" } else { "failed-to-stop" }
            Pids = ($pids -join ",")
        }
    }

    return $results
}

function Show-GpuSummary {
    if (Test-Path $NvidiaSmi) {
        & $NvidiaSmi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
    } else {
        Write-Host "nvidia-smi.exe not found at $NvidiaSmi"
    }
}

if ($RestoreDemo) {
    Write-Section "Restore Demo Services"
    & "$PSScriptRoot\reinit-demo-stack.ps1"
    exit $LASTEXITCODE
}

Write-Section "Stop Demo GPU Services"
$results = Stop-DemoGpuServices -DryRunMode:$DryRun
$results | Format-Table -AutoSize

if ($DryRun) {
    Write-Section "Dry Run"
    Write-Host "No process state was changed."
    exit 0
}

Write-Section "GPU"
Show-GpuSummary
