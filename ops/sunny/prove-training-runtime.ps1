param(
    [Parameter(Mandatory = $true)]
    [string]$ReportDir,

    [Parameter(Mandatory = $true)]
    [string]$RepoRoot,

    [Parameter(Mandatory = $true)]
    [string]$RepoSrc,

    [string]$Target = "vision",
    [string]$Device = "cuda"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

[void](New-Item -ItemType Directory -Force -Path $ReportDir)

$results = New-Object System.Collections.Generic.List[object]
$NvidiaSmi = Join-Path $env:WINDIR "System32\nvidia-smi.exe"
$ModuleProbePath = Join-Path $ReportDir "windows-python-modules-probe.py"
$TorchProbePath = Join-Path $ReportDir "windows-torch-cuda-probe.py"

function Save-Probe {
    param(
        [string]$Name,
        [scriptblock]$Script
    )

    $path = Join-Path $ReportDir "$Name.txt"
    $exitCode = 0

    try {
        $output = & $Script 2>&1 | Out-String
        if ($null -ne $LASTEXITCODE) {
            $exitCode = [int]$LASTEXITCODE
        }
    } catch {
        $output = ($_ | Out-String)
        $exitCode = 1
    }

    Set-Content -Path $path -Value $output -Encoding utf8
    $results.Add([pscustomobject]@{
        name = $Name
        exit_code = $exitCode
        output_path = $path
    })

    Write-Host "$Name exit_code=$exitCode"
}

$moduleProbe = @'
import importlib.util
import json

mods = ["torch", "torchvision", "torchaudio", "transformers", "datasets"]
payload = {name: importlib.util.find_spec(name) is not None for name in mods}
print(json.dumps(payload, indent=2))
'@

$torchProbe = @'
import json
import torch

payload = {
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
}
if torch.cuda.is_available():
    payload["device_name"] = torch.cuda.get_device_name(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    payload["free_mb"] = int(free_bytes / (1024 ** 2))
    payload["total_mb"] = int(total_bytes / (1024 ** 2))
print(json.dumps(payload, indent=2))
'@

Set-Content -Path $ModuleProbePath -Value $moduleProbe -Encoding utf8
Set-Content -Path $TorchProbePath -Value $torchProbe -Encoding utf8

Save-Probe "windows-python-version" { py -3.11 --version }
Save-Probe "windows-nvidia-smi" { & $NvidiaSmi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader }
Save-Probe "windows-python-modules" { py -3.11 $ModuleProbePath }
Save-Probe "windows-torch-cuda" { py -3.11 $TorchProbePath }
Save-Probe "windows-ml-lab-check" {
    $env:PYTHONPATH = $RepoSrc
    Set-Location $RepoRoot
    py -3.11 -m industry_ml_lab.cli check --target $Target --device $Device --output-dir artifacts/windows-proof --json
}

$summaryPath = Join-Path $ReportDir "windows-summary.json"
$summary = [pscustomobject]@{
    report_dir = $ReportDir
    repo_root = $RepoRoot
    repo_src = $RepoSrc
    target = $Target
    device = $Device
    all_commands_succeeded = (@($results | Where-Object { $_.exit_code -ne 0 }).Count -eq 0)
    entries = $results
}
$summary | ConvertTo-Json -Depth 4 | Set-Content -Path $summaryPath -Encoding utf8
$summary | ConvertTo-Json -Depth 4
