param(
    [Parameter(Mandatory = $true)]
    [string]$ReportDir,

    [Parameter(Mandatory = $true)]
    [string]$RepoRoot,

    [Parameter(Mandatory = $true)]
    [string]$RepoSrc,

    [Parameter(Mandatory = $true)]
    [string]$OutputDir,

    [Parameter(Mandatory = $true)]
    [string]$DatasetRoot,

    [int]$Epochs = 1,
    [string]$Device = "cuda"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

[void](New-Item -ItemType Directory -Force -Path $ReportDir)
[void](New-Item -ItemType Directory -Force -Path $OutputDir)

$results = New-Object System.Collections.Generic.List[object]
$TempRoot = Join-Path $env:TEMP "industry-ml-lab"
[void](New-Item -ItemType Directory -Force -Path $TempRoot)
$TrainProbePath = Join-Path $TempRoot "vision-smoke-$PID.py"
$TrainProbeReportPath = Join-Path $ReportDir "windows-train-vision-probe.py"

function Save-Step {
    param(
        [string]$Name,
        [scriptblock]$Script
    )

    $path = Join-Path $ReportDir "$Name.txt"
    $exitCode = 0

    try {
        $global:LASTEXITCODE = 0
        $output = & $Script 2>&1 | Out-String
        $exitCode = [int]$LASTEXITCODE
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

function Save-NativeStep {
    param(
        [string]$Name,
        [string]$FilePath,
        [string[]]$ArgumentList
    )

    $combinedPath = Join-Path $ReportDir "$Name.txt"
    $stdoutPath = Join-Path $ReportDir "$Name-stdout.txt"
    $stderrPath = Join-Path $ReportDir "$Name-stderr.txt"
    $commandPath = Join-Path $ReportDir "$Name-command.txt"
    $quotedArgs = @($ArgumentList | ForEach-Object {
        if ($_ -match '\s') {
            '"' + $_ + '"'
        } else {
            $_
        }
    })
    $commandText = (@($FilePath) + $quotedArgs) -join " "
    $exitCode = 0
    $stdout = ""
    $stderr = ""

    Set-Content -Path $commandPath -Value $commandText -Encoding utf8

    try {
        if (Test-Path $stdoutPath) {
            Remove-Item -Force $stdoutPath
        }
        if (Test-Path $stderrPath) {
            Remove-Item -Force $stderrPath
        }

        $process = Start-Process `
            -FilePath $FilePath `
            -ArgumentList $ArgumentList `
            -NoNewWindow `
            -Wait `
            -PassThru `
            -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError $stderrPath
        $exitCode = [int]$process.ExitCode
    } catch {
        $exitCode = 1
        $stderr = ($_ | Out-String)
        Set-Content -Path $stderrPath -Value $stderr -Encoding utf8
    }

    if (Test-Path $stdoutPath) {
        $stdout = Get-Content -Path $stdoutPath -Raw
    }
    if (Test-Path $stderrPath) {
        $stderr = Get-Content -Path $stderrPath -Raw
    }

    $combined = @(
        "COMMAND: $commandText"
        ""
        "STDOUT:"
        $stdout.TrimEnd()
        ""
        "STDERR:"
        $stderr.TrimEnd()
    ) -join "`r`n"

    Set-Content -Path $combinedPath -Value $combined -Encoding utf8
    $results.Add([pscustomobject]@{
        name = $Name
        exit_code = $exitCode
        output_path = $combinedPath
    })

    Write-Host "$Name exit_code=$exitCode"
}

$trainProbe = @"
import json
import traceback
from pathlib import Path

output_dir = Path(r"$OutputDir")
dataset_root = Path(r"$DatasetRoot")
traceback_path = Path(r"$ReportDir") / "windows-train-vision-traceback.txt"

try:
    from industry_ml_lab.config import VisionTrainConfig
    from industry_ml_lab.training.vision import train

    config = VisionTrainConfig(
        dataset_root=dataset_root,
        output_dir=output_dir,
        epochs=$Epochs,
        batch_size=64,
        learning_rate=3e-4,
        num_workers=0,
        pretrained=False,
        device=r"$Device",
    )

    metrics = train(config)
    print(json.dumps(metrics, indent=2))
except BaseException:
    traceback_path.write_text(traceback.format_exc(), encoding="utf-8")
    traceback.print_exc()
    raise
"@

Set-Content -Path $TrainProbePath -Value $trainProbe -Encoding utf8
Set-Content -Path $TrainProbeReportPath -Value $trainProbe -Encoding utf8

$env:PYTHONPATH = $RepoSrc
Set-Location $RepoRoot

Save-Step "windows-python-version" { py -3.11 --version }
Save-Step "windows-vision-check" {
    py -3.11 -m industry_ml_lab.cli check --target vision --device $Device --output-dir $OutputDir --dataset-root $DatasetRoot --json
}
Save-NativeStep "windows-train-vision" "py" @("-3.11", $TrainProbePath)
Save-Step "windows-output-listing" {
    Get-ChildItem -Path $OutputDir -Recurse |
        Select-Object FullName, Length, LastWriteTime |
        Format-Table -AutoSize
}
Save-Step "windows-metrics-json" {
    Get-Content -Path (Join-Path $OutputDir "metrics.json")
}

$summaryPath = Join-Path $ReportDir "windows-vision-smoke-summary.json"
$summary = [pscustomobject]@{
    report_dir = $ReportDir
    repo_root = $RepoRoot
    repo_src = $RepoSrc
    output_dir = $OutputDir
    dataset_root = $DatasetRoot
    epochs = $Epochs
    device = $Device
    all_commands_succeeded = (@($results | Where-Object { $_.exit_code -ne 0 }).Count -eq 0)
    entries = $results
}
$summary | ConvertTo-Json -Depth 4 | Set-Content -Path $summaryPath -Encoding utf8
$summary | ConvertTo-Json -Depth 4
