param(
    [Parameter(Mandatory = $true)]
    [string]$ReportDir,

    [Parameter(Mandatory = $true)]
    [string]$RepoRoot,

    [Parameter(Mandatory = $true)]
    [string]$RepoSrc,

    [Parameter(Mandatory = $true)]
    [string]$RecordsPath,

    [Parameter(Mandatory = $true)]
    [string]$OutputPath,

    [string]$Query = "free CUDA memory for training",
    [string]$Device = "cuda",
    [string]$ModelName = "sentence-transformers/all-MiniLM-L6-v2",
    [int]$BatchSize = 16,
    [int]$MaxLength = 256
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

[void](New-Item -ItemType Directory -Force -Path $ReportDir)
[void](New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutputPath))

$HfHome = Join-Path $env:LOCALAPPDATA "industry-ml-lab\hf-cache"
[void](New-Item -ItemType Directory -Force -Path $HfHome)
$env:HF_HOME = $HfHome

$cacheNote = @(
    "records_path=$RecordsPath"
    "output_path=$OutputPath"
    "hf_home=$HfHome"
) -join "`r`n"
Set-Content -Path (Join-Path $ReportDir "windows-embedding-cache.txt") -Value $cacheNote -Encoding utf8

$results = New-Object System.Collections.Generic.List[object]
$TempRoot = Join-Path $env:TEMP "industry-ml-lab"
[void](New-Item -ItemType Directory -Force -Path $TempRoot)
$IndexSummaryProbePath = Join-Path $TempRoot "embedding-index-summary-$PID.py"
$IndexSummaryProbeReportPath = Join-Path $ReportDir "windows-index-summary-probe.py"

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
    if ($null -eq $stdout) { $stdout = "" }
    if ($null -eq $stderr) { $stderr = "" }

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

$indexSummaryProbe = @"
import json
from pathlib import Path

payload = json.loads(Path(r"$OutputPath").read_text(encoding="utf-8"))
records = payload.get("records", [])
print(json.dumps({
    "embedding_model": payload.get("embedding_model"),
    "dimension": payload.get("dimension"),
    "record_count": len(records),
    "first_item_id": records[0]["item_id"] if records else None,
}, indent=2))
"@

Set-Content -Path $IndexSummaryProbePath -Value $indexSummaryProbe -Encoding utf8
Set-Content -Path $IndexSummaryProbeReportPath -Value $indexSummaryProbe -Encoding utf8

$env:PYTHONPATH = $RepoSrc
Set-Location $RepoRoot

Save-Step "windows-python-version" { py -3.11 --version }
Save-Step "windows-embedding-modules" {
    py -3.11 -c "import json, torch, transformers; print(json.dumps({'torch_version': torch.__version__, 'transformers_version': transformers.__version__, 'cuda_available': torch.cuda.is_available()}, indent=2))"
}
Save-NativeStep "windows-build-text-index" "py" @(
    "-3.11",
    "-m",
    "industry_ml_lab.cli",
    "build-text-index",
    "--records",
    $RecordsPath,
    "--output",
    $OutputPath,
    "--model-name",
    $ModelName,
    "--device",
    $Device,
    "--batch-size",
    "$BatchSize",
    "--max-length",
    "$MaxLength"
)
Save-NativeStep "windows-index-summary" "py" @("-3.11", $IndexSummaryProbePath)
Save-Step "windows-search-text-index" {
    py -3.11 -m industry_ml_lab.cli search-text-index `
        --index-path $OutputPath `
        --query $Query `
        --top-k 3 `
        --model-name $ModelName `
        --device $Device `
        --batch-size $BatchSize `
        --max-length $MaxLength
}

$summaryPath = Join-Path $ReportDir "windows-embedding-smoke-summary.json"
$summary = [pscustomobject]@{
    report_dir = $ReportDir
    repo_root = $RepoRoot
    repo_src = $RepoSrc
    records_path = $RecordsPath
    output_path = $OutputPath
    hf_home = $HfHome
    query = $Query
    device = $Device
    model_name = $ModelName
    batch_size = $BatchSize
    max_length = $MaxLength
    all_commands_succeeded = (@($results | Where-Object { $_.exit_code -ne 0 }).Count -eq 0)
    entries = $results
}
$summary | ConvertTo-Json -Depth 4 | Set-Content -Path $summaryPath -Encoding utf8
$summary | ConvertTo-Json -Depth 4

if (-not $summary.all_commands_succeeded) {
    exit 1
}
