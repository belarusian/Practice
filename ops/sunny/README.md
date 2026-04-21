# Sunny Ops

These scripts turn the current Sunny demo stack into repo-backed operational state.

## Files

- `audit-demo-stack.sh`
  - Runs from Sunny WSL2.
  - Reports WSL2 service state, local health checks, WireGuard status, and Windows-side process health.

- `reinit-demo-stack.sh`
  - Runs from Sunny WSL2.
  - Starts the WSL2 service layer, then invokes the Windows PowerShell script to ensure the demo processes are running.

- `reinit-demo-stack.ps1`
  - Runs on Windows, usually through the WSL2 wrapper.
  - Starts or verifies the Windows-side processes on `8013`, `8082`, `8083`, `9001`, `9002`, `9003`, and `9004`.

- `training-mode.sh`
  - Runs from Sunny WSL2.
  - Stops the Windows-side GPU demo services so the 4090 can be used for training.
  - Supports `--dry-run`, `--audit-after-stop`, and `--restore-demo`.

- `training-mode.ps1`
  - Runs on Windows, usually through the WSL2 wrapper.
  - Stops listeners on `8013`, `8082`, `8083`, `9001`, `9002`, `9003`, and `9004`, or restores them by delegating back to `reinit-demo-stack.ps1`.

- `prove-training-runtime.sh`
  - Runs from Sunny WSL2.
  - Captures a report for the real WSL training path and the Windows CUDA training path.
  - Writes logs and summaries under `artifacts/sunny-reports/<timestamp>/`.

- `prove-training-runtime.ps1`
  - Runs on Windows through the WSL2 wrapper.
  - Probes Windows Python 3.11, CUDA PyTorch availability, `nvidia-smi`, and `ml-lab check` against the repo.

- `run-remote-training-proof.sh`
  - Runs from another machine such as this laptop.
  - Syncs the repo to Sunny, executes the remote proof script there, and pulls the report back locally.

- `vision-smoke.sh`
  - Runs from Sunny WSL2.
  - Switches Sunny into training mode, invokes the Windows `vision` smoke train, and captures logs plus artifacts into a report directory.

- `vision-smoke.ps1`
  - Runs on Windows through the WSL2 wrapper.
  - Executes the repo-based `vision` readiness check and then runs `train-vision` with Windows Python `3.11`.
  - Preserves the generated Python probe in the report and captures separate stdout/stderr files for the train step.

- `run-remote-vision-smoke.sh`
  - Runs from another machine such as this laptop.
  - Syncs the repo to Sunny, executes the smoke train there, and pulls the report back locally.

- `audio-smoke.sh`
  - Runs from Sunny WSL2.
  - Same orchestration as `vision-smoke.sh`, but invokes Windows **`audio`** check and `train-audio` (SpeechCommands under `data/audio/`).

- `audio-smoke.ps1`
  - Runs on Windows through the WSL2 wrapper.
  - `ml-lab check --target audio`, then one-epoch audio training via the same probe pattern as vision.
  - If the dataset path is a **`\\wsl…` UNC** (typical when WSL passes `wslpath -w` for `data/audio`), the script **stages a copy under `%LOCALAPPDATA%\industry-ml-lab\audio-smoke-speech-commands`** so `torchaudio.load()` reads normal Windows paths, not the WSL redirector.

- `run-remote-audio-smoke.sh`
  - Runs from another machine such as this laptop.
  - Syncs the repo to Sunny, runs `audio-smoke.sh`, pulls the report back locally.

- `text-smoke.sh`
  - Runs from Sunny WSL2.
  - Same orchestration as `vision-smoke.sh`, but invokes Windows **`text`** check and `train-text-classifier` on GLUE/SST-2.
  - Uses bounded sample limits by default so the smoke proves the transformer path without becoming a full benchmark.

- `text-smoke.ps1`
  - Runs on Windows through the WSL2 wrapper.
  - `ml-lab check --target text`, then one-epoch DistilBERT sequence-classifier training via the same probe pattern as vision/audio.
  - If the dataset cache path is a **`\\wsl…` UNC**, the script uses a Windows-local cache under `%LOCALAPPDATA%\industry-ml-lab\text-smoke-datasets`; Hugging Face model cache is placed under `%LOCALAPPDATA%\industry-ml-lab\hf-cache`.

- `run-remote-text-smoke.sh`
  - Runs from another machine such as this laptop.
  - Syncs the repo to Sunny, runs `text-smoke.sh`, pulls the report back locally.

## Remote wrappers and exit codes

`run-remote-training-proof.sh`, `run-remote-vision-smoke.sh`, `run-remote-audio-smoke.sh`, and `run-remote-text-smoke.sh` use **`set -o pipefail`** so a failed remote `ssh` session is not masked by `tee`.

After rsync pulls `artifacts/sunny-reports/<stamp>/`, they run **`exit_from_summary_json.py`** on **`summary.json`**. The process exits **0** only when both the **SSH** step and **`all_commands_succeeded`** in the report are good—suitable for **CI** or **`make sunny-proof` / `make sunny-vision-smoke` / `make sunny-audio-smoke` / `make sunny-text-smoke`** as hard gates.

`audio-smoke.ps1`, `vision-smoke.ps1`, and `text-smoke.ps1` **`exit 1`** when any nested Windows step fails (not only when the script throws). **`summarize_report_status.py`** also forces overall failure if **`windows-*-smoke-summary.json`** in the report directory has **`all_commands_succeeded`: false**, so a stale green **`windows-*-smoke-console`** line cannot mask a red nested summary.

## Training target: `vision`, `audio`, and `text`

`prove-training-runtime.sh` and `run-remote-training-proof.sh` accept **`--target vision`** (default), **`--target audio`**, or **`--target text`**. The Windows path runs `ml-lab check --target …`, so **`torchaudio`** is required for **`audio`** and **`transformers`** plus **`datasets`** are required for **`text`**.

Before adding audio smoke, prove the Windows runtime on Sunny (PowerShell on the box or via WSL):

```powershell
Set-Location '\\wsl.localhost\Ubuntu\home\ml-lab\Practice'
$env:PYTHONPATH = '\\wsl.localhost\Ubuntu\home\ml-lab\Practice\src'
py -3.11 -c "import importlib.util, json, torch; print(json.dumps({'torch': True, 'torchaudio': importlib.util.find_spec('torchaudio') is not None, 'cuda_available': torch.cuda.is_available()}, indent=2))"
```

From a laptop, after **`torchaudio`** is confirmed on Sunny Windows:

```bash
make sunny-proof-audio
```

Equivalent:

```bash
bash ops/sunny/run-remote-training-proof.sh --with-training-mode --restore-demo --target audio
```

Proof-mode summary (**`--proof`**) still treats expected WSL red as non-gating; the Windows **`ml-lab check`** line is what must pass for **`audio`**.

For transformer text classification, use the dedicated smoke path after the Windows dependencies are present:

```bash
make sunny-proof-text
make sunny-text-smoke
```

The validated text smoke uses GLUE/SST-2, `distilbert/distilbert-base-uncased`, `train_sample_limit=512`, and `val_sample_limit=128`.

## Usage

Operator login and day-to-day access details now live in:

- [Sunny Operator Runbook](../../docs/sunny-operator-runbook.md)

On Sunny WSL2:

```bash
bash ops/sunny/audit-demo-stack.sh
bash ops/sunny/reinit-demo-stack.sh
bash ops/sunny/training-mode.sh --dry-run
bash ops/sunny/training-mode.sh
bash ops/sunny/training-mode.sh --restore-demo
bash ops/sunny/prove-training-runtime.sh --with-training-mode --restore-demo
bash ops/sunny/vision-smoke.sh
bash ops/sunny/audio-smoke.sh
bash ops/sunny/text-smoke.sh
```

From another machine:

```bash
bash ops/sunny/run-remote-training-proof.sh --with-training-mode --restore-demo
bash ops/sunny/run-remote-training-proof.sh --with-training-mode --restore-demo --target audio
bash ops/sunny/run-remote-vision-smoke.sh
bash ops/sunny/run-remote-audio-smoke.sh
bash ops/sunny/run-remote-text-smoke.sh
```

If you want to run the Windows portion directly:

```powershell
powershell -ExecutionPolicy Bypass -File .\ops\sunny\reinit-demo-stack.ps1
```

## Assumptions

These scripts assume the current audited layout on Sunny:

- WSL2 services are managed by `systemd`
- Windows-side Python uses `py -3.11`
- `llama-server.exe` lives under `SUNNY_LLAMA_ROOT\build\bin\`
- model weights live under `SUNNY_MODELS_ROOT`
- `voice-lab` lives under `SUNNY_VOICE_LAB_ROOT`
- `ocr_server.py` and `detect_server.py` live under `SUNNY_WINDOWS_SCRIPTS_ROOT`

If those paths change, update this directory first so the repo stays authoritative.

Public defaults use placeholder values. For a private lab checkout, copy `ops/sunny/lab.env.example` to `ops/sunny/lab.env` and set the real `SUNNY_*` variables there. That file is ignored by git.

## Mode Split

Sunny now has two explicit operating modes:

- demo mode
  - keeps the Windows GPU demo services online for telephony, OCR, detection, vision, and embeddings
  - use `reinit-demo-stack.sh`

- training mode
  - stops the Windows GPU demo services so the 4090 can be used for training
  - use `training-mode.sh`

The WSL2 service layer remains up in both modes. Training mode is specifically about freeing the Windows-side GPU consumers without disturbing the Linux-side phone, chat, tunnel, or database layer.

## Restore timeout and `summary.json`

`prove-training-runtime.sh` and `vision-smoke.sh` run `training-mode.sh --restore-demo` under `timeout(1)` (default **600** seconds, overridable with **`RESTORE_TIMEOUT_SECONDS`**). Model reload on the Windows demo can exceed that budget; `timeout` then exits **124** even though services become healthy shortly after.

When **`audit-after-restore`** passes, `summarize_report_status.py` sets **`all_commands_succeeded`** to **true** and adds **`restore_demo_note`** so a successful smoke or proof run is not marked failed purely because restore hit the wall-clock cap.

`prove-training-runtime.sh` invokes the summarizer with **`--proof`**, which makes **`wsl-ml-lab-check`** non-gating (expected red on Sunny today). Vision smoke does not use **`--proof`**; every captured step must pass.

## Why The Proof Scripts Exist

Local tests on a laptop are still useful for pure-Python logic and CLI regressions, but they do not prove Sunny's real training runtime.

Likewise, running `pytest` inside Sunny WSL2 is not the authoritative answer for CUDA training readiness. The authoritative checks are the report-producing Sunny flows:

- `run-remote-training-proof.sh` / `prove-training-runtime.sh`
- `run-remote-vision-smoke.sh` / `vision-smoke.sh`
- `run-remote-audio-smoke.sh` / `audio-smoke.sh`
- `run-remote-text-smoke.sh` / `text-smoke.sh`

The proof scripts exist to answer, from the actual host:

- whether Sunny WSL2 has the required Python and tooling
- whether Sunny Windows can import CUDA PyTorch
- whether the 4090 is actually free enough for training
- whether `ml-lab check` passes on the remote machine rather than only on a development laptop
