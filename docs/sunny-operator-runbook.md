# Sunny Operator Runbook

## Purpose

This is the practical runbook for a new operator who needs to:

- reach Sunny
- sync the repo to Sunny
- audit the current state
- free the 4090 for training
- run the currently proven Windows training path

Use this document for day-to-day operation. Use [Handoff Guide](docs/handoff-guide.md) for the broader project context.

## What Is Proven Right Now

- Sunny WSL2 is the control and ops layer
- Sunny Windows Python `3.11` is the current proven CUDA training path for `vision`, bounded `audio`, and bounded transformer `text` smoke runs
- Sunny WSL2 is **not** currently training-ready

Do not treat WSL2 as the active training runtime unless someone explicitly upgrades and re-proves it.

## Access Prerequisites

From an operator laptop or workstation, the current access route assumes:

- SSH key for Sunny WSL2 at `~/.ssh/id_ed25519`
- SSH key for the EC2 proxy at `~/.ssh/cc-proxy.pem`
- access to the AWS edge host `ubuntu@54.243.75.156`
- Sunny reachable through the WireGuard address `10.200.0.2`

If any of those assumptions change, update this repo first.

## Preferred Access Path

If you do not need an interactive shell yet, use the repo-owned remote proof command first:

```bash
make sunny-proof
```

That is a safe first action because it:

1. syncs the repo to Sunny
2. proves the real Sunny runtime
3. pulls the report back locally

It is better than guessing whether Sunny is ready.

## Manual Login To Sunny WSL2

Use this exact command from another machine:

```bash
ssh -o ProxyCommand="ssh -i ~/.ssh/cc-proxy.pem -W %h:%p ubuntu@54.243.75.156" \
  -o ConnectTimeout=8 \
  -i ~/.ssh/id_ed25519 \
  sasha@10.200.0.2
```

Once connected, the repo path on Sunny is:

```bash
cd /home/sasha/Practice
```

## Sync The Repo To Sunny

Preferred:

```bash
bash ops/sunny/run-remote-training-proof.sh --skip-sync
```

Use that only if the repo is already up to date on Sunny.

Manual sync from another machine:

```bash
rsync -az \
  --delete \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude 'artifacts/' \
  --exclude 'data/' \
  /path/to/Practice/ \
  -e "ssh -o 'ProxyCommand=ssh -i ~/.ssh/cc-proxy.pem -W %h:%p ubuntu@54.243.75.156' -i ~/.ssh/id_ed25519" \
  sasha@10.200.0.2:/home/sasha/Practice/
```

If you do not need fine-grained control, `make sunny-proof` is the better default because it syncs and proves in one step.

## Audit Sunny Before Touching Anything

From Sunny WSL2:

```bash
cd /home/sasha/Practice
bash ops/sunny/audit-demo-stack.sh
```

This tells you:

- whether the WSL service layer is healthy
- whether the Windows demo layer is healthy
- how much 4090 VRAM is currently in use

## Free The 4090 For Training

From Sunny WSL2:

```bash
cd /home/sasha/Practice
bash ops/sunny/training-mode.sh --dry-run
bash ops/sunny/training-mode.sh
```

What this should do:

- keep WSL services running
- stop the Windows demo GPU services
- print a Sunny GPU preflight summary

After training, restore demo mode:

```bash
bash ops/sunny/training-mode.sh --restore-demo
```

## Prove Sunny Runtime From The Real Host

From Sunny WSL2:

```bash
cd /home/sasha/Practice
bash ops/sunny/prove-training-runtime.sh --with-training-mode --restore-demo
```

From another machine:

```bash
cd /path/to/Practice
bash ops/sunny/run-remote-training-proof.sh --with-training-mode --restore-demo
```

Reports are written under:

- on Sunny: `artifacts/sunny-reports/<timestamp>/`
- on the calling machine: `artifacts/sunny-reports/<timestamp>/`

Look at these first:

- `windows-ml-lab-check.txt`
- `windows-torch-cuda.txt`
- `wsl-ml-lab-check.txt`
- `run.log`

## Regression Tests Vs Runtime Proof

Do not confuse repo regression tests with Sunny runtime validation.

Use `pytest` or `make test` for:

- pure-Python regression coverage
- CLI behavior that does not depend on the real Sunny Windows CUDA runtime
- general repo sanity on a development machine

Do **not** use `pytest` on Sunny WSL2 as the authoritative answer to "can Sunny train right now?"

For Sunny host validation, use:

- `make sunny-proof` to prove the real remote runtime and pull back a report
- `make sunny-proof-text` to prove Windows text-transformer dependencies without running a full smoke train
- `make sunny-vision-smoke` to run the repo-owned Windows CUDA vision smoke train
- `make sunny-audio-smoke` to run the repo-owned bounded Windows CUDA audio smoke train
- `make sunny-text-smoke` to run the repo-owned bounded Windows CUDA transformer text-classifier smoke train

If you are already logged into Sunny WSL2, the direct host-side commands are:

```bash
cd /home/sasha/Practice
bash ops/sunny/prove-training-runtime.sh --with-training-mode --restore-demo
bash ops/sunny/vision-smoke.sh
bash ops/sunny/audio-smoke.sh
bash ops/sunny/text-smoke.sh
```

Today, the authoritative training signal is the Windows `py -3.11` path inside those reports, not `pytest` in WSL2.

## Running The Proven Windows Path

The current proven training runtime is Windows Python `3.11`.

There are two practical ways to reach it.

### Option 1: Open Windows PowerShell On Sunny

If you are physically on Sunny or already operating in the Windows session, open PowerShell and use:

```powershell
Set-Location '\\wsl.localhost\Ubuntu\home\sasha\Practice'
$env:PYTHONPATH = '\\wsl.localhost\Ubuntu\home\sasha\Practice\src'
py -3.11 -m industry_ml_lab.cli check --target vision --device cuda --output-dir artifacts/windows-proof --json
```

This is the current known-good readiness command.

### Option 2: Invoke Windows PowerShell From WSL2

From Sunny WSL2:

```bash
cd /home/sasha/Practice
/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile
```

Then inside PowerShell:

```powershell
Set-Location '\\wsl.localhost\Ubuntu\home\sasha\Practice'
$env:PYTHONPATH = '\\wsl.localhost\Ubuntu\home\sasha\Practice\src'
py -3.11 -m industry_ml_lab.cli check --target vision --device cuda --output-dir artifacts/windows-proof --json
```

## Current Build / Install Reality

Be precise here:

- the repo-owned Windows `vision` smoke path is now proven
- the authoritative Sunny training path is Windows Python `3.11` plus CUDA
- Sunny WSL2 remains the control and ops layer, not the proven training runtime
- there is still **not yet** a standardized Windows environment bootstrap documented as fully generic beyond the current proven vision/audio path

So when someone asks “how do I build on Sunny?”, the honest answer today is:

- sync the repo
- use Windows Python `3.11`
- run the repo via `PYTHONPATH` from the Windows view of the WSL checkout
- use `make sunny-proof`, `make sunny-vision-smoke`, `make sunny-audio-smoke`, and, after dependency proof, `make sunny-text-smoke` as the operator checks

Do **not** tell people that WSL `uv sync`, `pytest`, or `make test` are the active training proof path. That is false today.

## Known-Good Vision Readiness Command

This command is proven on Sunny Windows:

```powershell
Set-Location '\\wsl.localhost\Ubuntu\home\sasha\Practice'
$env:PYTHONPATH = '\\wsl.localhost\Ubuntu\home\sasha\Practice\src'
py -3.11 -m industry_ml_lab.cli check --target vision --device cuda --output-dir artifacts/windows-proof --json
```

Expected result:

- `is_ready: true`
- CUDA device: `NVIDIA GeForce RTX 4090`
- free VRAM around `22.4 GB` when in training mode

## Proven Vision Smoke Path

The repo-owned Windows vision smoke train is now proven on Sunny.

Preferred from another machine:

```bash
cd /path/to/Practice
make sunny-vision-smoke
```

Directly from Sunny WSL2:

```bash
cd /home/sasha/Practice
bash ops/sunny/vision-smoke.sh
```

Expected result:

- one epoch of CIFAR-10 training completes on the 4090
- the report directory contains `vision-artifacts/metrics.json` and `vision-artifacts/model.pt`
- `summary.json` reports overall success
- `restore-demo` may hit a wall-clock timeout, but the run is still treated as successful when `audit-after-restore` passes

## Proven Audio Smoke Path

The repo-owned Windows audio smoke train is now proven on Sunny as a bounded smoke workload.

Preferred from another machine:

```bash
cd /path/to/Practice
make sunny-audio-smoke
```

Directly from Sunny WSL2:

```bash
cd /home/sasha/Practice
bash ops/sunny/audio-smoke.sh
```

Expected result:

- one bounded SpeechCommands training pass completes on the 4090
- the report directory contains `audio-artifacts/metrics.json` and `audio-artifacts/model.pt`
- the nested Windows audio smoke summary is green
- the dataset is staged to a Windows-local cache if the source path comes from `\\wsl.localhost\...`
- `restore-demo` may still hit a wall-clock timeout, but the run is treated as successful when `audit-after-restore` passes

Current smoke settings are intentionally bounded for operator speed, not benchmark realism:

- `train_sample_limit = 1024`
- `val_sample_limit = 256`

## Proven Transformer Text Smoke Path

The repo-owned Windows transformer text smoke train is now proven on Sunny as a bounded smoke workload.

Preferred from another machine:

```bash
cd /path/to/Practice
make sunny-text-smoke
```

Directly from Sunny WSL2:

```bash
cd /home/sasha/Practice
bash ops/sunny/text-smoke.sh
```

Expected result:

- one bounded GLUE/SST-2 DistilBERT fine-tune completes on the 4090
- the report directory contains `text-artifacts/metrics.json` and a Hugging Face `text-artifacts/model/` directory
- the nested Windows text smoke summary is green
- Hugging Face dataset and model caches are placed under Windows-local `%LOCALAPPDATA%\industry-ml-lab\...` paths
- `restore-demo` may still hit a wall-clock timeout, but the run is treated as successful when `audit-after-restore` passes

Current smoke settings are intentionally bounded for operator speed, not benchmark realism:

- `train_sample_limit = 512`
- `val_sample_limit = 128`

## What To Hand Back After Any Change

After operating on Sunny, the minimum useful artifact is:

- one audit log or proof report directory
- a short note on what changed
- whether demo mode was restored successfully

Do not hand back “it seems fine.” Hand back a report path.
