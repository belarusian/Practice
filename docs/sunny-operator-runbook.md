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
- Sunny Windows Python `3.11` is the current proven CUDA training path for `vision`
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

- there is **not yet** a single repo-owned Windows training smoke script
- there is **not yet** a standardized Windows environment bootstrap documented as fully proven
- what is proven is the repo-based `vision` readiness path on Windows Python `3.11`

So when someone asks “how do I build on Sunny?”, the honest answer today is:

- sync the repo
- use Windows Python `3.11`
- run the repo via `PYTHONPATH` from the Windows view of the WSL checkout
- prove readiness first

Do **not** tell people that WSL `uv sync` is the active training path. That is false today.

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

## Next Unproven Command

The next thing the new operator should add and prove is a real Windows vision smoke train.

The intended next command shape is:

```powershell
Set-Location '\\wsl.localhost\Ubuntu\home\sasha\Practice'
$env:PYTHONPATH = '\\wsl.localhost\Ubuntu\home\sasha\Practice\src'
py -3.11 -m industry_ml_lab.cli train-vision --epochs 1 --device cuda --output-dir artifacts/vision-smoke
```

Important:

- this is the **next proof target**
- do not describe it as already proven until the repo contains the log and resulting artifact

## Audio Caveat

The current Windows proof showed:

- `torch: true`
- `torchvision: true`
- `torchaudio: false`

So:

- `vision` is the current proven path
- `audio` still needs environment work and proof on Windows

## What To Hand Back After Any Change

After operating on Sunny, the minimum useful artifact is:

- one audit log or proof report directory
- a short note on what changed
- whether demo mode was restored successfully

Do not hand back “it seems fine.” Hand back a report path.
