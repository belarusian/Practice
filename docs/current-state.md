# Current State

This document is the processed, validated snapshot of where the repo and the live machine stack stand right now.

It combines:

- the repo-backed work already present in this repository
- the reported updates from collaborators
- direct validation performed against the repo and live machines

The goal is to separate verified state from reported-but-not-yet-validated claims.

## Validated Repo State

The ML lab scaffold is present and structurally intact.

Verified locally:

- Python package under `src/industry_ml_lab`
- CLI entrypoint `ml-lab`
- vision training baseline
- audio training baseline
- transformer text-classification baseline
- active-learning scoring and relabel queue generation
- local embedding index and search
- FastAPI serving layer
- Temporal workflow stubs
- `docker-compose.yml` for Postgres plus `pgvector`, Valkey, and MinIO
- AWS bootstrap and deployment scripts under `infra/aws/scripts`

Validation performed:

- `PYTHONPATH=src pytest -q` (`76 passed, 4 skipped` on this branch)
- `python3 -m compileall src`
- `python3 -m compileall tests`
- `PYTHONPATH=src python3 -m industry_ml_lab --help`
- `bash -n infra/aws/scripts/*.sh`

Status:

- repo codebase is syntactically healthy
- most of the lightweight regression coverage passes on this machine
- pure-Python and CLI surfaces are present
- the transformer text-classification path is implemented locally but not yet runtime-proven on Sunny
- heavyweight runtime paths are still only partially runtime-validated
- local `pytest` results are useful regression signals, but they are not the authoritative Sunny runtime proof path

## Validated Coder 1 Contributions

The following files exist in the worktree and were read directly:

- `docs/demo-stack.md`
- `docs/operating-model.md`
- `ops/sunny/README.md`
- `ops/sunny/audit-demo-stack.sh`
- `ops/sunny/reinit-demo-stack.sh`
- `ops/sunny/reinit-demo-stack.ps1`
- `tests/test_simple_index.py`
- `tests/test_active_learning.py`
- `tests/test_cli.py`
- `tests/test_serving_api.py`

Validation performed:

- `bash -n ops/sunny/*.sh`
- `PYTHONPATH=src pytest -q` resulting in `1 failed, 70 passed, 4 skipped`

What is verified:

- the demo stack has now been documented as distinct from the training lab
- the repo operating model is documented
- Sunny now has repo-backed audit and re-init entrypoints
- the first regression-test layer for the pure-Python and CLI paths is present
- serving API tests exist and skip cleanly when optional deps are not installed
- the bash wrappers are syntactically valid

What is not yet verified:

- the WSL wrapper `reinit-demo-stack.sh` has not yet been exercised end to end from the repo checkout
- the current checklist test suite still has one failing assertion in `tests/test_checklist.py` that expects a different error message than the one returned in this lite environment

## Validated Guy 3 Contributions

The following files exist in the worktree and were read directly:

- `ops/sunny/training-mode.sh`
- `ops/sunny/training-mode.ps1`

Validation performed:

- `bash -n ops/sunny/training-mode.sh`
- runtime execution on Sunny from `~/Practice`

What is verified:

- `training-mode.sh --dry-run` identifies the exact Windows demo GPU ports that would be drained: `8013`, `8082`, `8083`, `9001`, `9002`, `9003`, and `9004`
- the live training-mode stop path works on Sunny without disturbing the WSL2 service layer
- the live restore path works and returns Sunny to demo mode
- the repo now owns the main operational seam between "public demo host" and "local training box"

## Validated Live Machine State

### Mac Studio

Direct SSH works.

Validated:

- host is reachable at `10.106.1.184`
- hostname: `SashasMacStudio.mynetworksettings.com`
- live listener observed: `llama-server` on `:8080`

Correction (applied in repo):

- `docs/demo-stack.md` previously listed the Mac Studio LLM on `8013`; live check observed `llama-server` on `:8080`
- `docs/demo-stack.md` now lists Mac Studio at `:8080` and notes the distinction from Sunny Windows `:8013`

### Sunny WSL2

Sunny is reachable through the EC2 plus WireGuard path at `10.200.0.2`.

Validated:

- hostname: `Sunny`
- WSL memory is lightly used, about `700 MiB` used and `14 GiB` available
- listeners observed on WSL:
  - `:22`
  - `:8008`
  - `:8080`
  - `:8765`
- the repo was synced to `~/Practice` on Sunny and `bash ops/sunny/audit-demo-stack.sh` was run there successfully
- WSL services confirmed active in audit:
  - `ssh`
  - `nginx`
  - `postgresql`
  - `wg-quick@wg0`
  - `coturn`
  - `synapse`
  - `voice-phone.service`
- WSL health checks all passed:
  - Synapse `200`
  - Element/nginx `200`
  - phone webhook `200`
- WireGuard was healthy during audit with a fresh handshake to the EC2 peer
- `uv` is present in WSL at `/home/sasha/.local/bin/uv`
- the remote Sunny proof run reported WSL Python as `3.10.12`
- the remote Sunny proof run reported no WSL training modules installed:
  - `torch`: `false`
  - `torchvision`: `false`
  - `torchaudio`: `false`
- the repo-based WSL check failed as expected for the real host state:
  - Python too old
  - training deps missing
  - CUDA path not usable through local PyTorch in WSL

Interpretation:

- WSL2 itself is alive and not resource constrained
- the Linux service layer is not the current training bottleneck
- WSL2 is currently the service and ops layer on Sunny, not the proven training runtime

### Sunny Windows GPU State

Validated on the live machine:

- GPU: `NVIDIA GeForce RTX 4090`
- total VRAM: `24564 MiB`
- demo-mode audit showed `23697 MiB` used and `0%` utilization before training-mode stop
- post-restore audit showed `23877 MiB` used with the demo layer back online

Interpretation:

- the 4090 is heavily occupied in demo mode even when utilization is low
- memory pressure, not compute utilization, is the main local training constraint on Sunny

### Sunny Windows Demo Layer

Validated through the repo-backed audit and post-restore audit:

- `gpt-oss-20b` on `:8013` healthy
- `qwen3-vl` on `:8082` healthy
- `qwen3-embedding` on `:8083` healthy
- `stt-server` on `:9001` healthy
- `tts-server` on `:9002` healthy
- `ocr-server` on `:9003` healthy
- `detect-server` on `:9004` healthy

Interpretation:

- the demo layer is currently healthy
- that health is also the reason the 4090 is not free for training

### Sunny Training Mode

Repo-backed training-mode entrypoints now exist:

- `ops/sunny/training-mode.sh`
- `ops/sunny/training-mode.ps1`

Validated behavior from the live Sunny run:

- dry-run enumerated the current listener PIDs for the seven Windows GPU demo services without changing process state
- the live stop run terminated those listeners and reduced GPU memory usage from `23697 MiB` to `1072 MiB`
- the post-stop preflight reported:
  - `total_vram_mb=24564`
  - `used_vram_mb=1072`
  - `free_vram_mb=23492`
  - `gpu_util_percent=0`
  - `STATUS: Sunny GPU preflight passed`
- `--restore-demo` successfully delegated to the existing Windows re-init path and the follow-up audit showed the demo services healthy again

Interpretation:

- Sunny can now be flipped intentionally between demo mode and training mode from the repo checkout
- the mode switch problem is solved

### Sunny Remote Training Proof

Repo-backed remote proof entrypoints now exist:

- `ops/sunny/prove-training-runtime.sh`
- `ops/sunny/prove-training-runtime.ps1`
- `ops/sunny/run-remote-training-proof.sh`

Validated on Sunny from a laptop-triggered remote run:

- demo mode started with the 4090 heavily occupied, around `22.6 GiB` to `23.7 GiB` used
- training mode reduced GPU usage to about `962 MiB` to `966 MiB`
- the WSL proof showed:
  - `Python 3.10.12`
  - `uv` present
  - no `torch`, `torchvision`, or `torchaudio`
  - `ml-lab check --target vision --device cuda --json` failed with readiness errors
- the Windows proof showed:
  - `py -3.11` available
  - `nvidia-smi` available
  - CUDA PyTorch import and probe working
  - repo-based `ml-lab check --target vision --device cuda --json` passed with `is_ready: true`
  - free VRAM reported around `22.4 GiB`
- an independent follow-up audit confirmed the demo stack healthy again after the remote proof run:
  - WSL services healthy
  - Windows demo ports healthy
  - 4090 back to about `23.9 GiB` used in demo mode

Interpretation:

- the current proven Sunny training path is Windows Python `3.11` with CUDA, not WSL2
- WSL2 remains the right place for repo-backed ops, audit, and mode switching
- if we want Linux-native training on Sunny later, that is a separate environment-setup task, not an open question about current readiness

### Sunny Vision Smoke (repo-owned Windows train)

Validated **2026-04-20** on Sunny from a laptop-triggered run (`make sunny-vision-smoke`):

- `ops/sunny/vision-smoke.sh` stopped the Windows demo GPU services, then ran `vision-smoke.ps1` on Windows Python `3.11` with CUDA
- one epoch of CIFAR-10 vision training completed on the 4090 (`duration_seconds` about `97.5`, `best_val_accuracy` about `0.636`)
- training wrote `metrics.json` and `model.pt` under the report’s `vision-artifacts/` directory
- `summary.json` treats `restore-demo` exit **124** (wall-clock timeout) as consistent with success when **`audit-after-restore`** passes and other steps succeeded

Interpretation:

- the repo-owned **train-vision** path is now runtime-proven on Sunny’s Windows CUDA stack, not only `ml-lab check`
- WSL remains ops and mode-switch only for training

### Sunny Audio Smoke (repo-owned Windows train)

Validated **2026-04-20** on Sunny from a laptop-triggered run (`make sunny-audio-smoke`):

- `ops/sunny/audio-smoke.sh` stopped the Windows demo GPU services, then ran `audio-smoke.ps1` on Windows Python `3.11` with CUDA
- the Windows runtime now has a working `torchaudio` loader backend (`soundfile`) and stages `data/audio` off the `\\wsl.localhost\...` path into a Windows-local cache for training I/O
- the smoke path uses a bounded training slice (`train_sample_limit=1024`, `val_sample_limit=256`) so it proves the end-to-end audio train path without turning the smoke run into a full-dataset benchmark
- one epoch of SpeechCommands audio training completed on the 4090 and wrote `metrics.json` plus `model.pt` under the report’s `audio-artifacts/` directory
- the validated green run reported:
  - `windows-train-audio` exit `0`
  - `windows-metrics-json` exit `0`
  - `windows-audio-summary-check` exit `0`
  - `duration_seconds` about `1.6`

Interpretation:

- the repo-owned **train-audio** path is now runtime-proven on Sunny’s Windows CUDA stack for bounded smoke training, not only `ml-lab check --target audio`
- WSL remains the ops and mode-switch layer; Windows remains the active trainer
- audio smoke is now suitable as a rehearsal gate, not as a benchmark of full SpeechCommands training throughput

### Sunny Transformer Text Smoke (repo-owned Windows train)

Validated **2026-04-21** on Sunny from a laptop-triggered run (`make sunny-text-smoke`):

- `ml-lab train-text-classifier` fine-tunes a Hugging Face sequence-classification model with a native PyTorch training loop
- default task is GLUE/SST-2 using `distilbert/distilbert-base-uncased`
- the `text` checklist target validates `torch`, `transformers`, and `datasets`
- bounded sample limits are available for smoke-sized runs: `--train-sample-limit` and `--val-sample-limit`
- artifacts are written as a Hugging Face `save_pretrained()` model directory plus `metrics.json` and `model_meta.json`
- the Windows runtime has CUDA PyTorch plus transformer deps:
  - `torch`: `2.5.1+cu121`
  - CUDA available: `true`
  - `transformers`: `4.57.6`
  - `datasets`: `4.8.4`
- `ops/sunny/text-smoke.sh` stopped the Windows demo GPU services, then ran `text-smoke.ps1` on Windows Python `3.11` with CUDA
- the script used Windows-local Hugging Face caches under `%LOCALAPPDATA%\industry-ml-lab\...` instead of treating WSL UNC paths as the active dataset/model cache
- one bounded GLUE/SST-2 DistilBERT fine-tune completed on the 4090 and wrote `metrics.json`, `model_meta.json`, and a Hugging Face model directory under the report’s `text-artifacts/` directory
- the validated green run reported:
  - `windows-train-text` exit `0`
  - `windows-metrics-json` exit `0`
  - `windows-text-summary-check` exit `0`
  - `duration_seconds` about `12.0`
  - `best_val_accuracy` about `0.602`

Validation performed locally:

- `PYTHONPATH=src pytest tests/test_checklist.py tests/test_cli.py -q` -> `70 passed`
- `PYTHONPATH=src pytest -q` -> `76 passed, 4 skipped`
- `python3 -m compileall src tests` -> passed

Interpretation:

- the repo-owned **train-text-classifier** path is now runtime-proven on Sunny’s Windows CUDA stack for bounded smoke training, not only implemented locally
- this is the first transformer training path in the repo
- this is still not LLM fine-tuning; it is encoder-style transformer classification for tagging/content-understanding practice

## Processed View Of Coder 2 Report

Coder 2's repository analysis is directionally correct and consistent with the codebase.

Validated as accurate:

- the repo is a training bed for production-style ML systems
- the major subsystems listed in the report are present
- the architecture description matches the actual repo structure
- the stated gaps are real

Important nuance:

- the repo is structurally sound, but several areas are still scaffold-level rather than end-to-end operational
- "exists in the repo" and "runtime-proven in production-like conditions" are not the same thing

## Processed View Of Ops Collaborator Input

The latest collaborator input is aligned with the current repo direction.

What it adds:

- tighten `ops/sunny/*` rather than expanding into unrelated surface area
- keep `docs/demo-stack.md` synchronized with observed audit output
- treat the WSL2 to PowerShell handoff as a first-class operational seam
- use pasted logs and diffs from Sunny as the basis for reconciliation instead of relying on memory

Assessment:

- this is consistent with the repo-backed operating model already documented in `docs/operating-model.md`
- it is the right support role for the current phase
- it does not replace runtime validation on Sunny, but it does define a practical collaboration loop once those logs exist

Operational implication:

- after each `audit-demo-stack.sh` or `reinit-demo-stack.sh` run on Sunny, the resulting logs and diffs should be treated as review artifacts
- doc updates about ports, commands, and health expectations should follow those artifacts, not guesswork

## Current Reality

There are now two parallel tracks in the same repository:

1. the ML practice stack
   - training code
   - serving code
   - active learning
   - retrieval
   - workflows
   - AWS rehearsal scripts

2. the demo-stack operational memory
   - live service inventory
   - Sunny boot and audit scripts
   - repo-backed operating model

That split is good. It reflects the actual situation:

- one stack keeps the demos alive
- the other stack is where we build training and model-ops skill

## Constraints Right Now

The main immediate constraint is not AWS.

It is the remaining Sunny training-runtime decision:

- direct LAN SSH to Sunny still needs to be re-validated after the Windows port-forward refresh
- training sessions need a clear habit: stop demo GPU services first, train, then restore demo mode
- the remaining environment question is whether we want to upgrade Sunny WSL into a second valid training runtime, not whether Sunny can train at all

## Recommended Next Steps

1. Re-validate direct LAN SSH access to Sunny after the elevated Windows port-forward refresh.
2. Tighten the remaining restore-tail behavior in the remote smoke wrappers so the outer run finalizes as cleanly as the nested Windows report.
3. Decide explicitly whether WSL2 should stay service-only or be upgraded later to Python `3.11` plus training deps.
4. Treat each training session as a mode transition:
   - `bash ops/sunny/training-mode.sh`
   - run training
   - `bash ops/sunny/training-mode.sh --restore-demo`
5. Keep using `audit-demo-stack.sh` output and `artifacts/sunny-reports/*` proof reports as the source artifacts for doc updates about Sunny ports, health, and training readiness.
6. Start the embedding-transformer retrieval slice after the text smoke branch is merged.

## State Summary

Verified:

- repo scaffold exists and passes basic syntax checks
- the current lightweight test layer passes locally
- repo-backed Sunny ops files exist
- Mac Studio is reachable
- Sunny WSL2 is reachable and healthy through the tunnel
- the repo-backed Sunny audit runs successfully from `~/Practice`
- the Windows demo layer is healthy on Sunny
- Sunny training mode can intentionally drain the Windows demo GPU services
- Sunny's 4090 can be brought from demo-mode pressure down to training-ready free VRAM and then restored to demo mode
- Sunny WSL2 is not currently training-ready
- Sunny Windows Python `3.11` is the currently proven CUDA training path
- repo-owned vision smoke training (`make sunny-vision-smoke`) has completed end-to-end on Sunny with real metrics and `model.pt` artifacts (2026-04-20)
- repo-owned bounded audio smoke training (`make sunny-audio-smoke`) has completed end-to-end on Sunny with real metrics and `model.pt` artifacts (2026-04-20)
- repo-owned bounded transformer text-classifier smoke training (`make sunny-text-smoke`) has completed end-to-end on Sunny with real metrics and Hugging Face model artifacts (2026-04-21)

Not yet verified:

- direct LAN SSH path to Sunny
- full serving and workflow execution paths for the ML lab
