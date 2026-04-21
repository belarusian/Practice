# Handoff Guide

## Purpose

This document is for the next operator or team taking over the repo.

It answers four questions:

1. what this project is trying to do
2. what is already proven versus only scaffolded
3. how to operate the current machines safely
4. where to take the work next without reopening solved questions

Read this first, then use the linked documents for detail.

## What This Repo Is

This repo is a practice bed for production-style ML work.

It is not just a model demo repo. The intended learning surface is:

- dataset handling
- PyTorch training
- experiment and artifact flow
- active learning and relabel queues
- model serving
- retrieval
- workflow orchestration
- cost-aware cloud deployment

The live demo stack and the training lab are related, but they are not the same thing.

## Current Proven Machine Roles

### Sunny

- primary local CUDA box
- live demo host
- machine with the repo-backed mode switch between `demo mode` and `training mode`

Current proven split on Sunny:

- WSL2
  - use for ops, audit, repo-backed scripts, and mode switching
  - **not currently training-ready**

- Windows Python `3.11`
  - use for actual CUDA training work right now
  - **currently proven vision-training runtime**

### Mac Studio

- large-memory local inference machine
- not the current CUDA training target

### AWS

- thin public edge today
- later training and deployment rehearsal environment
- not required for day-one study work

## Documents To Trust

Start here for the current repo-backed truth:

- [Current State](docs/current-state.md)
- [Sunny Operator Runbook](docs/sunny-operator-runbook.md)
- [Sunny Ops](../ops/sunny/README.md)
- [Operating Model](docs/operating-model.md)
- [Demo Stack](docs/demo-stack.md)
- [Architecture](docs/architecture.md)

Use `Current State` for validated facts, not chat summaries.

## Operating Model

The repo is supposed to be the shared operational memory.

That means:

- boot logic belongs in the repo
- health checks belong in the repo
- machine-role assumptions belong in the repo
- recovery steps belong in the repo
- proof artifacts should be captured from the real machine, not described from memory

Treat manual file copy or hand-run commands as temporary staging steps, not the final operating model.

## The Core Decision That Is Already Made

Do not spend time debating whether Sunny WSL2 or Sunny Windows should be the training runtime.

That has already been answered by the repo-owned proof run:

- Sunny WSL2 failed the repo-based readiness check
- Sunny Windows passed the repo-based readiness check for the `vision` target on `cuda`

So the current rule is:

- use WSL2 to control and audit Sunny
- use Windows Python `3.11` to train on Sunny

If someone wants Linux-native training later, treat that as a new environment project, not as an unresolved assumption.

## Common Workflows

### 1. Audit Sunny Safely

From Sunny WSL2:

```bash
bash ops/sunny/audit-demo-stack.sh
```

Use this when you want to know:

- whether WSL services are healthy
- whether Windows demo services are healthy
- how much VRAM the 4090 is currently using

### 2. Free The 4090 For Training

From Sunny WSL2:

```bash
bash ops/sunny/training-mode.sh --dry-run
bash ops/sunny/training-mode.sh
```

Use `--dry-run` first when you want to see which Windows GPU services will be stopped.

Training mode is supposed to:

- keep WSL services up
- stop Windows GPU demo services
- verify that GPU memory is actually free enough for training

### 3. Restore Demo Mode

From Sunny WSL2:

```bash
bash ops/sunny/training-mode.sh --restore-demo
```

Use this after training so the demo surface comes back.

### 4. Prove Sunny Runtime From Sunny

From Sunny WSL2:

```bash
bash ops/sunny/prove-training-runtime.sh --with-training-mode --restore-demo
```

This captures a host-local report under `artifacts/sunny-reports/<timestamp>/`.

It proves:

- WSL Python and module state
- Windows Python and CUDA state
- repo-based `ml-lab check` results from the real host

### 5. Prove Sunny Runtime From Another Machine

From a laptop or another operator machine:

```bash
bash ops/sunny/run-remote-training-proof.sh --with-training-mode --restore-demo
```

This does three things:

1. syncs the repo to Sunny
2. runs the proof on Sunny
3. pulls the report back locally

This is the preferred proof path for handoff and review because it creates a local copy of the real-machine evidence.

## What Has Already Been Proven

The following are no longer open questions:

- Sunny demo mode is healthy
- Sunny training mode can free the 4090
- Sunny Windows is the currently proven CUDA training path for `vision`
- Sunny WSL2 is currently not training-ready
- the demo stack can be restored after the mode switch

Evidence exists in:

- `artifacts/sunny-reports/<timestamp>/windows-ml-lab-check.txt`
- `artifacts/sunny-reports/<timestamp>/windows-torch-cuda.txt`
- `artifacts/sunny-reports/<timestamp>/wsl-ml-lab-check.txt`

## What Is Still Only Partially Proven

These are the real remaining gaps:

- actual end-to-end training run on Sunny Windows from this repo
- audio training readiness on Sunny Windows
- serving path with optional serving deps installed on at least one machine
- full workflow execution
- AWS training rehearsal

Those are the right next questions. Do not reopen the already-proven ones.

## What To Do Next

### Priority 1: Add A Repo-Owned Windows Training Smoke Path

Goal:

- run a real `vision` smoke training job on Sunny Windows from this repo
- capture stdout, exit code, artifact paths, and timing

Why this is next:

- readiness is proven
- the next missing proof is a real training run

Recommended deliverable:

- one repo-owned script under `ops/sunny/` that:
  - flips Sunny into training mode
  - invokes Windows Python `3.11` against the repo
  - runs a one-epoch `train-vision` smoke job
  - writes logs to `artifacts/sunny-reports/<timestamp>/`
  - restores demo mode

### Priority 2: Prove The Audio Path On Sunny Windows

Current state:

- `vision` is proven ready
- `torchaudio` is not currently proven on the Windows path

Goal:

- install or verify `torchaudio`
- run the repo-based `audio` readiness check
- then run a small audio smoke training job

### Priority 3: Activate Serving Coverage On One Real Machine

Current state:

- serving tests exist
- they skip when the `serving` extra is not installed

Goal:

- install the `serving` extra on one machine
- run the serving API tests there
- prove the FastAPI path with real optional deps

### Priority 4: Promote The Remote Proof Pattern Into The Default Review Loop

Goal:

- make every major Sunny change produce:
  - a report directory
  - an audit log
  - a clear pass or fail outcome

That prevents future handoff drift.

### Priority 5: Add AWS Only When Local Muscle Exists

Do not rush here.

Move to AWS after:

- local vision training is proven
- local audio path is either proven or explicitly deferred
- artifact flow is clear

Then use AWS for:

- Linux parity
- Spot GPU training
- artifact persistence
- deployment rehearsal

## What Not To Do

- Do not assume laptop tests prove Sunny.
- Do not assume WSL training matters just because Linux is closer to AWS.
- Do not treat chat summaries as evidence when repo-owned proof artifacts exist.
- Do not let host boot behavior drift out of the repo.
- Do not spend AWS money before the local training path is actually exercised.

## Quick Start For A New Operator

If you were taking this over today, the shortest correct sequence would be:

1. Read [Sunny Operator Runbook](docs/sunny-operator-runbook.md).
2. Run `bash ops/sunny/run-remote-training-proof.sh --with-training-mode --restore-demo`.
3. Inspect the newest `artifacts/sunny-reports/<timestamp>/`.
4. Accept that Sunny Windows is the training path for now.
5. Implement the Windows vision smoke-training runner.
6. Capture the next proof report from the real box.

That is the cleanest continuation of the project.
