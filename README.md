# Industry ML Lab

This repository is a low-cost training bed for practicing the work behind production ML systems, not just model demos. It is designed to help you build muscle in PyTorch training, experiment management, active learning, embedding retrieval, model serving, and AWS deployment without immediately committing to a large managed-services bill.

## What This Covers

The lab is structured to map directly onto the capability profile you described:

- Image classification and computer vision with PyTorch
- Audio classification to build voice and audio intuition
- Dataset curation and active-learning style relabel queues
- Embedding-based retrieval with a path to `pgvector`
- Model serving with REST APIs, batching, artifact management, and health checks
- Workflow orchestration patterns for training and promotion
- AWS deployment decisions with explicit cost guardrails

## Architecture

The stack is intentionally phased:

- Local development: `uv`, Python package, sample data, pure-Python retrieval and labeling tools
- Training: PyTorch baselines for vision and audio
- Serving: FastAPI inference service for the vision model plus embedding search
- State: Postgres/pgvector, Valkey, and MinIO via `docker-compose.yml`
- Orchestration: Temporal worker and workflow stubs for training and promotion flows
- AWS: ephemeral GPU training on EC2 Spot, cheap CPU serving on EC2, S3 for artifacts, ECR for images

This is not optimized for maximum abstraction. It is optimized for learning the industry shape of the system while keeping the bill controlled.

## Quick Start

```bash
# Check vision-training readiness before training
uv run ml-lab check --target vision --verbose

# Build a local embedding index
uv run ml-lab build-index --records sample-data/embedding-records.jsonl --output artifacts/demo-index.json

# Search the index
uv run ml-lab search-index --index-path artifacts/demo-index.json --vector 0.92,0.08,0.04

# Generate an active learning relabel queue
uv run ml-lab active-learning-report --predictions sample-data/predictions.jsonl --output artifacts/relabel-queue.csv

# Train vision model (device automatically selected: cuda/mps/cpu)
uv run ml-lab train-vision --output-dir artifacts/vision-baseline
```

If you want the full API, workflow, vector, and model training flows, run `make sync` to install the full stack.

## Dependency Profiles

The repo is split into extras so the runtime can stay smaller:

- `dev`: linting and tests
- `serving`: FastAPI, vision inference, and API runtime
- `training`: PyTorch, torchvision, and torchaudio for training jobs
- `vector`: Postgres and `pgvector` clients
- `workflow`: Temporal client and worker runtime

Examples:

```bash
uv sync --extra dev
uv sync --extra serving --extra training
make sync
```

## Practice Tracks

1. Vision baseline
Train and fine-tune a ResNet classifier, track metrics, export artifacts, and serve predictions.

2. Audio baseline
Train a keyword-spotting style classifier on Speech Commands to cover audio pipelines and debugging.

3. Active learning
Score model uncertainty and generate a relabel queue from prediction outputs.

4. Retrieval
Build and query an embedding index locally, then swap the storage layer to Postgres with `pgvector`.

5. Orchestration
Wrap training and promotion steps in a Temporal workflow.

6. Deployment
Push model and API artifacts to AWS with a cost-aware topology.

## Recommended Learning Path

- Phase 1: Get the local CLI and demo data working
  - Run `make check` or `uv run ml-lab check --target vision --verbose`
  - Use the checklist to verify target-specific training deps and current device readiness
- Phase 2: Train the vision model, then wire the API to a real checkpoint
- Phase 3: Add the audio baseline and uncertainty-driven relabel loop
- Phase 4: Replace the local retrieval index with Postgres plus `pgvector`
- Phase 5: Containerize and deploy to AWS EC2 and S3
- Phase 6: Add a control plane in TypeScript or NestJS if you want direct exposure to the rest of the target stack

## Key Documents

- [Architecture](docs/architecture.md)
- [AWS Costs](docs/aws-costs.md)
- [Roadmap](docs/roadmap.md)
- [Training Checklist](docs/checklist.md)
- [Checklist Quick Reference](docs/checklist-quickref.md)
- [Current State](docs/current-state.md)
- [Sunny Operator Runbook](docs/sunny-operator-runbook.md)
- [Handoff Guide](docs/handoff-guide.md)
- [Demo Stack](docs/demo-stack.md)
- [Operating Model](docs/operating-model.md)
- [AWS Deployment Notes](infra/aws/README.md)
- [Sunny Ops](ops/sunny/README.md)

## Machine Split

- Sunny: CUDA training box and live demo host
- Mac Studio: large-memory local inference box
- AWS: thin public edge plus selective rehearsal environment

On Sunny, use:

- `bash ops/sunny/reinit-demo-stack.sh` for demo mode
- `bash ops/sunny/training-mode.sh --dry-run` to inspect what training mode would stop
- `bash ops/sunny/training-mode.sh` to free the 4090 for training
- `bash ops/sunny/training-mode.sh --restore-demo` to bring the demo GPU services back
- `bash ops/sunny/prove-training-runtime.sh --with-training-mode --restore-demo` to produce a host-local report of WSL and Windows training readiness
- `bash ops/sunny/vision-smoke.sh` to run the repo-owned Windows vision smoke training flow from Sunny WSL2

This mode split has already been validated on Sunny: the live demo stack used about `23.7 GiB` to `23.9 GiB` of 4090 VRAM, `training-mode.sh` reduced that to about `1.1 GiB` used with `23.5 GiB` free, and restore returned the demo services to healthy status.

From a separate machine, use `bash ops/sunny/run-remote-training-proof.sh --with-training-mode --restore-demo` to sync the repo to Sunny, run the proof there, and pull the report back under `artifacts/sunny-reports/`.

For the current repo-owned Windows training smoke loop from another machine, use `bash ops/sunny/run-remote-vision-smoke.sh`.

Current proven state on Sunny:

- WSL2 is healthy for services and ops, but not yet training-ready
- Windows Python `3.11` is the currently proven CUDA training path for this repo

## AWS Workflow

The repo now includes a concrete AWS bootstrap path:

1. Copy `infra/aws/lab.env.example` to `infra/aws/lab.env` and fill in your subnet, security group, and notification email.
2. Run `make aws-bootstrap` to create the S3 bucket and ECR repository.
3. Run `make aws-budget` to put a monthly budget and alert threshold in place.
4. Run `make aws-launch-trainer` to start an ephemeral Spot GPU trainer.
5. Run `make aws-upload-artifacts` after local or remote training to sync artifacts to S3.
6. Run `make aws-build-and-push-api` to publish the API container to ECR.
7. Run `make aws-launch-api` to launch a CPU EC2 instance that pulls the container and serves the model.

The AWS scripts live under [infra/aws/scripts](/Users/av4nda/Practice/infra/aws/scripts:1) and assume `us-east-1` by default.
