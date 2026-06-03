# Industry ML Lab

This repository is a low-cost training bed for practicing the work behind production ML systems, not just model demos. It is designed to build hands-on experience with PyTorch training, experiment management, active learning, embedding retrieval, model serving, and AWS deployment without immediately committing to a large managed-services bill.

## Marcus Aurelius Philosopher Model

**Note**: Marcus Aurelius training is also available in the [Training repo](https://github.com/belarusian/training) (`training/marcus/`).

| Repo | Model | Approach | Use Case |
|------|-------|----------|----------|
| **Practice** | Qwen3-100M | Full fine-tuning with PyTorch | Experimentation, learning, tiny LLM |
| **Training** | Qwen3-4B | SFT + GRPO with Unsloth LoRA | Production ML pipeline, advanced RLHF |

**Why both?**
- **Practice repo**: Marcus Aurelius as a learning experiment (separate use case)
- **Training repo**: Effect TypeScript fine-tuning (primary focus)

See `src/industry_ml_lab/training/philosopher.py` for the Practice implementation.

## What This Covers

The lab is structured around production-oriented ML capabilities:

- Image classification and computer vision with PyTorch
- Audio classification to build voice and audio intuition
- Transformer-based text classification for tagging and content understanding
- Transformer-based text embeddings for semantic search
- Dataset curation and active-learning style relabel queues
- Embedding-based retrieval with a path to `pgvector`
- Model serving with REST APIs, batching, artifact management, and health checks
- Workflow orchestration patterns for training and promotion
- AWS deployment decisions with explicit cost guardrails

## Architecture

The stack is intentionally phased:

- Local development: `uv`, Python package, sample data, pure-Python retrieval and labeling tools
- Training: PyTorch baselines for vision and audio, plus a Hugging Face transformer text classifier
- Serving: FastAPI inference service for the vision model plus embedding search
- State: Postgres/pgvector, Valkey, and MinIO via `docker-compose.yml`
- Orchestration: Temporal worker and workflow stubs for training and promotion flows
- AWS: ephemeral GPU training on EC2 Spot, cheap CPU serving on EC2, S3 for artifacts, ECR for images

This is not optimized for maximum abstraction. It is optimized for learning the industry shape of the system while keeping the bill controlled.

## Quick Start

```bash
# Check vision-training readiness before training
uv run ml-lab check --target vision --verbose

# Build a local embedding index from precomputed vectors
uv run ml-lab build-index --records sample-data/embedding-records.jsonl --output artifacts/demo-index.json

# Search the index
uv run ml-lab search-index --index-path artifacts/demo-index.json --vector 0.92,0.08,0.04

# Build a transformer text embedding index
uv run ml-lab build-text-index --records sample-data/text-records.jsonl --output artifacts/text-index.json

# Search with a natural-language query
uv run ml-lab search-text-index --index-path artifacts/text-index.json --query "free CUDA memory for training"

# Generate an active learning relabel queue
uv run ml-lab active-learning-report --predictions sample-data/predictions.jsonl --output artifacts/relabel-queue.csv

# Train vision model (device automatically selected: cuda/mps/cpu)
uv run ml-lab train-vision --output-dir artifacts/vision-baseline

# Fine-tune a bounded DistilBERT text classifier smoke run
uv run ml-lab train-text-classifier --output-dir artifacts/text-baseline --train-sample-limit 512 --val-sample-limit 128
```

The transformer text commands require the `transformer` dependency profile.

To install the full API, workflow, vector, and model training stack, run `make sync`.

## Dependency Profiles

The repo is split into extras so the runtime can stay smaller:

- `dev`: linting and tests
- `serving`: FastAPI, vision inference, and API runtime
- `training`: PyTorch, torchvision, and torchaudio for training jobs
- `transformer`: Hugging Face `transformers` and `datasets` for text classification and text embeddings
- `vector`: Postgres and `pgvector` clients
- `workflow`: Temporal client and worker runtime

Examples:

```bash
uv sync --extra dev
uv sync --extra serving --extra training
uv sync --extra transformer
make sync
```

## Practice Tracks

1. Vision baseline
Train and fine-tune a ResNet classifier, track metrics, export artifacts, and serve predictions.

2. Audio baseline
Train a keyword-spotting style classifier on Speech Commands to cover audio pipelines and debugging.

3. Transformer text classifier
Fine-tune a DistilBERT-style sequence classifier on GLUE/SST-2 to practice tagging and content understanding.

4. Active learning
Score model uncertainty and generate a relabel queue from prediction outputs.

5. Retrieval
Build and query an embedding index locally from either precomputed vectors or transformer text embeddings, then swap the storage layer to Postgres with `pgvector`.

6. Orchestration
Wrap training and promotion steps in a Temporal workflow.

7. Deployment
Push model and API artifacts to AWS with a cost-aware topology.

8. Marcus Aurelius Philosopher (tiny LLM)
Fine-tune Qwen3-100M on Meditations Q&A pairs using standard PyTorch. Focus: learning full fine-tuning vs LoRA approaches.

## Recommended Learning Path

- Phase 1: Get the local CLI and demo data working
  - Run `make check` or `uv run ml-lab check --target vision --verbose`
  - Use the checklist to verify target-specific training deps and current device readiness
- Phase 2: Train the vision model, then wire the API to a real checkpoint
- Phase 3: Add the audio baseline and uncertainty-driven relabel loop
- Phase 4: Add transformer text classification, then build transformer text embeddings for retrieval
- Phase 5: Replace the local retrieval index with Postgres plus `pgvector`
- Phase 6: Containerize and deploy to AWS EC2 and S3
- Phase 7: Add a control plane in TypeScript or NestJS for full-stack model operations practice
- Phase 8: Marcus Aurelius philosopher model - practice full fine-tuning vs LoRA approaches

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
- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)

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
- `bash ops/sunny/text-smoke.sh` to run the repo-owned Windows transformer text-classifier smoke flow from Sunny WSL2
- `bash ops/sunny/embedding-smoke.sh` to run the repo-owned Windows transformer text-embedding retrieval smoke flow from Sunny WSL2

This mode split has already been validated on Sunny: the live demo stack used about `23.7 GiB` to `23.9 GiB` of 4090 VRAM, `training-mode.sh` reduced that to about `1.1 GiB` used with `23.5 GiB` free, and restore returned the demo services to healthy status.

From a separate machine, use `bash ops/sunny/run-remote-training-proof.sh --with-training-mode --restore-demo` to sync the repo to Sunny, run the proof there, and pull the report back under `artifacts/sunny-reports/`.

For the current repo-owned Windows smoke loops from another machine, use `bash ops/sunny/run-remote-vision-smoke.sh`, `bash ops/sunny/run-remote-audio-smoke.sh`, `bash ops/sunny/run-remote-text-smoke.sh`, or `bash ops/sunny/run-remote-embedding-smoke.sh`.

Current proven state on Sunny:

- WSL2 is healthy for services and ops, but not yet training-ready
- Windows Python `3.11` is the currently proven CUDA training path for this repo
- repo-owned Sunny smoke runs are proven for vision, bounded audio, and bounded transformer text classification

The public repo uses placeholder SSH targets and Windows paths. For a private lab checkout, copy `ops/sunny/lab.env.example` to `ops/sunny/lab.env` and set your own `SUNNY_*` values. `ops/sunny/lab.env` is gitignored.

## AWS Workflow

The repo now includes a concrete AWS bootstrap path:

1. Copy `infra/aws/lab.env.example` to `infra/aws/lab.env` and fill in your subnet, security group, and notification email.
2. Run `make aws-bootstrap` to create the S3 bucket and ECR repository.
3. Run `make aws-budget` to put a monthly budget and alert threshold in place.
4. Run `make aws-launch-trainer` to start an ephemeral Spot GPU trainer.
5. Run `make aws-upload-artifacts` after local or remote training to sync artifacts to S3.
6. Run `make aws-build-and-push-api` to publish the API container to ECR.
7. Run `make aws-launch-api` to launch a CPU EC2 instance that pulls the container and serves the model.

The AWS scripts live under [infra/aws/scripts](infra/aws/scripts) and assume `us-east-1` by default.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
