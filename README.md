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
uv sync
uv run ml-lab build-index --records sample-data/embedding-records.jsonl --output artifacts/demo-index.json
uv run ml-lab search-index --index-path artifacts/demo-index.json --vector 0.92,0.08,0.04
uv run ml-lab active-learning-report --predictions sample-data/predictions.jsonl --output artifacts/relabel-queue.csv
```

If you want the full API or model training flows, install the project dependencies first with `uv sync`, then use the commands in the `Makefile`.

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
- Phase 2: Train the vision model, then wire the API to a real checkpoint
- Phase 3: Add the audio baseline and uncertainty-driven relabel loop
- Phase 4: Replace the local retrieval index with Postgres plus `pgvector`
- Phase 5: Containerize and deploy to AWS EC2 and S3
- Phase 6: Add a control plane in TypeScript or NestJS if you want direct exposure to the rest of the target stack

## Key Documents

- [Architecture](docs/architecture.md)
- [AWS Costs](docs/aws-costs.md)
- [Roadmap](docs/roadmap.md)
- [AWS Deployment Notes](infra/aws/README.md)

