# Architecture

## Goal

This lab is built to mirror the moving parts of a production ML system while staying small enough to understand end to end.

## Core Components

### 1. Machine Roles

- Sunny
  - CUDA training box
  - live demo host for speech, OCR, detect, and local vision or embedding services
  - machine that needs an explicit demo-mode versus training-mode split

- Mac Studio
  - large Apple Silicon inference box
  - reasoning, embeddings, and heavyweight local model serving
  - not the primary CUDA training machine

- AWS
  - thin public edge
  - Route53, EC2 proxy, TLS, WireGuard, and selective cloud rehearsal

### 2. Environment Checklist

- `src/industry_ml_lab/training/checklist.py`

The checklist turns machine assumptions into explicit readiness checks before training starts.

It is target-aware:

- `vision` validates the deps and device readiness needed for vision training
- `audio` validates the deps and device readiness needed for audio training

Key checks:

- Python version
- target-specific training dependencies
- requested or auto-selected device backend
- **actual free CUDA VRAM** when the resolved device is `cuda`
- disk space at the output-path parent
- dataset directory existence

On Sunny, the CUDA path matters because free VRAM is the real constraint. The checklist uses `torch.cuda.mem_get_info()` so it does not confuse "GPU exists" with "GPU is actually available for training."

Run with: `ml-lab check --target vision --verbose` or `make check`

### 3. Training

- `src/industry_ml_lab/training/vision.py`
- `src/industry_ml_lab/training/audio.py`

These cover the hands-on model training portion of the target skill set:

- dataset download and preparation
- architecture selection
- hyperparameter tuning
- training and validation loops
- checkpoint export and metrics persistence

### 4. Serving

- `src/industry_ml_lab/serving/api.py`
- `src/industry_ml_lab/serving/vision.py`

The API layer gives you a place to practice:

- loading promoted artifacts
- request validation
- health and readiness endpoints
- synchronous inference
- deployment packaging

### 5. Active Learning

- `src/industry_ml_lab/active_learning/scoring.py`

This is the first step toward a real labeling pipeline. The current implementation scores uncertainty from model outputs and writes a relabel queue, which is the pragmatic core of many annotation workflows.

### 6. Retrieval

- `src/industry_ml_lab/retrieval/simple_index.py`

The repo starts with a local JSON-based cosine-similarity index because it is easy to inspect and debug. The intended upgrade path is:

1. replace local embeddings with CLIP or another encoder
2. store vectors in Postgres using `pgvector`
3. expose search through the API
4. add offline evaluation for recall and quality

### 7. Orchestration

- `src/industry_ml_lab/workflows/*`

Temporal is included as the workflow pattern for:

- training jobs
- relabel queue generation
- model promotion
- retries and auditability

### 8. Local State Layer

`docker-compose.yml` defines:

- Postgres via `pgvector`
- Valkey
- MinIO

This is enough to practice artifact storage, metadata, embeddings, and job-state coordination before moving to AWS services.

## Sunny Workflow

Sunny now has two explicit operational paths:

- demo mode
  - use `ops/sunny/reinit-demo-stack.sh`
  - keeps the public demo surface online

- training mode
  - use `ops/sunny/training-mode.sh`
  - stops the Windows-side GPU demo services, preserves the WSL2 service layer, and runs a GPU preflight

This workflow has been validated on the live box: demo mode used about `23.7 GiB` of 4090 VRAM, training mode dropped that to about `1.1 GiB`, and restore returned the demo services to a healthy state.

This is the key workflow distinction for the local lab. Training on Sunny should happen only after the GPU demo services have been intentionally drained.

The current proven runtime split on Sunny is:

- WSL2 for ops, audit, and mode switching
- Windows Python `3.11` plus CUDA for actual training readiness

## AWS Topology

The recommended lab topology is deliberately simple:

- S3 for datasets, checkpoints, and evaluation outputs
- ECR for API images
- EC2 Spot GPU instance for training
- one small CPU EC2 instance for always-on API and workers, only if needed
- no NAT Gateway in the initial lab

That last point matters. NAT Gateway charges become a fixed tax on a small lab very quickly.

## Skill Mapping

This project directly supports the capability definition:

- Classification and content understanding: vision and audio baselines
- PyTorch proficiency: all training code lives in native PyTorch
- Data curation and labeling: uncertainty scoring and relabel queues
- Serving and latency thinking: FastAPI entry points and artifact loading
- Embedding retrieval: local index with a path to `pgvector`
- Workflow systems: Temporal stubs and worker patterns
- Cloud deployment: EC2, S3, ECR, and cost-aware operations

What is not fully implemented yet is the TypeScript or NestJS control plane. That is best treated as the next extension once the Python ML path is working end to end.
