# Roadmap

## Phase 1: Baseline Vision Training

Goal:
Train and evaluate a computer-vision classifier end to end.

Tasks:

- run the ResNet baseline on CIFAR-10
- save checkpoints and metrics
- compare pretrained versus from-scratch runs
- log common failure cases

Why it matters:
This covers the hands-on training piece of the target profile.

## Phase 2: Audio Track

Goal:
Add direct experience with voice and audio AI.

Tasks:

- train the Speech Commands baseline
- inspect confusion between similar commands
- tune augmentation and batch size
- record lessons from failed runs

Why it matters:
This gives you practical audio pipeline experience instead of leaving voice AI as a purely conceptual bullet point.

## Phase 3: Transformer Text Classification

Goal:
Add direct transformer fine-tuning experience for tagging and content understanding.

Tasks:

- fine-tune a DistilBERT-style classifier on GLUE/SST-2
- add bounded smoke runs for cheap validation
- save Hugging Face `save_pretrained()` artifacts and metrics
- compare CPU, MPS, and CUDA behavior across machines

Why it matters:
This maps directly to classification, tagging, and content-understanding systems without jumping straight to expensive LLM pretraining.

## Phase 4: Labeling and Active Learning

Goal:
Create a lightweight annotation workflow.

Tasks:

- export prediction probabilities from the trained model
- generate a relabel queue with uncertainty scoring
- review false positives and false negatives
- version the corrected labels

Why it matters:
This is how you move from model training to dataset improvement loops.

## Phase 5: Retrieval and Semantic Search

Goal:
Introduce embeddings and nearest-neighbor retrieval.

Tasks:

- swap in CLIP or another encoder
- embed image or audio metadata
- replace the local JSON index with Postgres plus `pgvector`
- evaluate search quality with hand-built queries

Why it matters:
It covers the retrieval and semantic-similarity part of the target profile.

## Phase 6: Serving and Monitoring

Goal:
Deploy a model as an actual service.

Tasks:

- package the FastAPI service in Docker
- serve a promoted vision checkpoint
- measure latency and batch behavior
- add request logs, model version reporting, and basic drift counters

Why it matters:
This is the bridge from notebooks to production behavior.

## Phase 7: Workflow Orchestration

Goal:
Make training and promotion reproducible.

Tasks:

- run the Temporal worker
- wrap training in a workflow
- attach relabel-queue generation as a follow-on step
- version and promote model artifacts

Why it matters:
Production ML depends on repeatable workflows, not ad hoc shell history.

## Phase 8: AWS Deployment

Goal:
Practice cost-aware cloud deployment.

Tasks:

- push artifacts to S3
- train on EC2 Spot GPU instances
- deploy the API to a small CPU EC2 instance
- add ECR and deployment scripts

Why it matters:
This builds operational judgment around infrastructure, reliability, and cost.

## Phase 9: Full-Stack Expansion

Goal:
Touch the wider stack mentioned in the target job profile.

Tasks:

- add a TypeScript or NestJS control plane
- persist job metadata in Postgres
- use Valkey for short-lived state
- add webhook callbacks for training completion

Why it matters:
This is where the repo grows from an ML lab into a more complete product-like system.
