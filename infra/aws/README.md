# AWS Deployment Notes

The repo includes runnable scripts under `infra/aws/scripts` to make this topology concrete instead of doc-only:

- `bootstrap.sh`: create or verify the S3 bucket and ECR repository
- `create-budget.sh`: create a monthly AWS Budget with alert emails
- `build-and-push-api.sh`: build the FastAPI image from `Dockerfile.api` and push it to ECR
- `launch-spot-trainer.sh`: package the repo, upload a source bundle to S3, and launch an EC2 Spot GPU trainer
- `upload-artifacts.sh`: sync local `artifacts/` to S3
- `launch-api-instance.sh`: launch a CPU EC2 instance that pulls the API image and model artifact

Copy `infra/aws/lab.env.example` to `infra/aws/lab.env` before using them.

## Recommended Lab Topology

Use the smallest topology that still teaches the important lessons:

- S3 bucket for datasets, checkpoints, metrics, and relabel queues
- ECR repository for the API container
- one EC2 Spot GPU instance for training
- one optional small CPU EC2 instance for API or workflow execution

Do not start with:

- NAT Gateway
- EKS
- managed Postgres
- managed Redis

Those are valid later, but they are the wrong place to begin if the goal is skill-building with cost control.

## Why Public-Subnet EC2 Is Fine Here

For a personal lab, putting the training instance in a public subnet with tight security groups is usually the right tradeoff:

- it avoids NAT Gateway fixed charges
- it keeps the topology simple enough to reason about
- it lets you focus on model work instead of networking trivia

The rule is not that public subnets are always best. The rule is that early-stage labs should avoid paying complexity tax before they have learned the core workflow.

## Starter AWS Layout

### Storage

- S3 bucket: `ml-lab-artifacts`
- folders:
  - `datasets/`
  - `models/`
  - `metrics/`
  - `relabel-queues/`

### Training Box

- instance family:
  - `g4dn.xlarge` Spot for cheapest path
  - `g6.xlarge` Spot for a stronger default
- attach a gp3 volume large enough for temporary datasets and checkpoints
- install Docker, `uv`, and your repo
- sync artifacts back to S3 after every run

### Serving Box

- instance family:
  - `t4g.medium` if you want the cheapest persistent service
  - `c7i.large` if you want more conventional x86 compatibility
- run the FastAPI service behind a simple reverse proxy only when you need an endpoint

## Deployment Sequence

1. Create the S3 bucket and ECR repository.
2. Push the API container image to ECR.
3. Launch the GPU EC2 Spot instance for training.
4. Sync the resulting artifacts to S3.
5. Launch or update the CPU API instance.
6. Point `ML_LAB_MODEL_PATH` at the promoted checkpoint.

The script equivalents are:

```bash
cp infra/aws/lab.env.example infra/aws/lab.env
make aws-bootstrap
make aws-budget
make aws-launch-trainer
make aws-upload-artifacts
make aws-build-and-push-api
make aws-launch-api
```

## Guardrails

1. Add an AWS Budget before doing repeated experiments.
2. Tag all resources with `Project=industry-ml-lab`.
3. Stop persistent CPU instances when not in use.
4. Delete idle EBS volumes.
5. Keep the GPU instance ephemeral by default.
6. Use the provided lifecycle policy to keep the ECR repository from accumulating stale images.
7. Prefer an IAM instance profile for EC2 access to S3 and ECR instead of long-lived credentials.

## What To Build Next

Once the base loop is working:

- move embeddings into Postgres plus `pgvector`
- add a systemd service or container supervisor on the CPU box
- add CI to build and push the API image
- add a NestJS control-plane service if you want direct practice with the wider target stack
