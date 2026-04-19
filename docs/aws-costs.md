# AWS Costs

All prices below were checked for `us-east-1` on April 18, 2026 using the AWS Pricing API and the EC2 Spot Price History API. Spot prices are live market values and can move, so treat them as point-in-time guidance rather than a contract.

## Recommended Default

For a practice lab, the cost-efficient baseline is:

- ephemeral GPU training on EC2 Spot
- cheap CPU serving only when you actually need an always-on endpoint
- S3 for artifacts
- gp3 EBS for short-lived training volumes
- no NAT Gateway in the initial setup
- self-host Postgres and Valkey locally or on one small box before moving to managed services

## Verified Prices

### Compute

| Resource | Price |
| --- | ---: |
| `g4dn.xlarge` on-demand | `$0.5260/hour` |
| `g4dn.xlarge` spot | `$0.2319/hour` |
| `g6.xlarge` on-demand | `$0.8048/hour` |
| `g6.xlarge` spot | `$0.3548/hour` |
| `g5.xlarge` on-demand | `$1.0060/hour` |
| `g5.xlarge` spot | `$0.6070/hour` |
| `c7i.large` on-demand | `$0.08925/hour` |
| `t4g.medium` on-demand | `$0.0336/hour` |

### Storage and Networking

| Resource | Price |
| --- | ---: |
| S3 Standard, first 50 TB | `$0.023/GB-month` |
| EBS `gp3` storage | `$0.08/GB-month` |
| EBS `gp3` provisioned IOPS | `$0.005/IOPS-month` |
| Public IPv4 address | `$0.005/hour` |
| NAT Gateway hourly charge | `$0.045/hour` |
| NAT Gateway data processing | `$0.045/GB` |

### Managed Services, If You Choose Them

| Resource | Price |
| --- | ---: |
| RDS PostgreSQL `db.t4g.micro` Single-AZ | `$0.016/hour` |
| ElastiCache Valkey `cache.t4g.small` | `$0.0256/hour` |
| ElastiCache Redis `cache.t4g.small` | `$0.032/hour` |

## What This Means In Practice

### Lean Lab

Assume:

- 20 GPU training hours per month
- 100 GB gp3 volume kept all month
- 50 GB in S3
- no always-on API

Estimated monthly cost:

- `g4dn.xlarge` spot: `20 * 0.2319 = $4.64`
- `g6.xlarge` spot: `20 * 0.3548 = $7.10`
- EBS gp3 100 GB: `100 * 0.08 = $8.00`
- S3 50 GB: `50 * 0.023 = $1.15`

Total:

- with `g4dn.xlarge` spot: about `$13.79/month`
- with `g6.xlarge` spot: about `$16.25/month`

This is the sweet spot for learning.

### Practical Always-On Lab

Add:

- one `t4g.medium` API or worker box running full month
- one public IPv4 address
- 40 GPU training hours on `g6.xlarge` spot

Estimated monthly cost:

- `t4g.medium`: `720 * 0.0336 = $24.19`
- public IPv4: `720 * 0.005 = $3.60`
- `g6.xlarge` spot for 40 hours: `40 * 0.3548 = $14.19`
- EBS gp3 100 GB: `$8.00`
- S3 50 GB: `$1.15`

Total: about `$51.13/month`

That is still reasonable, and it lets you practice deployment and monitoring.

### Managed Upgrade

If you move to managed services, fixed cost rises fast:

- RDS PostgreSQL `db.t4g.micro`: about `$11.52/month`
- ElastiCache Valkey `cache.t4g.small`: about `$18.43/month`

That means managed Postgres plus managed cache adds about `$29.95/month` before storage, data transfer, or compute.

## Cost Guardrails

1. Do not start with a NAT Gateway.
2. Put the training box in a public subnet if this is a personal lab.
3. Stop CPU boxes when idle.
4. Terminate GPU instances as soon as training is done.
5. Store artifacts in S3, not on long-lived EBS volumes.
6. Use Spot for training unless the experiment cannot tolerate interruption.
7. Add AWS Budgets before scaling the lab.

## Suggested Default Choices

- Start with `g4dn.xlarge` Spot if you want the cheapest workable GPU path.
- Prefer `g6.xlarge` Spot if you want a more modern default and can tolerate a slightly higher rate.
- Skip `g5.xlarge` unless you know you need it.
- Use `t4g.medium` for a persistent lightweight API or worker node.
- Delay RDS and ElastiCache until the single-box version becomes a constraint.

## Official Sources

- EC2 On-Demand pricing: https://aws.amazon.com/ec2/pricing/on-demand/
- Amazon S3 pricing: https://aws.amazon.com/s3/pricing/
- Amazon EBS pricing: https://aws.amazon.com/ebs/pricing/
- Amazon VPC pricing: https://aws.amazon.com/vpc/pricing/
- Amazon RDS for PostgreSQL pricing: https://aws.amazon.com/rds/postgresql/pricing/
- Amazon ElastiCache pricing: https://aws.amazon.com/elasticache/pricing/
- NAT Gateway pricing details: https://docs.aws.amazon.com/vpc/latest/userguide/nat-gateway-pricing.html

