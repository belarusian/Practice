#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

load_lab_env
resolve_s3_bucket
aws_account_id >/dev/null

echo "Using region: ${AWS_REGION}"
echo "Using bucket: ${S3_BUCKET}"
echo "Using ECR repository: ${ECR_REPOSITORY}"

if aws s3api head-bucket --bucket "${S3_BUCKET}" 2>/dev/null; then
  echo "S3 bucket already exists: ${S3_BUCKET}"
else
  if [[ "${AWS_REGION}" == "us-east-1" ]]; then
    aws s3api create-bucket --bucket "${S3_BUCKET}"
  else
    aws s3api create-bucket \
      --bucket "${S3_BUCKET}" \
      --create-bucket-configuration "LocationConstraint=${AWS_REGION}"
  fi
  echo "Created S3 bucket: ${S3_BUCKET}"
fi

aws s3api put-bucket-versioning \
  --bucket "${S3_BUCKET}" \
  --versioning-configuration Status=Enabled

aws s3api put-public-access-block \
  --bucket "${S3_BUCKET}" \
  --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

if aws ecr describe-repositories --repository-names "${ECR_REPOSITORY}" >/dev/null 2>&1; then
  echo "ECR repository already exists: ${ECR_REPOSITORY}"
else
  aws ecr create-repository \
    --repository-name "${ECR_REPOSITORY}" \
    --image-scanning-configuration scanOnPush=true \
    --image-tag-mutability MUTABLE >/dev/null
  echo "Created ECR repository: ${ECR_REPOSITORY}"
fi

aws ecr put-lifecycle-policy \
  --repository-name "${ECR_REPOSITORY}" \
  --lifecycle-policy-text "file://${SCRIPT_DIR}/../ecr-lifecycle-policy.json" >/dev/null

cat <<EOF
Bootstrap complete.
S3 bucket: ${S3_BUCKET}
ECR repository: ${ECR_REPOSITORY}
EOF

