#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AWS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${AWS_DIR}/../.." && pwd)"
LAB_ENV_FILE="${ML_LAB_AWS_ENV_FILE:-${AWS_DIR}/lab.env}"

load_lab_env() {
  if [[ -f "${LAB_ENV_FILE}" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "${LAB_ENV_FILE}"
    set +a
  fi

  : "${PROJECT_NAME:=industry-ml-lab}"
  : "${AWS_REGION:=us-east-1}"
  : "${ECR_REPOSITORY:=${PROJECT_NAME}-api}"

  export PROJECT_NAME AWS_REGION AWS_DEFAULT_REGION="${AWS_REGION}" ECR_REPOSITORY
}

require_env() {
  local missing=0
  local name

  for name in "$@"; do
    if [[ -z "${!name:-}" ]]; then
      echo "Missing required environment variable: ${name}" >&2
      missing=1
    fi
  done

  if [[ "${missing}" -ne 0 ]]; then
    echo "Populate ${LAB_ENV_FILE} or export the variables before running this script." >&2
    exit 1
  fi
}

aws_account_id() {
  if [[ -z "${AWS_ACCOUNT_ID:-}" ]]; then
    AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
    export AWS_ACCOUNT_ID
  fi
  printf '%s\n' "${AWS_ACCOUNT_ID}"
}

default_bucket_name() {
  local account_id
  account_id="$(aws_account_id)"
  printf '%s\n' "${PROJECT_NAME}-${account_id}-${AWS_REGION}" | tr '[:upper:]_' '[:lower:]-'
}

resolve_s3_bucket() {
  if [[ -z "${S3_BUCKET:-}" ]]; then
    S3_BUCKET="$(default_bucket_name)"
    export S3_BUCKET
  fi
}

resolve_x86_ami() {
  if [[ -n "${AMI_ID:-}" ]]; then
    printf '%s\n' "${AMI_ID}"
    return
  fi

  aws ssm get-parameter \
    --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
    --query 'Parameter.Value' \
    --output text
}

