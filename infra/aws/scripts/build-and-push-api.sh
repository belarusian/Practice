#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

load_lab_env
ACCOUNT_ID="$(aws_account_id)"

DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"
IMAGE_TAG="${IMAGE_TAG:-$(git -C "${REPO_ROOT}" rev-parse --short HEAD)}"
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_URI="${REGISTRY}/${ECR_REPOSITORY}:${IMAGE_TAG}"
LATEST_URI="${REGISTRY}/${ECR_REPOSITORY}:latest"

if ! aws ecr describe-repositories --repository-names "${ECR_REPOSITORY}" >/dev/null 2>&1; then
  echo "ECR repository ${ECR_REPOSITORY} does not exist. Run bootstrap first." >&2
  exit 1
fi

aws ecr get-login-password --region "${AWS_REGION}" | \
  docker login --username AWS --password-stdin "${REGISTRY}"

docker build \
  --platform "${DOCKER_PLATFORM}" \
  -f "${REPO_ROOT}/Dockerfile.api" \
  -t "${IMAGE_URI}" \
  -t "${LATEST_URI}" \
  "${REPO_ROOT}"

docker push "${IMAGE_URI}"
docker push "${LATEST_URI}"

cat <<EOF
Pushed API image.
Primary image: ${IMAGE_URI}
Latest image: ${LATEST_URI}
EOF

