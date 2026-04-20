#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

load_lab_env
ACCOUNT_ID="$(aws_account_id)"

require_env SUBNET_ID SECURITY_GROUP_ID IAM_INSTANCE_PROFILE MODEL_S3_URI

AMI_ID="$(resolve_x86_ami)"
API_INSTANCE_TYPE="${API_INSTANCE_TYPE:-c7i.large}"
API_PORT="${API_PORT:-8000}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}"
INDEX_S3_URI="${INDEX_S3_URI:-}"

USER_DATA_FILE="$(mktemp)"
trap 'rm -f "${USER_DATA_FILE}"' EXIT

cat >"${USER_DATA_FILE}" <<EOF
#!/bin/bash
set -euxo pipefail

dnf update -y
dnf install -y awscli docker
systemctl enable --now docker

mkdir -p /opt/industry-ml-lab/model
mkdir -p /opt/industry-ml-lab/index

aws s3 cp "${MODEL_S3_URI}" /opt/industry-ml-lab/model/model.pt

if [[ -n "${INDEX_S3_URI}" ]]; then
  aws s3 cp "${INDEX_S3_URI}" /opt/industry-ml-lab/index/demo-index.json
fi

aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
docker pull "${IMAGE_URI}"
docker rm -f industry-ml-lab-api || true
docker run -d \
  --name industry-ml-lab-api \
  --restart unless-stopped \
  -p ${API_PORT}:8000 \
  -e ML_LAB_MODEL_PATH=/opt/models/model.pt \
  -e ML_LAB_INDEX_PATH=/opt/index/demo-index.json \
  -e ML_LAB_SERVICE_VERSION=${IMAGE_TAG} \
  -v /opt/industry-ml-lab/model:/opt/models:ro \
  -v /opt/industry-ml-lab/index:/opt/index:ro \
  "${IMAGE_URI}"
EOF

TAG_SPECIFICATIONS="[
  {
    \"ResourceType\": \"instance\",
    \"Tags\": [
      {\"Key\": \"Name\", \"Value\": \"${PROJECT_NAME}-api\"},
      {\"Key\": \"Project\", \"Value\": \"${PROJECT_NAME}\"},
      {\"Key\": \"Role\", \"Value\": \"api\"}
    ]
  }
]"

declare -a CMD=(
  aws ec2 run-instances
  --region "${AWS_REGION}"
  --image-id "${AMI_ID}"
  --instance-type "${API_INSTANCE_TYPE}"
  --instance-initiated-shutdown-behavior stop
  --subnet-id "${SUBNET_ID}"
  --security-group-ids "${SECURITY_GROUP_ID}"
  --iam-instance-profile "Name=${IAM_INSTANCE_PROFILE}"
  --tag-specifications "${TAG_SPECIFICATIONS}"
  --user-data "file://${USER_DATA_FILE}"
  --count 1
)

if [[ -n "${KEY_NAME:-}" ]]; then
  CMD+=(--key-name "${KEY_NAME}")
fi

INSTANCE_ID="$("${CMD[@]}" --query 'Instances[0].InstanceId' --output text)"

aws ec2 wait instance-running --region "${AWS_REGION}" --instance-ids "${INSTANCE_ID}"

PUBLIC_IP="$(aws ec2 describe-instances \
  --region "${AWS_REGION}" \
  --instance-ids "${INSTANCE_ID}" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)"

cat <<EOF
Launched API instance.
Instance ID: ${INSTANCE_ID}
Instance type: ${API_INSTANCE_TYPE}
Public IP: ${PUBLIC_IP}
Image URI: ${IMAGE_URI}
EOF

