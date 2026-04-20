#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

load_lab_env
resolve_s3_bucket

require_env SUBNET_ID SECURITY_GROUP_ID

ACCOUNT_ID="$(aws_account_id)"
AMI_ID="$(resolve_x86_ami)"
TRAINER_INSTANCE_TYPE="${TRAINER_INSTANCE_TYPE:-g6.xlarge}"
TRAINER_VOLUME_SIZE_GB="${TRAINER_VOLUME_SIZE_GB:-200}"
AUTO_TRAIN_COMMAND="${AUTO_TRAIN_COMMAND:-}"
TIMESTAMP="$(date -u +%Y%m%d%H%M%S)"
GIT_SHA="$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
BUNDLE_NAME="${PROJECT_NAME}-source-${GIT_SHA}-${TIMESTAMP}.tar.gz"
BUNDLE_PATH="${TMPDIR:-/tmp}/${BUNDLE_NAME}"
BUNDLE_S3_KEY="source-bundles/${BUNDLE_NAME}"

tar \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='.ruff_cache' \
  --exclude='.mypy_cache' \
  --exclude='artifacts' \
  --exclude='data' \
  -czf "${BUNDLE_PATH}" \
  -C "${REPO_ROOT}" .

aws s3 cp "${BUNDLE_PATH}" "s3://${S3_BUCKET}/${BUNDLE_S3_KEY}" >/dev/null
SOURCE_BUNDLE_URL="$(aws s3 presign "s3://${S3_BUCKET}/${BUNDLE_S3_KEY}" --expires-in 86400)"
rm -f "${BUNDLE_PATH}"

USER_DATA_FILE="$(mktemp)"
trap 'rm -f "${USER_DATA_FILE}"' EXIT

cat >"${USER_DATA_FILE}" <<EOF
#!/bin/bash
set -euxo pipefail

dnf update -y
dnf install -y awscli docker git jq python3 python3-pip tar gzip tmux unzip
systemctl enable --now docker

curl -LsSf https://astral.sh/uv/install.sh | sh
ln -sf /root/.local/bin/uv /usr/local/bin/uv

mkdir -p /opt/industry-ml-lab
cd /opt/industry-ml-lab
curl -L "${SOURCE_BUNDLE_URL}" -o source.tar.gz
tar -xzf source.tar.gz
rm -f source.tar.gz

cat >/opt/industry-ml-lab/sync-artifacts.sh <<EOS
#!/bin/bash
set -euxo pipefail
if [[ -d /opt/industry-ml-lab/artifacts ]]; then
  aws s3 sync /opt/industry-ml-lab/artifacts "s3://${S3_BUCKET}/artifacts/"
fi
EOS
chmod +x /opt/industry-ml-lab/sync-artifacts.sh

cat >/etc/profile.d/industry-ml-lab.sh <<EOS
export AWS_DEFAULT_REGION=${AWS_REGION}
export ML_LAB_ARTIFACT_ROOT=/opt/industry-ml-lab/artifacts
EOS

cd /opt/industry-ml-lab
if [[ -n "${AUTO_TRAIN_COMMAND}" ]]; then
  nohup bash -lc "${AUTO_TRAIN_COMMAND}" >/var/log/industry-ml-lab-bootstrap.log 2>&1 &
fi
EOF

SPOT_OPTIONS='{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}'
if [[ -n "${SPOT_MAX_PRICE:-}" ]]; then
  SPOT_OPTIONS="{\"MarketType\":\"spot\",\"SpotOptions\":{\"MaxPrice\":\"${SPOT_MAX_PRICE}\",\"SpotInstanceType\":\"one-time\",\"InstanceInterruptionBehavior\":\"terminate\"}}"
fi

BLOCK_DEVICE_MAPPINGS="[{\"DeviceName\":\"/dev/xvda\",\"Ebs\":{\"VolumeSize\":${TRAINER_VOLUME_SIZE_GB},\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]"
TAG_SPECIFICATIONS="[
  {
    \"ResourceType\": \"instance\",
    \"Tags\": [
      {\"Key\": \"Name\", \"Value\": \"${PROJECT_NAME}-trainer\"},
      {\"Key\": \"Project\", \"Value\": \"${PROJECT_NAME}\"},
      {\"Key\": \"Role\", \"Value\": \"trainer\"}
    ]
  }
]"

declare -a CMD=(
  aws ec2 run-instances
  --region "${AWS_REGION}"
  --image-id "${AMI_ID}"
  --instance-type "${TRAINER_INSTANCE_TYPE}"
  --instance-market-options "${SPOT_OPTIONS}"
  --instance-initiated-shutdown-behavior terminate
  --subnet-id "${SUBNET_ID}"
  --security-group-ids "${SECURITY_GROUP_ID}"
  --block-device-mappings "${BLOCK_DEVICE_MAPPINGS}"
  --tag-specifications "${TAG_SPECIFICATIONS}"
  --user-data "file://${USER_DATA_FILE}"
  --count 1
)

if [[ -n "${KEY_NAME:-}" ]]; then
  CMD+=(--key-name "${KEY_NAME}")
fi

if [[ -n "${IAM_INSTANCE_PROFILE:-}" ]]; then
  CMD+=(--iam-instance-profile "Name=${IAM_INSTANCE_PROFILE}")
fi

INSTANCE_ID="$("${CMD[@]}" --query 'Instances[0].InstanceId' --output text)"

aws ec2 wait instance-running --region "${AWS_REGION}" --instance-ids "${INSTANCE_ID}"

PUBLIC_IP="$(aws ec2 describe-instances \
  --region "${AWS_REGION}" \
  --instance-ids "${INSTANCE_ID}" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)"

cat <<EOF
Launched Spot trainer.
Instance ID: ${INSTANCE_ID}
Instance type: ${TRAINER_INSTANCE_TYPE}
Public IP: ${PUBLIC_IP}
Source bundle: s3://${S3_BUCKET}/${BUNDLE_S3_KEY}
EOF
