#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

load_lab_env
resolve_s3_bucket

ARTIFACTS_DIR="${ARTIFACTS_DIR:-${REPO_ROOT}/artifacts}"
ARTIFACTS_PREFIX="${ARTIFACTS_PREFIX:-artifacts}"

if [[ ! -d "${ARTIFACTS_DIR}" ]]; then
  echo "Artifacts directory does not exist: ${ARTIFACTS_DIR}" >&2
  exit 1
fi

aws s3 sync "${ARTIFACTS_DIR}" "s3://${S3_BUCKET}/${ARTIFACTS_PREFIX}/"

echo "Uploaded ${ARTIFACTS_DIR} to s3://${S3_BUCKET}/${ARTIFACTS_PREFIX}/"

