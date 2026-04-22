#!/usr/bin/env bash

set -euo pipefail
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAB_ENV_FILE="${SUNNY_LAB_ENV_FILE:-$ROOT_DIR/ops/sunny/lab.env}"
if [[ -f "$LAB_ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$LAB_ENV_FILE"
  set +a
fi
STAMP="$(date +%Y%m%d-%H%M%S)-embedding-smoke"
LOCAL_REPORT_DIR="$ROOT_DIR/artifacts/sunny-reports/$STAMP"
REMOTE_REPO_DIR="${SUNNY_REPO_DIR:-/home/ml-lab/Practice}"
REMOTE_REPORT_DIR="$REMOTE_REPO_DIR/artifacts/sunny-reports/$STAMP"
SUNNY_SSH_TARGET="${SUNNY_SSH_TARGET:-mlops@10.0.0.2}"
SUNNY_PROXY_TARGET="${SUNNY_PROXY_TARGET:-ubuntu@203.0.113.10}"
SUNNY_PROXY_KEY="${SUNNY_PROXY_KEY:-$HOME/.ssh/example-proxy-key.pem}"
SUNNY_IDENTITY_FILE="${SUNNY_IDENTITY_FILE:-$HOME/.ssh/id_ed25519}"
QUERY="free CUDA memory for training"
DEVICE="cuda"
MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE=16
MAX_LENGTH=256
SKIP_SYNC=false
SKIP_TRAINING_MODE=false
NO_RESTORE_DEMO=false

usage() {
  cat <<'EOF'
Usage:
  bash ops/sunny/run-remote-embedding-smoke.sh [options]

Options:
  --query TEXT            Natural-language query for search smoke
  --device NAME           Embedding device (default: cuda)
  --model-name NAME       Hugging Face encoder model name (default: sentence-transformers/all-MiniLM-L6-v2)
  --batch-size N          Embedding batch size (default: 16)
  --max-length N          Tokenizer max length (default: 256)
  --skip-sync             Do not rsync the local repo to Sunny before running
  --skip-training-mode    Do not stop demo GPU services first
  --no-restore-demo       Do not restore the demo stack after the run
  --help                  Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --query)
      QUERY="$2"
      shift
      ;;
    --device)
      DEVICE="$2"
      shift
      ;;
    --model-name)
      MODEL_NAME="$2"
      shift
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift
      ;;
    --max-length)
      MAX_LENGTH="$2"
      shift
      ;;
    --skip-sync)
      SKIP_SYNC=true
      ;;
    --skip-training-mode)
      SKIP_TRAINING_MODE=true
      ;;
    --no-restore-demo)
      NO_RESTORE_DEMO=true
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

mkdir -p "$LOCAL_REPORT_DIR"

SSH_PROXY_OPT="ProxyCommand=ssh -i $SUNNY_PROXY_KEY -W %h:%p $SUNNY_PROXY_TARGET"
RSYNC_RSH="ssh -o 'ProxyCommand=ssh -i $SUNNY_PROXY_KEY -W %h:%p $SUNNY_PROXY_TARGET' -i $SUNNY_IDENTITY_FILE"

if [[ "$SKIP_SYNC" == false ]]; then
  rsync -az \
    --delete \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude '__pycache__/' \
    --exclude 'artifacts/' \
    --exclude 'data/' \
    "$ROOT_DIR/" \
    -e "$RSYNC_RSH" \
    "$SUNNY_SSH_TARGET:$REMOTE_REPO_DIR/"
fi

REMOTE_ARGS=(
  "--report-dir" "$REMOTE_REPORT_DIR"
  "--query" "$QUERY"
  "--device" "$DEVICE"
  "--model-name" "$MODEL_NAME"
  "--batch-size" "$BATCH_SIZE"
  "--max-length" "$MAX_LENGTH"
)

if [[ "$SKIP_TRAINING_MODE" == true ]]; then
  REMOTE_ARGS+=("--skip-training-mode")
fi

if [[ "$NO_RESTORE_DEMO" == true ]]; then
  REMOTE_ARGS+=("--no-restore-demo")
fi

REMOTE_ARGS_STR=""
printf -v REMOTE_ARGS_STR ' %q' "${REMOTE_ARGS[@]}"

SSH_EXIT_CODE=0
ssh \
  -o "$SSH_PROXY_OPT" \
  -o ConnectTimeout=8 \
  -i "$SUNNY_IDENTITY_FILE" \
  "$SUNNY_SSH_TARGET" \
  "cd '$REMOTE_REPO_DIR' && bash ops/sunny/embedding-smoke.sh$REMOTE_ARGS_STR" \
  | tee "$LOCAL_REPORT_DIR/remote-console.txt" || SSH_EXIT_CODE=$?

RSYNC_EXIT_CODE=0
rsync -az \
  -e "$RSYNC_RSH" \
  "$SUNNY_SSH_TARGET:$REMOTE_REPORT_DIR/" \
  "$LOCAL_REPORT_DIR/" || RSYNC_EXIT_CODE=$?

SUMMARY_EXIT=0
if [[ "$RSYNC_EXIT_CODE" -eq 0 ]]; then
  if [[ -f "$LOCAL_REPORT_DIR/summary.json" ]]; then
    python3 "$ROOT_DIR/ops/sunny/exit_from_summary_json.py" "$LOCAL_REPORT_DIR/summary.json" || SUMMARY_EXIT=$?
  else
    echo "run-remote-embedding-smoke.sh: expected summary.json under $LOCAL_REPORT_DIR after rsync" >&2
    SUMMARY_EXIT=2
  fi
else
  echo "run-remote-embedding-smoke.sh: rsync failed while copying $REMOTE_REPORT_DIR to $LOCAL_REPORT_DIR" >&2
  SUMMARY_EXIT=2
fi

printf 'LOCAL_REPORT_DIR=%s\n' "$LOCAL_REPORT_DIR"

if [[ "$SSH_EXIT_CODE" -ne 0 ]] || [[ "$RSYNC_EXIT_CODE" -ne 0 ]] || [[ "$SUMMARY_EXIT" -ne 0 ]]; then
  exit 1
fi
exit 0
