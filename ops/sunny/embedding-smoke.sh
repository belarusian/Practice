#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
POWERSHELL="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
REPORT_STAMP="$(date +%Y%m%d-%H%M%S)-embedding-smoke"
REPORT_DIR="$ROOT_DIR/artifacts/sunny-reports/$REPORT_STAMP"
RECORDS_PATH="$ROOT_DIR/sample-data/text-records.jsonl"
ARTIFACT_DIR_SET=false
QUERY="free CUDA memory for training"
DEVICE="cuda"
MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE=16
MAX_LENGTH=256
RESTORE_DEMO=true
USE_TRAINING_MODE=true
RESTORE_TIMEOUT_SECONDS="${RESTORE_TIMEOUT_SECONDS:-600}"

usage() {
  cat <<'EOF'
Usage:
  bash ops/sunny/embedding-smoke.sh [options]

Options:
  --report-dir PATH       Report directory on Sunny
  --artifact-dir PATH     Directory for generated embedding artifacts
  --records PATH          JSONL text records to embed
  --query TEXT            Natural-language query for search smoke
  --device NAME           Embedding device (default: cuda)
  --model-name NAME       Hugging Face encoder model name (default: sentence-transformers/all-MiniLM-L6-v2)
  --batch-size N          Embedding batch size (default: 16)
  --max-length N          Tokenizer max length (default: 256)
  --skip-training-mode    Do not stop demo GPU services first
  --no-restore-demo       Do not restore the demo stack after the run
  --help                  Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --report-dir)
      REPORT_DIR="$2"
      shift
      ;;
    --artifact-dir)
      ARTIFACT_DIR="$2"
      ARTIFACT_DIR_SET=true
      shift
      ;;
    --records)
      RECORDS_PATH="$2"
      shift
      ;;
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
    --skip-training-mode)
      USE_TRAINING_MODE=false
      ;;
    --no-restore-demo)
      RESTORE_DEMO=false
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

if [[ "$ARTIFACT_DIR_SET" == false ]]; then
  ARTIFACT_DIR="$REPORT_DIR/embedding-artifacts"
fi
OUTPUT_PATH="$ARTIFACT_DIR/text-index.json"

mkdir -p "$REPORT_DIR"
mkdir -p "$ARTIFACT_DIR"

STATUS_FILE="$REPORT_DIR/status.tsv"
LOG_FILE="$REPORT_DIR/run.log"
: >"$STATUS_FILE"
: >"$LOG_FILE"

log() {
  printf '%s\n' "$*" | tee -a "$LOG_FILE"
}

capture_cmd() {
  local name="$1"
  shift
  local output_file="$REPORT_DIR/${name}.txt"
  local exit_code=0

  log ""
  log "== $name =="
  log "COMMAND: $*"

  if "$@" >"$output_file" 2>&1; then
    exit_code=0
  else
    exit_code=$?
  fi

  printf '%s\t%s\t%s\n' "$name" "$exit_code" "$output_file" >>"$STATUS_FILE"
  log "EXIT CODE: $exit_code"
  cat "$output_file" | tee -a "$LOG_FILE"
  return 0
}

capture_cmd "host-meta" bash -lc "hostname && whoami && date && pwd"
capture_cmd "audit-before" bash "$ROOT_DIR/ops/sunny/audit-demo-stack.sh"

if [[ "$USE_TRAINING_MODE" == true ]]; then
  capture_cmd "training-mode" bash "$ROOT_DIR/ops/sunny/training-mode.sh"
fi

capture_cmd \
  "windows-embedding-smoke-console" \
  "$POWERSHELL" \
  -NoProfile \
  -ExecutionPolicy Bypass \
  -File "$(wslpath -w "$ROOT_DIR/ops/sunny/embedding-smoke.ps1")" \
  -ReportDir "$(wslpath -w "$REPORT_DIR")" \
  -RepoRoot "$(wslpath -w "$ROOT_DIR")" \
  -RepoSrc "$(wslpath -w "$ROOT_DIR/src")" \
  -RecordsPath "$(wslpath -w "$RECORDS_PATH")" \
  -OutputPath "$(wslpath -w "$OUTPUT_PATH")" \
  -Query "$QUERY" \
  -Device "$DEVICE" \
  -ModelName "$MODEL_NAME" \
  -BatchSize "$BATCH_SIZE" \
  -MaxLength "$MAX_LENGTH"

capture_cmd \
  "windows-embedding-summary-check" \
  python3 "$ROOT_DIR/ops/sunny/exit_from_summary_json.py" "$REPORT_DIR/windows-embedding-smoke-summary.json"

if [[ "$RESTORE_DEMO" == true ]]; then
  capture_cmd "restore-demo" timeout "$RESTORE_TIMEOUT_SECONDS" bash "$ROOT_DIR/ops/sunny/training-mode.sh" --restore-demo
  capture_cmd "audit-after-restore" bash "$ROOT_DIR/ops/sunny/audit-demo-stack.sh"
fi

python3 "$ROOT_DIR/ops/sunny/summarize_report_status.py" \
  "$STATUS_FILE" \
  "$REPORT_DIR/summary.json" \
  "$ARTIFACT_DIR"

log ""
log "REPORT_DIR=$REPORT_DIR"
log "ARTIFACT_DIR=$ARTIFACT_DIR"
