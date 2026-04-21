#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
POWERSHELL="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
REPORT_STAMP="$(date +%Y%m%d-%H%M%S)-audio-smoke"
REPORT_DIR="$ROOT_DIR/artifacts/sunny-reports/$REPORT_STAMP"
DATASET_ROOT="$ROOT_DIR/data/audio"
EPOCHS=1
DEVICE="cuda"
RESTORE_DEMO=true
USE_TRAINING_MODE=true
RESTORE_TIMEOUT_SECONDS="${RESTORE_TIMEOUT_SECONDS:-600}"
OUTPUT_DIR_SET=false

usage() {
  cat <<'EOF'
Usage:
  bash ops/sunny/audio-smoke.sh [options]

Options:
  --report-dir PATH       Report directory on Sunny
  --output-dir PATH       Output directory for training artifacts
  --dataset-root PATH     Dataset root for audio training (SpeechCommands)
  --epochs N              Number of training epochs (default: 1)
  --device NAME           Training device (default: cuda)
  --skip-training-mode   Do not stop demo GPU services first
  --no-restore-demo      Do not restore the demo stack after the run
  --help                  Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --report-dir)
      REPORT_DIR="$2"
      shift
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      OUTPUT_DIR_SET=true
      shift
      ;;
    --dataset-root)
      DATASET_ROOT="$2"
      shift
      ;;
    --epochs)
      EPOCHS="$2"
      shift
      ;;
    --device)
      DEVICE="$2"
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

if [[ "$OUTPUT_DIR_SET" == false ]]; then
  OUTPUT_DIR="$REPORT_DIR/audio-artifacts"
fi

mkdir -p "$REPORT_DIR"
mkdir -p "$DATASET_ROOT"

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
  "windows-audio-smoke-console" \
  "$POWERSHELL" \
  -NoProfile \
  -ExecutionPolicy Bypass \
  -File "$(wslpath -w "$ROOT_DIR/ops/sunny/audio-smoke.ps1")" \
  -ReportDir "$(wslpath -w "$REPORT_DIR")" \
  -RepoRoot "$(wslpath -w "$ROOT_DIR")" \
  -RepoSrc "$(wslpath -w "$ROOT_DIR/src")" \
  -OutputDir "$(wslpath -w "$OUTPUT_DIR")" \
  -DatasetRoot "$(wslpath -w "$DATASET_ROOT")" \
  -Epochs "$EPOCHS" \
  -Device "$DEVICE"

capture_cmd \
  "windows-audio-summary-check" \
  python3 "$ROOT_DIR/ops/sunny/exit_from_summary_json.py" "$REPORT_DIR/windows-audio-smoke-summary.json"

if [[ "$RESTORE_DEMO" == true ]]; then
  capture_cmd "restore-demo" timeout "$RESTORE_TIMEOUT_SECONDS" bash "$ROOT_DIR/ops/sunny/training-mode.sh" --restore-demo
  capture_cmd "audit-after-restore" bash "$ROOT_DIR/ops/sunny/audit-demo-stack.sh"
fi

python3 "$ROOT_DIR/ops/sunny/summarize_report_status.py" \
  "$STATUS_FILE" \
  "$REPORT_DIR/summary.json" \
  "$OUTPUT_DIR"

log ""
log "REPORT_DIR=$REPORT_DIR"
log "OUTPUT_DIR=$OUTPUT_DIR"
