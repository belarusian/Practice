#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
POWERSHELL="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
REPORT_STAMP="$(date +%Y%m%d-%H%M%S)"
REPORT_DIR="$ROOT_DIR/artifacts/sunny-reports/$REPORT_STAMP"
TARGET="vision"
DEVICE="cuda"
WITH_TRAINING_MODE=false
RESTORE_DEMO=false
RESTORE_TIMEOUT_SECONDS="${RESTORE_TIMEOUT_SECONDS:-600}"

usage() {
  cat <<'EOF'
Usage:
  bash ops/sunny/prove-training-runtime.sh [options]

Options:
  --report-dir PATH      Report directory on Sunny
  --target NAME          Training target to probe (default: vision)
  --device NAME          Device to probe for ml-lab check (default: cuda)
  --with-training-mode   Stop Windows GPU demo services before probing
  --restore-demo         Restore the demo stack after probing
  --help                 Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --report-dir)
      REPORT_DIR="$2"
      shift
      ;;
    --target)
      TARGET="$2"
      shift
      ;;
    --device)
      DEVICE="$2"
      shift
      ;;
    --with-training-mode)
      WITH_TRAINING_MODE=true
      ;;
    --restore-demo)
      RESTORE_DEMO=true
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

mkdir -p "$REPORT_DIR"

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

capture_text() {
  local name="$1"
  local text="$2"
  local output_file="$REPORT_DIR/${name}.txt"

  printf '%s\n' "$text" >"$output_file"
  printf '%s\t0\t%s\n' "$name" "$output_file" >>"$STATUS_FILE"
  log ""
  log "== $name =="
  cat "$output_file" | tee -a "$LOG_FILE"
}

capture_cmd "host-meta" bash -lc "hostname && whoami && date && pwd"
capture_cmd "audit-before" bash "$ROOT_DIR/ops/sunny/audit-demo-stack.sh"

if [[ "$WITH_TRAINING_MODE" == true ]]; then
  capture_cmd "training-mode" bash "$ROOT_DIR/ops/sunny/training-mode.sh"
fi

capture_cmd "wsl-python-version" bash -lc "python3 --version"
capture_cmd "wsl-tooling" bash -lc "command -v python3 || true; command -v uv || true; ls /dev/nvidia* 2>/dev/null || true; command -v /mnt/c/Windows/System32/nvidia-smi.exe || true"
capture_cmd "wsl-modules" python3 - <<'PY'
import importlib.util
import json
import sys

modules = ["torch", "torchvision", "torchaudio", "transformers", "datasets"]
payload = {
    "python_version": sys.version,
    "modules": {name: importlib.util.find_spec(name) is not None for name in modules},
}
print(json.dumps(payload, indent=2))
PY

capture_cmd "wsl-ml-lab-check" bash -lc "cd '$ROOT_DIR' && PYTHONPATH=src python3 -m industry_ml_lab.cli check --target '$TARGET' --device '$DEVICE' --output-dir artifacts/remote-proof --json"

if [[ -x "$POWERSHELL" ]]; then
  capture_cmd \
    "windows-probe-console" \
    "$POWERSHELL" \
    -NoProfile \
    -ExecutionPolicy Bypass \
    -File "$(wslpath -w "$ROOT_DIR/ops/sunny/prove-training-runtime.ps1")" \
    -ReportDir "$(wslpath -w "$REPORT_DIR")" \
    -RepoRoot "$(wslpath -w "$ROOT_DIR")" \
    -RepoSrc "$(wslpath -w "$ROOT_DIR/src")" \
    -Target "$TARGET" \
    -Device "$DEVICE"
else
  capture_text "windows-probe-console" "powershell.exe not found at $POWERSHELL"
fi

if [[ "$RESTORE_DEMO" == true ]]; then
  capture_cmd "restore-demo" timeout "$RESTORE_TIMEOUT_SECONDS" bash "$ROOT_DIR/ops/sunny/training-mode.sh" --restore-demo
  capture_cmd "audit-after-restore" bash "$ROOT_DIR/ops/sunny/audit-demo-stack.sh"
fi

python3 "$ROOT_DIR/ops/sunny/summarize_report_status.py" --proof \
  "$STATUS_FILE" \
  "$REPORT_DIR/summary.json"

log ""
log "REPORT_DIR=$REPORT_DIR"
