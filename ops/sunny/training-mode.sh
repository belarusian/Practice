#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
POWERSHELL="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
NVIDIA_SMI="/mnt/c/Windows/System32/nvidia-smi.exe"
MIN_FREE_VRAM_MB="${MIN_FREE_VRAM_MB:-4096}"

DRY_RUN=false
RESTORE_DEMO=false
AUDIT_AFTER_STOP=false

usage() {
  cat <<'EOF'
Usage:
  bash ops/sunny/training-mode.sh [--dry-run] [--audit-after-stop]
  bash ops/sunny/training-mode.sh --restore-demo

Behavior:
  Default mode stops the Windows-side GPU demo services so Sunny can be used
  for training, then runs a local GPU preflight.

Options:
  --dry-run           Show what would be stopped without changing process state
  --audit-after-stop  Run audit-demo-stack.sh after stopping demo GPU services
  --restore-demo      Restore the Windows demo layer using reinit-demo-stack.ps1
  --help              Show this help text
EOF
}

section() {
  printf '\n== %s ==\n' "$1"
}

show_gpu_summary() {
  if [[ ! -x "$NVIDIA_SMI" ]]; then
    echo "nvidia-smi.exe not found at $NVIDIA_SMI"
    return 1
  fi

  "$NVIDIA_SMI" \
    --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu \
    --format=csv,noheader
}

training_preflight() {
  local gpu_line total_mb used_mb util_pct free_mb

  if [[ ! -x "$NVIDIA_SMI" ]]; then
    echo "training preflight failed: nvidia-smi.exe not found at $NVIDIA_SMI" >&2
    return 1
  fi

  gpu_line="$("$NVIDIA_SMI" --query-gpu=memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits | head -n 1)"
  total_mb="$(printf '%s\n' "$gpu_line" | awk -F',' '{gsub(/ /, "", $1); print $1}')"
  used_mb="$(printf '%s\n' "$gpu_line" | awk -F',' '{gsub(/ /, "", $2); print $2}')"
  util_pct="$(printf '%s\n' "$gpu_line" | awk -F',' '{gsub(/ /, "", $3); print $3}')"
  free_mb=$(( total_mb - used_mb ))

  section "Training Preflight"
  printf 'total_vram_mb=%s used_vram_mb=%s free_vram_mb=%s gpu_util_percent=%s\n' \
    "$total_mb" "$used_mb" "$free_mb" "$util_pct"

  if (( free_mb < MIN_FREE_VRAM_MB )); then
    printf 'STATUS: not ready for training, free VRAM below threshold (%s MiB required)\n' "$MIN_FREE_VRAM_MB" >&2
    return 1
  fi

  echo "STATUS: Sunny GPU preflight passed"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      ;;
    --restore-demo)
      RESTORE_DEMO=true
      ;;
    --audit-after-stop)
      AUDIT_AFTER_STOP=true
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

if [[ ! -x "$POWERSHELL" ]]; then
  echo "powershell.exe not found at $POWERSHELL" >&2
  exit 1
fi

if [[ "$RESTORE_DEMO" == true ]]; then
  section "Restore Demo Services"
  "$POWERSHELL" -NoProfile -ExecutionPolicy Bypass -File "$(wslpath -w "$ROOT_DIR/ops/sunny/training-mode.ps1")" -RestoreDemo
  section "Audit"
  bash "$ROOT_DIR/ops/sunny/audit-demo-stack.sh"
  exit 0
fi

section "Current GPU"
show_gpu_summary || true

section "Stop Demo GPU Services"
PS_ARGS=()
if [[ "$DRY_RUN" == true ]]; then
  PS_ARGS+=(-DryRun)
fi

"$POWERSHELL" -NoProfile -ExecutionPolicy Bypass -File "$(wslpath -w "$ROOT_DIR/ops/sunny/training-mode.ps1")" "${PS_ARGS[@]}"

if [[ "$DRY_RUN" == true ]]; then
  section "Dry Run Complete"
  echo "No process state was changed."
  exit 0
fi

section "GPU After Stop"
show_gpu_summary || true

if [[ "$AUDIT_AFTER_STOP" == true ]]; then
  section "Audit After Stop"
  bash "$ROOT_DIR/ops/sunny/audit-demo-stack.sh"
fi

training_preflight

