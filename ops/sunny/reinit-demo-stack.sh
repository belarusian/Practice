#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAB_ENV_FILE="${SUNNY_LAB_ENV_FILE:-$ROOT_DIR/ops/sunny/lab.env}"
if [[ -f "$LAB_ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$LAB_ENV_FILE"
  set +a
fi
POWERSHELL="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
SERVICES=(
  ssh
  nginx
  postgresql
  wg-quick@wg0
  coturn
  synapse
  voice-phone.service
)

section() {
  printf '\n== %s ==\n' "$1"
}

section "Starting WSL2 Services"
sudo systemctl start "${SERVICES[@]}"
systemctl is-active "${SERVICES[@]}"

section "WSL2 Health"
curl -fsS --max-time 5 http://127.0.0.1:8008/_matrix/client/versions >/dev/null
curl -fsS --max-time 5 http://127.0.0.1:8080/ >/dev/null
curl -fsS --max-time 5 -X POST http://127.0.0.1:8765/incoming >/dev/null
echo "WSL2 services responded successfully."

section "Starting Windows Demo Processes"
if [[ ! -x "$POWERSHELL" ]]; then
  echo "powershell.exe not found at $POWERSHELL"
  exit 1
fi

"$POWERSHELL" -NoProfile -ExecutionPolicy Bypass -File "$(wslpath -w "$ROOT_DIR/ops/sunny/reinit-demo-stack.ps1")"

section "Audit"
bash "$ROOT_DIR/ops/sunny/audit-demo-stack.sh"
