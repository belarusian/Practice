#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
POWERSHELL="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
NVIDIA_SMI="/mnt/c/Windows/System32/nvidia-smi.exe"

section() {
  printf '\n== %s ==\n' "$1"
}

check_http() {
  local label="$1"
  local method="$2"
  local url="$3"
  local code

  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 -X "$method" "$url" || true)"
  if [[ "$code" =~ ^[23][0-9][0-9]$ || "$code" == "405" ]]; then
    printf 'ok   %-24s %s (%s)\n' "$label" "$url" "$code"
  else
    printf 'bad  %-24s %s (%s)\n' "$label" "$url" "${code:-000}"
  fi
}

section "Host"
hostname
whoami
date
uptime

section "WSL2 Services"
systemctl --type=service --state=running | egrep 'ssh|nginx|postgres|turn|coturn|wg|synapse|voice' || true

section "WSL2 Service Status"
systemctl is-active ssh nginx postgresql wg-quick@wg0 coturn synapse voice-phone.service 2>/dev/null || true

section "WSL2 Listening Ports"
ss -tulpn | egrep ':(22|5432|8008|8080|8765|3478|5349|51820) ' || true

section "WSL2 Health Checks"
check_http "synapse" "GET" "http://127.0.0.1:8008/_matrix/client/versions"
check_http "element/nginx" "GET" "http://127.0.0.1:8080/"
check_http "phone webhook" "POST" "http://127.0.0.1:8765/incoming"

section "WireGuard"
ip addr show wg0 2>/dev/null || true
sudo wg show 2>/dev/null || true

section "GPU"
if [[ -x "$NVIDIA_SMI" ]]; then
  "$NVIDIA_SMI" --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader || true
else
  echo "nvidia-smi.exe not found at $NVIDIA_SMI"
fi

section "Windows Demo Processes"
if [[ -x "$POWERSHELL" ]]; then
  "$POWERSHELL" -NoProfile -ExecutionPolicy Bypass -File "$(wslpath -w "$ROOT_DIR/ops/sunny/reinit-demo-stack.ps1")" -StatusOnly
else
  echo "powershell.exe not found at $POWERSHELL"
fi
