#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=state.sh
source "$(dirname "${BASH_SOURCE[0]}")/state.sh"

if [[ -f "$PID_FILE" ]]; then
  pid="$(<"$PID_FILE")"
  # Confirm this is still our server before signalling a potentially reused PID.
  if ps -p "$pid" -o command= | grep -F "manage.py runserver" >/dev/null; then
    pkill -P "$pid" 2>/dev/null || true
    kill "$pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
fi

docker rm -f "$DEX_CONTAINER" >/dev/null 2>&1 || true
rm -f "$DEX_CONFIG"

echo "Aqueduct local demo stopped. Data retained in $STATE_DIR"
