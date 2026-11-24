#!/usr/bin/env bash
# Shared checkout-local state paths for start.sh and stop.sh.
# Variables are consumed by the sourcing scripts.
# shellcheck disable=SC2034
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The checkout holding these scripts, unless DEMO_CHECKOUT names another (e.g. another worktree).
REPO_ROOT="$(git -C "${DEMO_CHECKOUT:-$SCRIPT_DIR}" rev-parse --show-toplevel)"
CHECKOUT_ID="$(printf '%s' "$REPO_ROOT" | git hash-object --stdin)"
TMP_ROOT="${TMPDIR:-/tmp}"
STATE_DIR="${TMP_ROOT%/}/aqueduct-local-demo-${CHECKOUT_ID}"
PID_FILE="$STATE_DIR/django.pid"
DEX_CONFIG="$STATE_DIR/dex.yaml"
DEX_PAGE="$STATE_DIR/dex-password.html"
LOG_FILE="$STATE_DIR/django.log"
DEX_CONTAINER="aqueduct-local-demo-${CHECKOUT_ID}"
