#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=state.sh
source "$(dirname "${BASH_SOURCE[0]}")/state.sh"
DJANGO_ROOT="$REPO_ROOT/aqueduct"
DEX_CONFIG_TEMPLATE="$REPO_ROOT/mock-dex-config.yaml"

mkdir -p "$STATE_DIR"
chmod 700 "$STATE_DIR"
export AQUEDUCT_DEMO_STATE_DIR="$STATE_DIR"
export DJANGO_SETTINGS_MODULE=demo_settings
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

WEB_PORT="${WEB_PORT:-8000}"
DEX_PORT="${DEX_PORT:-5556}"
DEMO_EMAIL="${DEMO_EMAIL:-you@example.com}"
DEMO_PASSWORD="1234"
SECRET_KEY="${SECRET_KEY:-local-demo-secret-key}"
ALLOWED_HOSTS="${ALLOWED_HOSTS:-localhost 127.0.0.1}"

for command in docker uv curl sed; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Required command not found: $command" >&2
    exit 1
  fi
done

# Make start idempotent: tear down any previous demo first.
"$SCRIPT_DIR/stop.sh" >/dev/null 2>&1 || true

if command -v lsof >/dev/null 2>&1; then
  for port in "$WEB_PORT" "$DEX_PORT"; do
    if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
      echo "Port $port is already in use. Stop that process and retry." >&2
      exit 1
    fi
  done
fi

export WEB_PORT="${WEB_PORT}"
export DEX_PORT="${DEX_PORT}"
export DJANGO_DEBUG=True
export ALLOWED_HOSTS="${ALLOWED_HOSTS}"
export SECRET_KEY="${SECRET_KEY}"
export OIDC_OP_JWKS_ENDPOINT="http://localhost:${DEX_PORT}/dex/keys"
export OIDC_RP_CLIENT_ID="example-app"
export OIDC_RP_CLIENT_SECRET="ZXhhbXBsZS1hcHAtc2VjcmV0"
export OIDC_OP_AUTHORIZATION_ENDPOINT="http://localhost:${DEX_PORT}/dex/auth"
export OIDC_OP_TOKEN_ENDPOINT="http://localhost:${DEX_PORT}/dex/token"
export OIDC_OP_USER_ENDPOINT="http://localhost:${DEX_PORT}/dex/userinfo"
export ADMIN_SUPERUSER_EMAILS="${DEMO_EMAIL}"

sed \
  -e "s|http://localhost:5556|http://localhost:${DEX_PORT}|g" \
  -e "s|http://localhost:8000|http://localhost:${WEB_PORT}|g" \
  -e "s|you@example.com|${DEMO_EMAIL}|g" \
  "$DEX_CONFIG_TEMPLATE" >"$DEX_CONFIG"

echo "Starting Dex SSO..."
docker run -d --rm --name "$DEX_CONTAINER" \
  -p "127.0.0.1:${DEX_PORT}:5556" \
  -v "$DEX_CONFIG:/etc/dex/config.yaml:ro" \
  dexidp/dex:v2.42.1 dex serve /etc/dex/config.yaml >/dev/null

echo "Preparing the local database..."
cd "$DJANGO_ROOT"
uv run --frozen python manage.py migrate --noinput --verbosity 0

uv run --frozen python - <<'PY'
import django

django.setup()

from management.models import Org, Snippet, SnippetType, Team

org, _ = Org.objects.get_or_create(name="default")
Team.objects.get_or_create(name="Demo Team", org=org)
Snippet.objects.filter(type=SnippetType.CONFIG, active=True).update(active=False)
Snippet.objects.create(
    name="Local demo org admin",
    type=SnippetType.CONFIG,
    active=True,
    code=(
        "class LocalDemoConfig(ConfigSnippet):\n"
        "    def user_group(self, claims):\n"
        "        return 'org-admin'\n"
    ),
)
print("Seeded demo org, team, and org-admin login snippet.")
PY

echo "Starting Django..."
nohup uv run --frozen python manage.py runserver "127.0.0.1:${WEB_PORT}" --noreload \
  >"$LOG_FILE" 2>&1 &
echo $! >"$PID_FILE"

ready=0
for _ in $(seq 1 30); do
  if curl -fsS -o /dev/null "http://localhost:${WEB_PORT}/login/" 2>/dev/null \
    && curl -fsS -o /dev/null "http://localhost:${DEX_PORT}/dex/keys" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 1
done

if [[ "$ready" -ne 1 ]]; then
  echo "The demo did not become ready. Last log lines:" >&2
  tail -20 "$LOG_FILE" >&2 || true
  exit 1
fi

DEMO_WEB_URL="http://localhost:${WEB_PORT}" \
DEMO_DEX_URL="http://localhost:${DEX_PORT}" \
DEMO_EMAIL="${DEMO_EMAIL}" \
DEMO_PASSWORD="${DEMO_PASSWORD}" \
uv run --frozen python "$SCRIPT_DIR/smoke_test.py"

cat <<EOF

Aqueduct local demo is ready.
  Login:  http://localhost:${WEB_PORT}/login/
  User:   ${DEMO_EMAIL}
  Pass:   ${DEMO_PASSWORD}
  Tokens: http://localhost:${WEB_PORT}/aqueduct/management/tokens/create/
  State:  ${STATE_DIR}
  Log:    ${LOG_FILE}
  Stop:   ${SCRIPT_DIR}/stop.sh
EOF
