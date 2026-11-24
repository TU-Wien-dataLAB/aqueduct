#!/usr/bin/env bash
# Benchmark mode: the management pages on a throwaway PostgreSQL database, Silk off.
#
#   bench.sh up                     start the database, migrate, seed DEMO_REQUESTS (1000000)
#   bench.sh profile [--out FILE]   time every page as each role (bench.py profile --help)
#   bench.sh snapshot [--out FILE]  record what every page shows as each role
#   bench.sh compare BASE NEW       name the pages, roles and keys where two snapshots differ
#   bench.sh table FILE...          print profiles as a Markdown table, a column per file
#   bench.sh manage ARGS...         run manage.py against the benchmark database
#   bench.sh down                   remove the database
set -euo pipefail

# shellcheck source=state.sh
source "$(dirname "${BASH_SOURCE[0]}")/state.sh"
DJANGO_ROOT="$REPO_ROOT/aqueduct"
BENCH_DIR="${TMP_ROOT%/}/aqueduct-bench-${CHECKOUT_ID}"
BENCH_CONTAINER="aqueduct-bench-${CHECKOUT_ID}"
POSTGRES_IMAGE="postgres:15" # The version docker-compose.yaml runs.
# psycopg is the optional "postgresql" extra; --with adds it without changing the project's venv.
PSYCOPG="psycopg[binary]>=3.3.4"

mkdir -p "$BENCH_DIR"
chmod 700 "$BENCH_DIR"
export AQUEDUCT_DEMO_STATE_DIR="$BENCH_DIR"
export AQUEDUCT_BENCH_DIR="$BENCH_DIR" AQUEDUCT_BENCH_REPO="$REPO_ROOT"
export DJANGO_SETTINGS_MODULE=bench_settings
export PYTHONPATH="$SCRIPT_DIR:$DJANGO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SECRET_KEY="${SECRET_KEY:-local-bench-secret-key}"
export LITELLM_ROUTER_CONFIG_FILE_PATH="${LITELLM_ROUTER_CONFIG_FILE_PATH:-$REPO_ROOT/example_router_config.yaml}"

for command in docker uv git; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Required command not found: $command" >&2
    exit 1
  fi
done

database_port() {
  local address
  if ! address="$(docker port "$BENCH_CONTAINER" 5432/tcp 2>/dev/null | head -1)" \
    || [[ -z "$address" ]]; then
    echo "The benchmark database is not running. Start it with: $0 up" >&2
    exit 1
  fi
  export AQUEDUCT_BENCH_DB_PORT="${address##*:}"
}

django() {
  database_port
  cd "$DJANGO_ROOT"
  uv run --frozen --quiet --with "$PSYCOPG" "$@"
}

psql_bench() {
  docker exec "$BENCH_CONTAINER" psql -U aqueduct -d aqueduct -v ON_ERROR_STOP=1 -qtAc "$1"
}

up() {
  if ! docker container inspect "$BENCH_CONTAINER" >/dev/null 2>&1; then
    echo "Starting PostgreSQL ($POSTGRES_IMAGE)..."
    # The seeded requests and their indexes, about 300 MB, overflow the default 128 MB of shared
    # buffers; then a page's time depends on what the pages before it read.
    docker run -d --name "$BENCH_CONTAINER" \
      -e POSTGRES_USER=aqueduct -e POSTGRES_PASSWORD=aqueduct -e POSTGRES_DB=aqueduct \
      -p 127.0.0.1::5432 "$POSTGRES_IMAGE" -c shared_buffers=1GB >/dev/null
  else
    docker start "$BENCH_CONTAINER" >/dev/null
  fi
  # Over TCP: during its first start the image runs a socket-only server, then restarts.
  for _ in $(seq 1 60); do
    if docker exec "$BENCH_CONTAINER" pg_isready -h 127.0.0.1 -U aqueduct -d aqueduct \
      >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done

  echo "Migrating..."
  django python manage.py migrate --noinput --verbosity 0
  local requests="${DEMO_REQUESTS:-1000000}"
  echo "Seeding up to $requests requests (about a minute for a million on a fresh database)..."
  # The benchmark signs in without Dex, so the Dex logins and login page go nowhere.
  DEMO_REQUESTS="$requests" DEX_CONFIG=/dev/null DEX_PAGE=/dev/null \
    django python manage.py shell --no-imports --verbosity 0 <"$SCRIPT_DIR/seed.py" >/dev/null
  # Fresh planner statistics, so every run sees the same query plans.
  psql_bench "VACUUM ANALYZE"
  cat <<EOF

Benchmark database is ready.
  Requests:  $(psql_bench "SELECT count(*) FROM management_request")
  Container: $BENCH_CONTAINER (127.0.0.1:$AQUEDUCT_BENCH_DB_PORT)
  Results:   $BENCH_DIR
  Next:      $0 profile, $0 snapshot
EOF
}

command="${1:-}"
shift || true
case "$command" in
  up) up ;;
  profile | snapshot) django python "$SCRIPT_DIR/bench.py" "$command" "$@" ;;
  compare | table)
    cd "$DJANGO_ROOT"
    uv run --frozen --quiet python "$SCRIPT_DIR/bench.py" "$command" "$@"
    ;;
  manage) django python manage.py "$@" ;;
  down)
    docker rm -f -v "$BENCH_CONTAINER" >/dev/null 2>&1 || true
    echo "Benchmark database removed. Results kept in $BENCH_DIR"
    ;;
  *)
    sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//' >&2
    exit 2
    ;;
esac
