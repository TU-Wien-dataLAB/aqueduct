---
name: aqueduct-local-demo
description: Start a local Aqueduct demo for manual UI testing. Use when the user asks to run or preview the Aqueduct interface locally.
---

# Aqueduct local demo

Run Aqueduct with Dex SSO and an isolated SQLite database, seeded with the `default` org and `Demo Team`. The demo user is an org admin.

Resolve script paths relative to this skill's directory, not the working directory. The scripts locate the checkout themselves.

## Start

Run `scripts/start.sh`.

**Done when** it prints the login URL and `Smoke test passed`, proving the OIDC login and token-create page work end to end.

The script prints a checkout-specific state directory under `${TMPDIR:-/tmp}` containing the database, logs, and process state. The normal checkout database is untouched. Django and Dex listen on loopback only. Docker must be running; the script checks required CLI tools.

## Configure

Pass feature settings inline when starting, for example:

```bash
MAX_ACCESS_KEY_EXPIRATION_DAYS=7 <skill-directory>/scripts/start.sh
```

Use `WEB_PORT`, `DEX_PORT`, and `DEMO_EMAIL` to override the script's defaults. The demo password is fixed at `1234`.

Feature settings are inherited. The demo owns its database, Django settings module, debug mode, OIDC endpoints and credentials, and superuser email. `SECRET_KEY` and `ALLOWED_HOSTS` accept overrides with local-demo defaults. SQLite is always used, even when the caller normally uses PostgreSQL. Caches and task queues are in-memory; background jobs are not processed and existing Redis/Celery services are untouched.

## Hand off

Give the user:

- Login URL and demo credentials printed by the script.
- Instructions: click **Login with SSO**, sign in, then **Grant Access** on the Dex approval screen.
- API-key form URL printed by the script.
- Any feature-setting overrides applied for this run.
- State directory printed by the script, for logs and reset.

## Stop or reset

Run `scripts/stop.sh`.

**Done when** this demo's Django process and Dex container have exited and their ports refuse connections.

Data persists across restarts until the OS removes temporary files. For a clean demo, stop it, delete only `db.sqlite3` inside the printed state directory, then start again.

## Restart behavior

Starting stops the previous demo for the same checkout, preserves its database, and reseeds its org, team, and org-admin login snippet. Other active config snippets in the demo database are deactivated. Different checkouts have separate state and Dex containers; choose different ports to run them simultaneously.
