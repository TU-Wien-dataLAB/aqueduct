---
name: aqueduct-local-demo
description: Start a local Aqueduct demo for manual UI testing. Use when the user asks to run or preview the Aqueduct interface locally.
---

# Aqueduct local demo

Run Aqueduct with Dex SSO and an isolated SQLite database, seeded with test data: a user for each role (admin, org admin, user), teams, keys, and ten days of requests on each key. Dex's login page shows one button per user, so signing in takes no typing.

Resolve script paths relative to this skill's directory, not the working directory. The scripts locate the checkout themselves.

## Start

Run `scripts/start.sh`.

**Done when** it prints the login URL, the users, and `Smoke test passed`, proving the OIDC login through Dex's user buttons and the token-create page work end to end.

The script prints a checkout-specific state directory under `${TMPDIR:-/tmp}` containing the database, logs, and process state. The normal checkout database is untouched. Django and Dex listen on loopback only. Docker must be running; the script checks required CLI tools.

## Configure

Pass feature settings inline when starting, for example:

```bash
MAX_ACCESS_KEY_EXPIRATION_DAYS=7 <skill-directory>/scripts/start.sh
```

Use `WEB_PORT` and `DEX_PORT` to override the script's default ports. Set `DEMO_CHECKOUT` to another checkout's path, such as another worktree, to run that checkout with these scripts; pass the same value to `stop.sh`.

Feature settings are inherited. The demo owns its database, Django settings module, debug mode, OIDC endpoints and credentials, and user roles. `SECRET_KEY`, `ALLOWED_HOSTS`, and `LITELLM_ROUTER_CONFIG_FILE_PATH` accept overrides with local-demo defaults; the seeded requests use the models of that router config. SQLite is always used, even when the caller normally uses PostgreSQL. Caches and task queues are in-memory; background jobs are not processed and existing Redis/Celery services are untouched.

## Test data

`PEOPLE` in `scripts/seed.py` is the one list of users: it sets each user's role, org, teams, and keys, and generates their Dex logins (password `1234`) and their buttons on Dex's login page, which replaces Dex's `password.html` template. A Dex image upgrade may change that template's fields; the page uses `.PostURL` and `.Invalid`.

## Hand off

Give the user:

- Login URL and the users printed by the script.
- Instructions: click **Sign in with SSO**, then a user. To switch users, log out and sign in again.
- API-key form URL printed by the script.
- Any feature-setting overrides applied for this run.
- State directory printed by the script, for logs and reset.

## Stop or reset

Run `scripts/stop.sh`.

**Done when** this demo's Django process and Dex container have exited and their ports refuse connections.

Data persists across restarts until the OS removes temporary files. For fresh test data, with requests up to today, stop it, delete only `db.sqlite3` inside the printed state directory, then start again.

## Restart behavior

Starting stops the previous demo for the same checkout, preserves its database, and rewrites the login snippet that sets each user's role and org. Other active config snippets in the demo database are deactivated. Test data is added only for users the database does not have yet. Different checkouts have separate state and Dex containers; choose different ports to run them simultaneously.
