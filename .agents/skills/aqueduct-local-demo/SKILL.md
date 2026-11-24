---
name: aqueduct-local-demo
description: Start a local Aqueduct demo for manual UI testing, or benchmark the management pages on PostgreSQL. Use when the user asks to run or preview the Aqueduct interface locally, or to measure page speed or query counts.
---

# Aqueduct local demo

Run Aqueduct with Dex SSO and an isolated SQLite database, seeded with test data: a user for each role (admin, org admin, user), teams, keys, ten days of requests, and files and batches. Dex's login page shows one button per user, so signing in takes no typing.

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

Choose the request volume by what the user wants to check:

| Purpose | Setting | Seeding time |
|---|---|---|
| Review a UI change | default, `DEMO_REQUESTS=1500` | 2 s |
| Check page speed and query cost | [Benchmark](#benchmark) mode, not this demo | about 60 s |

`DEMO_REQUESTS` is the number of requests the database holds after seeding, spread over all keys and the last `DEMO_DAYS` days (default 10). Each start tops the database up to that number, so raising it on an existing demo adds the difference; lowering it removes nothing. This demo is for looking, not timing: SQLite runs the pages about ten times slower than PostgreSQL, with other query plans, and Silk, on here, EXPLAINs every SELECT.

The keys in `FILE_KEYS` in `scripts/seed.py` each get two plain files and a batch in every status, with input, output and error files. These records exist only in the demo database: the demo sets no `AQUEDUCT_FILES_API_URL`, so the pages list them, but Delete file and Cancel batch report a failure.

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

Starting stops the previous demo for the same checkout, preserves its database, and rewrites the login snippet that sets each user's role and org. Other active config snippets in the demo database are deactivated. Users, teams and keys are added only for users the database does not have yet; requests are topped up to `DEMO_REQUESTS`, and a key in `FILE_KEYS` gets files and batches if it has no files. Different checkouts have separate state and Dex containers; choose different ports to run them simultaneously.

## Benchmark

Use the benchmark mode to time the management pages, count their queries, or show that a change leaves what they show untouched. `scripts/bench.sh` runs the pages on a throwaway PostgreSQL 15 container (the version `docker-compose.yaml` runs) with Silk off, seeded by `scripts/seed.py`. The container has 1 GB of shared buffers, so the seeded requests stay cached and a page's time does not depend on what the pages before it read. It leaves this demo's SQLite state and the caller's own database alone; like the demo, its container and results directory belong to the checkout.

1. `bench.sh up` starts the database, migrates, seeds `DEMO_REQUESTS` (default `1000000`, about a minute) and refreshes the planner statistics. **Done when** it prints `Requests: 1000000`. Later runs reuse the data, so one seeded database serves every commit of a branch; run `up` again after a migration.
2. At each commit, run `bench.sh profile` and `bench.sh snapshot`. They write `profile-<commit>.json` and `snapshot-<commit>.json` to the results directory `up` printed; `--out FILE` picks another path.
3. `bench.sh compare BASE NEW` diffs two snapshots. **Done when** it prints `Snapshots match`. Otherwise each line names `role | page | key path: before -> after`. `--ignore KEY` skips a context key that a change alters on purpose; report every ignored key with the result.
4. `bench.sh table FILE...` prints the profiles as a Markdown table: a row per role and page, a column per file, headed by its commit. `--metric sql_ms` or `--metric queries` replaces wall time; `--role org-admin` keeps one role.
5. `bench.sh down` removes the database and keeps the results.

`bench.sh manage ARGS` runs `manage.py` against the benchmark database, for example `manage shell` to time one query against its rewrite.

### Reading the results

- The profiler signs in as the platform admin, the org admin and a plain member (`admin`, `org-admin`, `member`). It requests every sidebar page, a key page, a team page, and filtered variants of the Requests log and the Usage page through Django's test client, in-process: no server, network or browser.
- Per page, `wall_ms` is the whole request, view and template; `sql_ms` the time spent executing statements; `queries` their number. Each is from the fastest of five warm runs (`--runs N`); `wall_median_ms` shows the spread.
- Timings are relative to the machine. Compare columns measured on one idle machine in one session; differences of a few milliseconds or about 10% are noise. Query counts are exact.
- Profile and snapshot freeze the clock at the newest seeded request, so "last 24 h" covers the same rows on every run, however old the database.
- A snapshot records, per page and role, the dict the view hands its template (`response.context_data`): totals, chart series, table rows with what the view set on them, counts and pagination, as plain JSON with floats rounded to six places. The pages only read the data, so snapshots of different commits compare as long as nothing was seeded between them; `compare` refuses snapshots of different data.
