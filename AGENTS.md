# Agent Guidelines for Aqueduct AI Gateway

## Project Structure
- Django project root is in the `aqueduct/` subdirectory
- All `manage.py` commands must be run from `aqueduct/` directory
- Project uses `uv` for Python package management

## Apps
- `management`: Core data/domain + UI app (orgs, users, tokens, usage/request tracking, shared gateway-related models, and frontend/admin surfaces).
- `gateway`: API gateway app (endpoint views, auth/routing/integration logic, and most integration tests).

## Build/Lint/Test Commands
- **Run tests**: `cd aqueduct && uv run python manage.py test` or `cd aqueduct && uv run python manage.py test <app_name>.tests.<TestClass>` (run full suite before finishing major/cross-cutting work)
- **Run single test**: `cd aqueduct && uv run python manage.py test <app_name>.tests.<TestClass>.test_method` (prefer targeted tests during active development)
- **Lint**: `uv run pre-commit run --all-files`
- **Format**: No auto-formatting configured, follow existing style
- **Migrations**: `cd aqueduct && uv run python manage.py makemigrations` then `cd aqueduct && uv run python manage.py migrate`

## Test Mock API (OpenAI)
- Test runner: `gateway.tests.utils.test_runner.MockServerTestRunner` (enabled via `TEST_RUNNER` in `aqueduct/aqueduct/settings.py`)
- Toggle: `TESTS_USE_MOCK_API` env var (`True` by default) controls starting the mock server
- Shared server: one FastAPI mock server for entire suite; base class patches `OPENAI_BASE_URL`/`OPENAI_API_KEY` and points files API to mock
- Default mocks live in `aqueduct/mock_api/mock_configs.py` (`default_post_configs`, `default_get_configs`, `default_delete_configs`, `default_post_stream_configs`)
- Per-test override: `with self.mock_server.patch_external_api("chat/completions", MockConfig(...)):`
- Response types: `MockConfig` (JSON), `MockStreamingConfig` (SSE bytes), `MockPlainTextConfig` (text)
- Streaming behavior: mock server uses request JSON `"stream": true` to pick `default_post_stream_configs`
- Paths: use normalized endpoint paths (no `/v1/`); dynamic IDs use `id` wildcard in config keys (e.g. `vector_stores/id/files`)
- Add new endpoint mock by adding a config entry in `mock_configs.py` (avoid new FastAPI routes unless behavior cannot be expressed by config)
- Use `reset`/`patch_external_api` to avoid cross-test bleed when custom mocks are set

## Checklist
- Scope: identify touched app/module, entrypoints, and side effects (DB/auth/external API)
- Baseline: check `git status` and read nearby implementation/tests in the same area
- Pattern-match first: find 1-3 similar endpoints/tests and mirror structure, naming, and error style
- Reuse: prefer existing helpers/base test classes/fixtures over new abstractions
- Plan smallest diff that satisfies the request

## Change Policy
- Prefer consistency over novelty; follow local conventions before introducing new patterns
- Keep diffs minimal and task-focused; avoid unrelated refactors/renames
- Reuse existing utilities and extension points whenever possible
- Do not silently change contracts (status codes, payload shape, defaults, limits, auth behavior) unless requested
- If deviating from an established pattern, call it out explicitly in handoff

## Notes
- use the `todowrite` tool to plan your work
