<div align="center">
  <img src="./docs/assets/Aqueduct Icon.png" width="50%" alt="Aqueduct AI Gateway Logo" />
</div>

# Aqueduct AI Gateway

[![GitHub License](https://img.shields.io/github/license/tu-wien-datalab/aqueduct)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/tu-wien-datalab/aqueduct)](https://github.com/tu-wien-datalab/aqueduct/commits/main)
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)


> [!NOTE]
> This project is still in active development! Contributions welcome!

**Aqueduct AI Gateway** aims to provide a **simple yet fully-featured** AI gateway you can self-host with:

- no [SSO tax](https://konghq.com/pricing)
- no [observability tax](https://www.litellm.ai/enterprise)
- no [self-hosting tax](https://portkey.ai/pricing)
- no [org management tax](https://www.litellm.ai/enterprise)
- etc.

We aim to achieve this by:

- Using a **pass-through approach** that avoids re-implementing the entire OpenAI-compatible APIs or every new feature, and
- Building on **Django** for a clean, efficient, and maintainable implementation.

If you don’t need user self-service, take a look at [Envoy AI Gateway](https://aigateway.envoyproxy.io) instead!

![AI Gateway Architecture](./docs/assets/screenshot.png "AI Gateway Architecture")

## 🚀 Implementation Roadmap

This project aims to use Django for user management/API with minimal additional implementation to create a comprehensive
AI gateway. The implementation follows a phased approach:

1. ✅ ~~**Data Model and Administrative Interface:**~~
    * ~~Implementation of "Teams" and "Organizations" within the data model.~~
    * ~~Development of an administrative UI using Django for user, team and token management.~~

2. 🚧 **Gateway Relay and Usage Tracking:**
    * ~~Development of a gateway server to relay requests to LLM providers.~~
    * ~~Request routing to specific provider endpoints based on model selection.~~
    * ~~Request parsing for usage tracking.~~
    * ~~Pre-/Post-processing of requests (e.g. to correctly parse the `/models` endpoint to list available models).~~
    * Add usage checks as pre- and post-processing steps to limit requests.
    * A functional `docker-compose.yml`
    * Database schema optimization for write-heavy request/usage logging.

3. 🔄 **Advanced Features:**
    * A `/metrics` endpoint for monitoring.
    * Dashboard to track usage of orgs, teams and users.
    * Implement [Guardrails](https://github.com/guardrails-ai/guardrails) using post-processing on completions. 
    * Model cooldown and retry logic.
    * Granular access control based on models and usage limits.
    * Management of MCP tool calling server endpoints e.g. from
      the [MCP Server list](https://github.com/modelcontextprotocol/servers?tab=readme-ov-file).
      * host a code-sandbox MCP server for executing tool calls.
    * Daily usage quotas/limits for models.
    * Simple chat interface to try models.

4. 🔄 **(Optional) API Abstraction:**
    * Implementation of a stable API interface (e.g., OpenAI-compatible).
    * Development of backend adapters for different LLM providers.

## 💻 Development Install

To set up the project for development using `uv`, follow these steps:

1. Ensure you have `uv` installed. If not, you can install it via pip: `pip install uv`.
2. Clone the repository:
   ```bash
   git clone https://github.com/tu-wien-datalab/aqueduct.git
   cd aqueduct
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   uv venv
   source .venv/bin/activate
   uv sync
   ```
4. You can now run the Django development server:
   ```bash
   uv run aqueduct/manage.py runserver
   ```

## ⚙️ Architecture

![AI Gateway Architecture](./docs/assets/AI%20Gateway%20Architecture.excalidraw.svg "AI Gateway Architecture")

The gateway server processes requests, interacts with the Django API for token verification, and updates usage logs.
Django frontend provides a UI with role-based access control:

**Role-Based Access Control (RBAC):**

| Role           | Functionality                                                     |
| -------------- |-------------------------------------------------------------------|
| User           | API key generation, team key viewing.                             |
| Team-Admin     | Team API key management, team usage viewing.                      |
| Org-Admin      | Team creation, user management within the organization.           |
| (Super)Admin   | Organization management, global usage limit modification.         |

**Organization and User Management:**

* User assignment to organizations based on SSO group memberships.
* Specific admin group used to assign Admins.
* Org-Admin management by Admin in UI (group).

**Usage Reporting and Metrics:**

* Usage reports based on user roles (team, organization, global).
* `/metrics` endpoint for system monitoring + usage.


## ❓ Open Questions

* Integration of Open Policy Agent (OPA) for authorization and policy enforcement?

## 🔗 Relevant Links

* [Token Validator](https://github.com/TU-Wien-dataLAB/token-validator)