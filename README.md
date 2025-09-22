<div align="center">
  <img src="./docs/assets/Aqueduct Icon.png" width="50%" alt="Aqueduct AI Gateway Logo" />
</div>

# Aqueduct AI Gateway

[![GitHub License](https://img.shields.io/github/license/tu-wien-datalab/aqueduct)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/tu-wien-datalab/aqueduct)](https://github.com/tu-wien-datalab/aqueduct/commits/main)
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![Static Badge](https://img.shields.io/badge/Documentation-GitHub%20Pages-brightgreen)](https://tu-wien-datalab.github.io/aqueduct/)


> [!NOTE]
> This project is still in active development! Contributions welcome!

**Aqueduct AI Gateway** aims to provide a **simple yet fully-featured** AI gateway you can self-host with:

- no [SSO tax](https://konghq.com/pricing)
- no [observability tax](https://www.litellm.ai/enterprise)
- no [self-hosting tax](https://portkey.ai/pricing)
- no [org management tax](https://www.litellm.ai/enterprise)
- etc.

We aim to achieve this by:

- Building on top of the **LiteLLM Router SDK** to provide enhanced control and routing capabilities, while avoiding re-implementing the entire OpenAI-compatible APIs or every new feature, and
- Using **Django** for a clean, efficient, and maintainable implementation.

For more information, please read the [Documentation](https://tu-wien-datalab.github.io/aqueduct/).

If you don’t need user self-service, take a look at [Envoy AI Gateway](https://aigateway.envoyproxy.io) instead!

![AI Gateway Architecture](./docs/assets/screenshot.png "AI Gateway Architecture")

## 🚀 Getting Started

The recommended way to get Aqueduct running locally is with Docker Compose. This will start the Django app, a PostgreSQL
database, Celery with Redis as the broker, Tika for text extraction from files, and a local mock OIDC provider (Dex)
for authentication.

1. **Clone the repository**
   ```bash
   git clone https://github.com/tu-wien-datalab/aqueduct.git
   cd aqueduct
   ```

2. **Set the necessary environment variables**
Most of the necessary environment variables are provided in the `.example.env` file.
You only need to set the `OPENAI_API_KEY`:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```
   This variable is used by the sample router configuration, provided in the
`example_router_config.yaml` file. Adjust it if you want to use other models.

3. **Start the services**
   ```bash
   docker compose up --build
   ```
   This will build and start all required services using the provided `.example.env` and
`example_router_config.yaml` files for environment variables and the router configuration.

4. **Access the application**

    - The web UI will be available at [http://localhost:8000](http://localhost:8000)
    - The local OIDC provider (Dex) will be running at [http://localhost:5556/dex](http://localhost:5556/dex)
    - Default login credentials for Dex are:
        - **Username:** `you@example.com`
        - **Password:** `1234`

You can now access the admin UI and start exploring the gateway features.

> **NOTE:**
> This starts Django in debug mode and is not suitable for production deployments. Change the
> [necessary settings](https://docs.djangoproject.com/en/5.2/topics/settings/#the-basics) for a production deployment.


For other installation methods, check out the [Getting Started Guide](https://tu-wien-datalab.github.io/aqueduct/getting-started/).

---


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
    * ~~Support for streaming requests.~~
    * ~~Add usage checks as pre-processing steps to limit requests.~~
    * ~~A functional `docker-compose.yml`~~
      * ~~Add mock OIDC server to compose for fully local development.~~
    * Thorough unit/integration testing of `management` and `gateway`.
    * ~~Add documentation.~~

3. 🔄 **Advanced Features:**
   * ~~Limit org/team/user access to specific models.~~
   * Database schema optimization for write-heavy request/usage logging.
     * ~~Data retention~~
     * Partitioning?
   * Add `locust` load testing.
   * A `/metrics` endpoint for monitoring.
   * ~~Dashboard to track usage of orgs, teams and users.~~
   * Support users belonging to multiple `Orgs`.
   * Management of MCP tool calling server endpoints e.g. from
     the [MCP Server list](https://github.com/modelcontextprotocol/servers?tab=readme-ov-file) (see [#10](https://github.com/TU-Wien-dataLAB/aqueduct/issues/10)).
   * Daily usage quotas/limits for models.
   * Add responses API (see [#12](https://github.com/TU-Wien-dataLAB/aqueduct/issues/12))
   * Add Batch API for throughput-heavy workloads, e.g., synthetic data generation (see [#11](https://github.com/TU-Wien-dataLAB/aqueduct/issues/11)).
   * Implement [Guardrails](https://github.com/guardrails-ai/guardrails) on completions.


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
