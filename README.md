<div align="center">
  <img src="./docs/assets/Aqueduct Icon.png" width="50%" alt="Aqueduct AI Gateway Logo" />
</div>

# Aqueduct AI Gateway

[![GitHub License](https://img.shields.io/github/license/tu-wien-datalab/aqueduct)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/tu-wien-datalab/aqueduct)](https://github.com/tu-wien-datalab/aqueduct/commits/main)
[![Project Status: Concept – Minimal or no implementation has been done yet, or the repository is only intended to be a limited example, demo, or proof-of-concept.](https://www.repostatus.org/badges/latest/concept.svg)](https://www.repostatus.org/#concept)


**Aqueduct AI Gateway** is an open-source project designed to provide a centralized access point for Large Language Models (LLMs), with features for authentication, usage tracking, and rate limiting.

*This project is still in active development!*

## 🛠️ Technical Overview

This project aims to combine [Directus](https://github.com/directus/directus) for user management/API with minimal additional implementation to create a comprehensive AI gateway. The implementation follows a phased approach:

1.  **Data Model and Administrative Interface:**
    * Implementation of "Teams" and "Organizations" within the data model.
    * Development of an administrative UI using Directus for user and team management.
    * Focus on authentication and basic management functionalities.

2.  **Gateway Relay and Usage Tracking:**
    * Development of a gateway server to relay requests to LLM providers.
    * Implementation of request parsing for usage tracking.
    * Database schema optimization for write-heavy usage logging.
    * Implementation of request buffering for usage logging.

3.  **Multi-Provider Support and Advanced Features:**
    * Extension of the data model to include "Models" representing LLM endpoints.
    * Implementation of a `/models` endpoint to list available models.
    * Request routing to specific provider endpoints based on model selection.
    * Implementation of granular access control based on models and usage limits.
    * Development of a `/metrics` endpoint for monitoring.
    * Implementation of model cooldown and retry logic.
    * Implementation of MCP tool calling endpoints.

4.  **(Optional) API Abstraction:**
    * Implementation of a stable API interface (e.g., OpenAI-compatible).
    * Development of backend adapters for different LLM providers.

## ⚙️ Architecture

![AI Gateway Architecture](./docs/assets/AI%20Gateway%20Architecture.excalidraw.svg "AI Gateway Architecture")

The gateway server processes requests, interacts with the Directus API for token management, and updates usage logs. Directus provides a UI with role-based access control:

**Role-Based Access Control (RBAC):**

| Role           | Functionality                                                               | Page Access                                                   |
| -------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------- |
| User           | API key generation, team key viewing.                                       | Token Page                                                    |
| Team-Admin     | Team API key management, team usage viewing.                                  | Token + Usage Page                                            |
| Org-Admin      | Team creation, user management within the organization.                               | Token + Usage + Team Management Page                        |
| (Super)Admin   | Organization management, global usage limit modification. | Token + Usage + Team Management + Admin Page |

**Organization and User Management:**

* User assignment to organizations based on SSO group memberships.
* Specific admin group used to assign Admins.
* Org-Admin management by Admin in UI (boolean flag).
* Default team with configurable usage limits.
* Automatic role assignment via Directus flows upon user login.

**Usage Reporting and Metrics:**

* Usage reports based on user roles (team, organization, global).
* Visualized through Directus dashboards?
* `/metrics` endpoint for system monitoring.


## 🚀 Implementation Roadmap

1.  Docker Compose setup for PostgreSQL and Directus.
2.  Database migration tool integration (Alembic, Prisma Migrate, ...).
3.  Database schema definition.
4.  Setup container for database migrations.
5.  Directus configuration and template export (`directus-template-cli`).
6.  Automated Directus template application in separate container or modification of database migration container.
7.  API gateway server development using FastAPI.

## ❓ Open Questions

* Integration of Open Policy Agent (OPA) for authorization and policy enforcement?

## 🔗 Relevant Links

* [Directus](https://directus.io)
* [directus-template-cli](https://github.com/directus-labs/directus-template-cli)
* [Token Validator](https://github.com/TU-Wien-dataLAB/token-validator)