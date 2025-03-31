<div align="center" min-width="100%">
    <img src="./docs/assets/Aqueduct Icon.png" width="33%" />
</div>


# Aqueduct AI Gateway

![GitHub License](https://img.shields.io/github/license/tu-wien-datalab/aqueduct)
![GitHub last commit](https://img.shields.io/github/last-commit/tu-wien-datalab/aqueduct)


The goal of this project is to replace paid solutions for AI gateways by implementing a simple but effective free open-source solution. 

## 💫 Overview 

We require an AI API gateway that unifies access to LLMs and handles token management and authentication. It also tracks usage and can deny requests based on the number of requests, token usage or API cost of an external provider.
A solution is already implemented called Token Validator that handles authentication of tokens over an API: https://github.com/TU-Wien-dataLAB/token-validator


## 🏆 Proposed Solution 
The plan is to extend the existing simplistic Token Validator in 3 steps that extend the functionality while being useable at any stage. To keep the implementation simple, we will not implement the whole API definition ourselves but instead will rely mostly on API-pass-through and only parse requests from the necessary endpoints (e.g. to track token usage). The user-management API + UI will be implemented using Directus.
1. Extend the data model to include "Teams" and "Orgs". At this stage, the gateway can still only be used for authentication, however, there already needs to be an Admin UI that can be used to manage this.
2. Extend the gateway server to act as a "relay" between LLM providers and the users that passes through requests but extracts useful information. As a "step zero" the gateway could just pass on any request it receives. The gateway server then needs to parse specific requests and track usage. Usage is limited by teams and its assigned users. The database table that logs usage should be implemented to support write heavy workload. The requests from the gateway should also be buffered (e.g. write usage logs only every 2s). At this point, the AI gateway server is fully functional for a single LLM provider.
    1. To enable multiple providers, the data model needs to be updated to include models, which describe the endpoint to connect to, and the /models endpoint is not passthrough anymore but returns the model list from the database. The main calls to the providers are still pass-through but are now routed to the specific endpoint.
    2. Tokens of Teams/Users can only gain access to specific models with specific usage limits.
    3. Implement a /metrics endpoint for monitoring.
    4. Implement model cooldown and retrying logic.
    5. Implement MCP tool calling option
4. (Optional) The gateway can be implemented to "terminate" the connections and provide a stable OpenAI-based API interface itself, while the backends can be configured through adapters. The pros of the pass-through approach are that API changes in the providers are available without any modifications, however, the cons are that results might be different for different models/model providers.

## 💎 Details

![Architecgure](./docs/assets/AI%20Gateway%20Architecture.excalidraw.svg "AI Gateway Architecture")

The nodes in the AI Gateway server represent transformations of the request. The gateway queries the Direcuts API for token information and updates the usage of the token after the request. The UI is provided by Directus.

UI Users can have the following access rights:

| Role | Description | Page Access |
|------|------------|------------|
| User | Can only re-generate their own keys and view team keys. | Token Page |
| Team-Admin | Can re-generate their own and their team key. | Token + Usage Page |
| Org-Admin | Can create teams within the Org and assign users to them. | Token + Usage + Team Management Page |
| (Super)Admin | Can edit Orgs and has access to every team. Can change limits for teams. | Token + Usage + Team Management + Admin Page |


Admins assign Org-Admins which handle user management. Upon login, users are assigned to an Org based on the SSO group. Within each Org there is a default team that has a configurable default usage limit (e.g. 3 RPM, 20000 TPM). Users are assigned to that team by default.
### Org/Role Assignment
The role/org assignment happens in a Directus flow upon login. The org name gets parsed based on the SSO groups, and the user gets assigned to the correct org. If the user is part of a specific "admin" group, they get admin access. Org admins are assigned by an admin in the UI (boolean flag on user).
### Usage Report
Admins (Team-Admins/Org-Admins/Admins) can see usage report on the usage site. Team-Admins see only their current team usage, Org-Admins see usage reports of all teams in the Org and Admins see all usage reports (per team/Org/overall). Usage reports are queried from the analytics database.
Could be implemented as a Directus dashboard.

## ❓Open Questions
- Can we utilize Open Policy Agent? From my current understanding, it would replace "Check Auth Header" and "Check token/rate limit" in the API Gateway.


## 🚀 Implementation Roadmap
1. Create docker-compose that deploys Postgresql and Directus
2. Find suitable database migration tool (Alembic, Prisma Migrate, ...) to manage database versions
3. Write data schema in ORM used by migration tool
4. Add setup container that runs migrations on database
5. Configure everything on directus side and export as template using https://github.com/directus-labs/directus-template-cli
6. Add setup container that applies Directus template or modify migration container
7. Write simple pass-through API gateway server in FastAPI

## 🔗 Relevant Links 
- https://directus.io
- https://github.com/directus-labs/directus-template-cli
- https://github.com/TU-Wien-dataLAB/token-validator

