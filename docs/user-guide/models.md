---
title: Models
parent: User Guide
nav_order: 3
---

# Models

The Models page displays the available endpoints and their associated models.

![Models Page](../assets/user_guide/models_page.png)

Each endpoint has a slug, which is included in the Aqueduct URL path to distinguish the endpoint in API calls. The
remaining path is forwarded to the actual endpoint (Internal URL). For example, a call to `/vllm/v1/models` will forward
`v1/models` to the Internal URL of the `vllm` endpoint. If a model is referenced in the call, it must be part of the
endpoint.

Admins can add and edit endpoints and models in the Admin Panel.
