---
title: List Models
parent: API Reference
nav_order: 4
---

# List Models

The `/models` endpoint retrieves the list of available models for your token.

## HTTP Request

```http
GET /models
Host: your-aqueduct-domain.com
Authorization: Bearer YOUR_AQUEDUCT_TOKEN
```

## Examples

### cURL Example

```bash
curl https://your-aqueduct-domain.com/models \
  -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN"
```

### Python Example (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://your-aqueduct-domain.com",
    api_key="YOUR_AQUEDUCT_TOKEN",
)

models = client.models.list()
print(models.data)

```

## Sample Response

```json
{
  "data": [
    {
      "id": "text-embedding-ada-002",
      "object": "model",
      "created": 1680000000,
      "owned_by": "aqueduct"
    },
    {
      "id": "gpt-4",
      "object": "model",
      "created": 1680000000,
      "owned_by": "aqueduct"
    }
  ],
  "object": "list"
}
```

## Model Aliases

Model aliases are resolved before model validation. If you use an alias in the `model` parameter, it will be resolved to the actual model name before checking if the model is available for your token. See the [Models user guide](../user-guide/models.md) for more information about aliases.
