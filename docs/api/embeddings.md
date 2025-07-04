---
title: Embeddings
parent: API Reference
nav_order: 3
---

# Embeddings

The `/embeddings` endpoint generates vector embeddings for input text.

## HTTP Request

```http
POST /embeddings
POST /v1/embeddings
Host: your-aqueduct-domain.com
Authorization: Bearer YOUR_AQUEDUCT_TOKEN
Content-Type: application/json
```

## Request Body

The request body should be a JSON object compatible with the OpenAI [EmbeddingCreateParams](https://platform.openai.com/docs/api-reference/embeddings/create) schema.

| Parameter | Type               | Description                                      |
| --------- | ------------------ | ------------------------------------------------ |
| `model`   | string             | The name of the model to use.                    |
| `input`   | string or [string] | The input text or array of texts to embed.       |

See the OpenAI documentation for a full list of parameters.

## Examples

### cURL Example

```bash
curl https://your-aqueduct-domain.com/embeddings \
  -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-embedding-model-name",
    "input": "The quick brown fox jumps over the lazy dog."
  }'
```

### Python Example (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://your-aqueduct-domain.com",
    api_key="YOUR_AQUEDUCT_TOKEN",
)

response = client.embeddings.create(
    model="your-embedding-model-name",
    input="The quick brown fox jumps over the lazy dog.",
)
print(response.data[0].embedding)
```

## Sample Response

```json
{
  "data": [
    {
      "object": "embedding",
      "embedding": [ /* float values */ ],
      "index": 0
    }
  ],
  "model": "text-embedding-ada-002",
  "object": "list"
}
```

## Error Responses

Same as [Completions](completions.md) endpoint.