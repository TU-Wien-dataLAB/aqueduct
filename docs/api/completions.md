---
title: Completions
parent: API Reference
nav_order: 1
---

# Completions

The `/completions` endpoint generates text completions based on the provided prompt.

## HTTP Request

```http
POST /completions
POST /v1/completions
Host: your-aqueduct-domain.com
Authorization: Bearer YOUR_AQUEDUCT_TOKEN
Content-Type: application/json
```

## Request Body

The request body should be a JSON object compatible with the OpenAI [CompletionCreateParams](https://platform.openai.com/docs/api-reference/completions/create) schema.

| Parameter           | Type               | Description                                           |
| --------------------| ------------------ | ----------------------------------------------------- |
| `model`             | string             | The name of the model to use.                         |
| `prompt`            | string or [string] | The prompt(s) to generate completions for.            |
| `suffix`            | string             | Optional text to append after the prompt.             |
| `max_completion_tokens` | integer            | Maximum number of tokens to generate.                 |
| `temperature`       | number             | Sampling temperature to use.                          |
| `top_p`             | number             | Nucleus sampling probability.                         |
| `n`                 | integer            | Number of completions to generate for each prompt.     |
| `stream`            | boolean            | If true, send back partial progress as events.        |
| `stop`              | string or [string] | Up to 4 sequences where the API will stop generating. |
| `presence_penalty`  | number             | Penalize new tokens based on existing presence.       |
| `frequency_penalty` | number             | Penalize new tokens based on existing frequency.      |
| `user`              | string             | A unique identifier for the end-user.                 |

See the OpenAI documentation for a full list of parameters.

## Examples

### cURL Example

```bash
curl https://your-aqueduct-domain.com/completions \
  -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name",
    "prompt": "Once upon a time",
    "max_completion_tokens": 50,
    "temperature": 0.7
  }'
```

### Python Example (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://your-aqueduct-domain.com",
    api_key="YOUR_AQUEDUCT_TOKEN",
)

response = client.completions.create(
    model="your-model-name",
    prompt="Once upon a time",
    max_tokens=50,
    temperature=0.7,
)
print(response.choices[0].text)
```

## Streaming Responses

To receive a streamed response, set `"stream": true` in the request body. The Aqueduct Gateway will return a Server-Sent Events (SSE) stream with `data: ...` chunks following the OpenAI streaming format.

```bash
curl https://your-aqueduct-domain.com/completions \
  -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name",
    "prompt": "Once upon a time",
    "max_completion_tokens": 50,
    "stream": true
  }'
```

## Error Responses

Aqueduct maps backend errors to HTTP status codes similar to the OpenAI API:

| Status Code | Description                                           |
| ----------- | ----------------------------------------------------- |
| 200         | OK                                                    |
| 400         | Bad request (invalid parameters)                      |
| 401         | Unauthorized (invalid API token)                      |
| 403         | Forbidden (permission denied)                         |
| 404         | Not found (model or endpoint not available)           |
| 422         | Unprocessable entity                                  |
| 429         | Rate limit exceeded                                   |
| 504         | Gateway timeout                                       |
| 500         | Internal server error (upstream or gateway error)     |
| 503         | Service unavailable                                   |
