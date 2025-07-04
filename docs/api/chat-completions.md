---
title: Chat Completions
parent: API Reference
nav_order: 2
---

# Chat Completions

The `/chat/completions` endpoint generates conversational completions based on a sequence of messages.

## HTTP Request

```http
POST /chat/completions
POST /v1/chat/completions
Host: your-aqueduct-domain.com
Authorization: Bearer YOUR_AQUEDUCT_TOKEN
Content-Type: application/json
```

## Request Body

The request body should be a JSON object compatible with the OpenAI [ChatCompletionCreateParams](https://platform.openai.com/docs/api-reference/chat/completions/create) schema.

| Parameter           | Type               | Description                                                      |
| --------------------| ------------------ | ---------------------------------------------------------------- |
| `model`             | string             | The name of the model to use.                                    |
| `messages`          | array              | List of messages with roles (`system`, `user`, `assistant`).      |
| `max_tokens`        | integer            | Maximum number of tokens to generate.                            |
| `temperature`       | number             | Sampling temperature to use.                                     |
| `top_p`             | number             | Nucleus sampling probability.                                    |
| `n`                 | integer            | Number of chat completion choices to generate.                   |
| `stream`            | boolean            | If true, stream back partial progress as events.                 |
| `stop`              | string or [string] | Up to 4 sequences where the API will stop generating.            |
| `presence_penalty`  | number             | Penalize new tokens based on existing presence.                  |
| `frequency_penalty` | number             | Penalize new tokens based on existing frequency.                 |
| `user`              | string             | A unique identifier for the end-user.                            |
| `response_format`   | object             | (Optional) Define a JSON schema for structured responses.        |

See the OpenAI documentation for a full list of parameters.

### JSON Schema Responses

To generate structured JSON output, include a `response_format` field with a JSON schema:

```json
"response_format": {
  "type": "json_schema",
  "json_schema": {
    "name": "schema",
    "schema": {
      "type": "object",
      "properties": {
        "greeting": {"type": "string"},
        "count": {"type": "integer"}
      },
      "required": ["greeting", "count"]
    }
  }
}
```

## Examples

### cURL Example

```bash
curl https://your-aqueduct-domain.com/chat/completions \
  -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
      "model": "your-model-name",
      "messages": [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "Write me a short poem!"}
      ]
  }'
```

### Python Example (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://your-aqueduct-domain.com",
    api_key="YOUR_AQUEDUCT_TOKEN",
)

response = client.chat.completions.create(
    model="your-model-name",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write me a short poem!"}
    ]
)
print(response.choices[0].message.content)
```

### Streaming Example

```bash
curl https://your-aqueduct-domain.com/chat/completions \
  -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
      "model": "your-model-name",
      "messages": [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "Tell me a story."}
      ],
      "stream": true
  }'
```

## Error Responses

Same as [Completions](completions.md) endpoint.