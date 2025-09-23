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
| `max_completion_tokens` | integer            | Maximum number of tokens to generate.                            |
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

### Multimodal Input Example

```bash
curl https://your-aqueduct-domain.com/chat/completions \
  -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
      "model": "your-model-name",
      "messages": [
          {
              "role": "user",
              "content": [
                  {"type": "text", "text": "What’s in this image?"},
                  {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"}}
              ]
          }
      ],
      "max_completion_tokens": 50,
      "temperature": 0.0
  }'
```

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_AQUEDUCT_TOKEN",
    base_url="https://your-aqueduct-domain.com/v1",
)

image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

response = client.chat.completions.create(
    model="your-model-name",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s in this image?"},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ],
    max_completion_tokens=50,
    temperature=0.0,
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

### File Input

For more information about file inputs, see the [OpenAI documentation on PDF files](https://platform.openai.com/docs/guides/pdf-files?api-mode=chat&lang=python).

You can include files in your chat completions using either file IDs (for previously uploaded files) or base64-encoded data.

#### File Size Limits

When uploading files via base64 encoding, the following size limits apply:

- **Individual file limit**: Each file must be smaller than 10MB
- **Total request limit**: The combined size of all files in a single request must not exceed 32MB

If these limits are exceeded, the API will return a 400 Bad Request error with details about the size violation.

#### Using File IDs

First upload a file using the files API:

```bash
curl https://your-aqueduct-domain.com/files \
    -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN" \
    -F purpose="user_data" \
    -F file="@sample.pdf"
```

Then reference the file in your chat completion:

```bash
curl "https://your-aqueduct-domain.com/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN" \
    -d '{
        "model": "your-model-name",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {
                            "file_id": "file-6F2ksmvXxt4VdoqmHRw6kL"
                        }
                    },
                    {
                        "type": "text",
                        "text": "What is the main topic of this document?"
                    }
                ]
            }
        ]
    }'
```

#### Using Base64-Encoded Files

```python
import base64
from openai import OpenAI

client = OpenAI(
    base_url="https://your-aqueduct-domain.com",
    api_key="YOUR_AQUEDUCT_TOKEN",
)

with open("sample.pdf", "rb") as f:
    data = f.read()

base64_string = base64.b64encode(data).decode("utf-8")

completion = client.chat.completions.create(
    model="your-model-name",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "file",
                    "file": {
                        "filename": "sample.pdf",
                        "file_data": f"data:application/pdf;base64,{base64_string}",
                    }
                },
                {
                    "type": "text",
                    "text": "What is the main topic of this document?",
                }
            ],
        },
    ],
)

print(completion.choices[0].message.content)
```

## Error Responses

Same as [Completions](completions.md) endpoint.
