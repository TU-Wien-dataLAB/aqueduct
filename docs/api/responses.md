---
title: Responses
parent: API Reference
nav_order: 8
---

# Responses

The `/responses` endpoint provides access to OpenAI's Responses API for creating and managing structured AI responses with advanced tool integration capabilities.

## HTTP Requests

### Create Response
```http
POST /responses
POST /v1/responses
Host: your-aqueduct-domain.com
Authorization: Bearer YOUR_AQUEDUCT_TOKEN
Content-Type: application/json
```

### Get/Delete Response
```http
GET /v1/responses/{response_id}
DELETE /v1/responses/{response_id}
Host: your-aqueduct-domain.com
Authorization: Bearer YOUR_AQUEDUCT_TOKEN
```

### Get Response Input Items
```http
GET /v1/responses/{response_id}/input_items
Host: your-aqueduct-domain.com
Authorization: Bearer YOUR_AQUEDUCT_TOKEN
```

## Create Response Request Body

The request body should be a JSON object compatible with the OpenAI Responses API schema.

| Parameter               | Type               | Description                                                  |
|-------------------------| ------------------ | ------------------------------------------------------------ |
| `model`                 | string             | The name of the model to use. You can also use a [model alias](../user-guide/models.md). |
| `input`                 | array              | List of input messages with roles and content.               |
| `max_output_tokens`     | integer            | Maximum number of tokens to generate in the response.       |
| `tools`                 | array              | List of tools available for the response to use.            |
| `tool_choice`           | object             | Controls which tools the response can use.                   |
| `temperature`           | number             | Sampling temperature to use.                                 |
| `top_p`                 | number             | Nucleus sampling probability.                                |
| `stream`                | boolean            | If true, stream back partial progress as events.             |
| `user`                  | string             | A unique identifier for the end-user.                        |

## Access Controls and Tool Availability

The Responses API includes access controls for tool usage:

### Native Tools
- Only specific native tools are permitted for security and stability reasons
- Not all native tools may be available in your deployment
- Attempts to use unauthorized native tools will return a 400 error

### MCP (Model Context Protocol) Tools
- MCP tools are supported through server-based integration
- Each MCP tool must reference a configured server via `server_label`
- The token must have access to the specified MCP server
- Tokens can exclude specific MCP servers through their configuration
- Missing or unauthorized MCP servers result in 404 errors

For detailed information about MCP server configuration and usage, see the [Model Context Protocol (MCP)](mcp.md) documentation.

### Tool Types Supported
| Tool Type | Description | Access Control |
|-----------|-------------|----------------|
| `function` | Custom function tools | No additional restrictions |
| `custom` | Custom-defined tools | No additional restrictions |
| `mcp` | Model Context Protocol tools | Server access validation required |

## Examples

### cURL Example - Basic Response

```bash
curl https://your-aqueduct-domain.com/v1/responses \
  -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
      "model": "your-model-name",
      "input": [
          {"role": "user", "content": "Hello, how are you?"}
      ],
      "max_output_tokens": 50
  }'
```

### Python Example (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://your-aqueduct-domain.com/v1",
    api_key="YOUR_AQUEDUCT_TOKEN",
)

response = client.responses.create(
    model="your-model-name",
    input=[{"role": "user", "content": "Hello, how are you?"}],
    max_output_tokens=50
)

print(response.output)
```

### Example with Function Tools

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://your-aqueduct-domain.com/v1",
    api_key="YOUR_AQUEDUCT_TOKEN",
)

tools = [
    {
        "type": "function",
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
          "type": "object",
          "properties": {
              "location": {
                  "type": "string",
                  "description": "The city and state, e.g. San Francisco, CA",
              },
              "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
          },
          "required": ["location", "unit"],
        }
    }
]

response = client.responses.create(
  model="your-model-name",
  tools=tools,
  input="What is the weather like in Boston today?",
  tool_choice="auto"
)

print(response)
```

### Example with MCP Tools

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://your-aqueduct-domain.com/v1",
    api_key="YOUR_AQUEDUCT_TOKEN",
)

# Use an MCP tool from a configured server
response = client.responses.create(
    model="your-model-name",
    input=[{"role": "user", "content": "Search for recent AI news"}],
    tools=[
        {
            "type": "mcp",
            "name": "search_web",
            "server_label": "search",  # Must be configured and accessible to token
            "description": "Search the web for information"
        }
    ]
)
```

### Streaming Example

```bash
curl https://your-aqueduct-domain.com/v1/responses \
  -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
      "model": "your-model-name",
      "input": [
          {"role": "user", "content": "Tell me a story."}
      ],
      "stream": true
  }'
```

### Retrieve Response

```bash
curl https://your-aqueduct-domain.com/v1/responses/response_abc123 \
  -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN"
```

### Delete Response

```bash
curl -X DELETE https://your-aqueduct-domain.com/v1/responses/response_abc123 \
  -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN"
```

### Get Response Input Items

```bash
curl https://your-aqueduct-domain.com/v1/responses/response_abc123/input_items \
  -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN"
```

## Error Responses

| Status Code | Error | Description |
|-------------|-------|-------------|
| 400 | Bad Request | Invalid parameters or unauthorized tool type |
| 401 | Unauthorized | Invalid or missing authentication token |
| 404 | Not Found | Response ID or MCP server not found |
| 429 | Too Many Requests | Rate limits exceeded |
| 500 | Internal Server Error | Server-side processing error |

## Response Caching

Responses are cached after creation and can be retrieved using the response ID. The cache is automatically invalidated when a response is deleted.

## Streaming

When `stream: true` is specified, responses are sent using Server-Sent Events (SSE) format, allowing for real-time processing of partial responses.