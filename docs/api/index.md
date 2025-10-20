---
title: API Reference
nav_order: 5
---

# API Reference

The API Reference provides detailed information on the HTTP endpoints exposed by the Aqueduct AI Gateway.
All endpoints are fully compatible with the OpenAI API, so you can use existing OpenAI clients (such as the OpenAI Python SDK)
or plain HTTP clients (e.g., `curl`) to make requests.

## Available Endpoints

| Endpoint                     | Method      | Description                             |
|------------------------------|-------------|-----------------------------------------|
| `/completions`               | POST        | Text completions                        |
| `/chat/completions`          | POST        | Chat-based completions                  |
| `/embeddings`                | POST        | Text embeddings                         |
| `/models`                    | GET         | List available models                   |
| `/files`                     | GET, POST   | Manage input files for batch processing |
| `/files/{file_id}`           | GET, DELETE | Retrieve or delete a specific file      |
| `/files/{file_id}/content`   | GET         | Download file contents                  |
| `/batches`                   | GET, POST   | List or create batch jobs               |
| `/batches/{batch_id}`        | GET         | Retrieve batch status and metadata      |
| `/batches/{batch_id}/cancel` | POST        | Cancel a batch job                      |
| `/audio/speech`              | POST        | Generate audio from text                |
| `/images/generations`        | POST        | Generate images from text               |
| `/mcp-servers/{name}/mcp`    | GET, POST, DELETE | Model Context Protocol server interaction |

Click an endpoint below to learn more:

- [Completions](completions.md)
- [Chat Completions](chat-completions.md)
- [Embeddings](embeddings.md)
- [List Models](models.md)
- [Files](files.md)
- [Batches](batches.md)
- [Audio Speech](speech.md)
- [Image generation](image_generation.md)
- [Model Context Protocol (MCP)](mcp.md)

---

## Troubleshooting

Here are some common errors and how to resolve them:

- **401 Unauthorized**
  - **Cause:** Invalid or missing API token.
  - **Solution:** Ensure you are using a valid Aqueduct token in the `Authorization` header.

- **404 Not Found**
  - **Cause:** Incorrect endpoint URL or model name.
  - **Solution:** Double-check the endpoint URL and model name. Use the `/models` endpoint to list available models.

- **429 Too Many Requests**
  - **Cause:** Rate limit exceeded.
  - **Solution:** Wait before making more requests, or contact your administrator to increase your rate limit.

- **Other Errors**
  - Check your request payload for correct formatting.
  - Ensure your Aqueduct endpoint is reachable and you have network access.

If you continue to experience issues, consult the Aqueduct documentation or contact support.
