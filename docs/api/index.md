---
title: API Reference
nav_order: 5
---

# API Reference

The API Reference provides detailed information on the HTTP endpoints exposed by the Aqueduct AI Gateway.
All endpoints are fully compatible with the OpenAI API, so you can use existing OpenAI clients (such as the OpenAI Python SDK)
or plain HTTP clients (e.g., `curl`) to make requests.

## Available Endpoints

| Endpoint               | Method | Description                   |
| ---------------------- | ------ | ----------------------------- |
| `/completions`         | POST   | Text completions              |
| `/chat/completions`    | POST   | Chat-based completions        |
| `/embeddings`          | POST   | Text embeddings               |
| `/models`              | GET    | List available models         |
| `/files`               | GET, POST | Manage input files for batch processing |
| `/files/{file_id}`     | GET, DELETE | Retrieve or delete a specific file |
| `/files/{file_id}/content` | GET | Download file contents |
| `/batches`             | GET, POST | List or create batch jobs |
| `/batches/{batch_id}`  | GET    | Retrieve batch status and metadata |
| `/batches/{batch_id}/cancel` | POST | Cancel a batch job |
| `/audio/speech`        | POST   | Generate audio from text |

Click an endpoint below to learn more:

- [Completions](completions.md)
- [Chat Completions](chat-completions.md)
- [Embeddings](embeddings.md)
- [List Models](models.md)
- [Files](files.md)
- [Batches](batches.md)
- [Audio Speech](speech.md)
