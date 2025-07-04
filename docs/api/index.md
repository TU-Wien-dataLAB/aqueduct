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

Click an endpoint below to learn more:

- [Completions](completions.md)
- [Chat Completions](chat-completions.md)
- [Embeddings](embeddings.md)
- [List Models](models.md)