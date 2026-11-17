---
title: Rerank
parent: API Reference
nav_order: 10
---

# Rerank

The `/rerank` endpoint reranks a list of documents by their relevance to a query, using a dedicated reranking model. It is intended for Retrieval-Augmented Generation (RAG) pipelines and uses the `/v1/rerank` API that OpenAI-compatible backends (e.g. llama.cpp, vLLM) expose. The endpoint relays requests and responses to the backing model, so both common rerank wire formats are supported: the **Jina style** (`model`, `object`, `usage`, `results`) and the **Cohere style** (`id`, `results`, `meta`).

## HTTP Request

```http
POST /rerank
POST /v1/rerank
Host: your-aqueduct-domain.com
Authorization: Bearer YOUR_AQUEDUCT_TOKEN
Content-Type: application/json
```

## Request Body

| Parameter          | Type            | Description                                                                                                   |
| ------------------ | --------------- | --------------------------------------------------------------------------------------------------------------- |
| `model`            | string          | The name of the reranking model to use. You can also use a [model alias](../user-guide/models.md).            |
| `query`            | string          | The query to rank the documents against.                                                                        |
| `documents`        | [string] or [object] | The documents to rerank. Each entry is a string, or an object with a `text` field.                            |
| `top_n`            | integer         | Optional. Only return the top `n` most relevant documents.                                                      |
| `return_documents` | boolean         | Optional. Whether to include the document in each result. Jina-style backends default to `true`, Cohere-style backends to `false`. |
| `rank_fields`      | [string]        | Optional, Cohere style. Keys of the document objects to consider when ranking, in priority order.              |

## Examples

### cURL Example

```bash
curl https://your-aqueduct-domain.com/rerank \
  -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name",
    "query": "What is Aqueduct?",
    "documents": [
      "Aqueduct is an AI gateway developed by TU Wien.",
      "Vienna is the capital of Austria.",
      "Aqueduct supports multiple AI model providers."
    ],
    "top_n": 2,
    "return_documents": true
  }'
```

### Python Example

```python
import httpx

response = httpx.post(
    "https://your-aqueduct-domain.com/v1/rerank",
    headers={"Authorization": "Bearer YOUR_AQUEDUCT_TOKEN"},
    json={
        "model": "your-model-name",
        "query": "What is Aqueduct?",
        "documents": [
            "Aqueduct is an AI gateway developed by TU Wien.",
            "Vienna is the capital of Austria.",
            "Aqueduct supports multiple AI model providers.",
        ],
        "top_n": 2,
        "return_documents": True,
    },
)
for result in response.json()["results"]:
    print(result["index"], result["relevance_score"], result.get("document"))
```

## Sample Response

The response is passed through unchanged from the backing model, so the exact shape depends on the backend.

### Jina-style

```json
{
  "model": "your-model-name",
  "object": "list",
  "usage": {
    "prompt_tokens": 42,
    "total_tokens": 42
  },
  "results": [
    {
      "index": 0,
      "relevance_score": 0.97,
      "document": {"text": "Aqueduct is an AI gateway developed by TU Wien."}
    },
    {
      "index": 2,
      "relevance_score": 0.84,
      "document": {"text": "Aqueduct supports multiple AI model providers."}
    }
  ]
}
```

### Cohere-style

```json
{
  "id": "abc123",
  "results": [
    {
      "index": 0,
      "relevance_score": 0.97,
      "document": {"text": "Aqueduct is an AI gateway developed by TU Wien."}
    },
    {
      "index": 2,
      "relevance_score": 0.84,
      "document": {"text": "Aqueduct supports multiple AI model providers."}
    }
  ],
  "meta": {
    "api_version": {"version": "1"},
    "billed_units": {"total": 1},
    "tokens": {"input_tokens": 42, "output_tokens": 0}
  }
}
```

In both formats:

- `index` refers to the position of the document in the original `documents` array.
- `relevance_score` is a relevance value (typically `0.0`–`1.0`), higher is more relevant.
- The `document` field is included when `return_documents` is `true`.
- Results are returned sorted by `relevance_score`, highest first.

## Provider Notes

The request is relayed directly to the deployment's OpenAI-compatible endpoint (``{api_base}/rerank``), so the backing model is configured in the [router config](../user-guide/models.md) like any other model. For an OpenAI-compatible (e.g. llama.cpp / vLLM) rerank endpoint, configure the deployment's ``api_base`` to end in ``/v1`` and set a model ``id`` so the gateway can resolve the deployment:

```yaml
model_list:
  - model_name: your-model-name
    litellm_params:
      model: openai/your-model-name
      api_base: https://your-inference-host/v1
      api_key: "os.environ/LOCAL_API_KEY"
    model_info:
      id: your-model-name
      mode: rerank
      aliases: ["reranker"]
```

## Error Responses

Same as [Completions](completions.md) endpoint.
