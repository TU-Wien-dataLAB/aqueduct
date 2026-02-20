---
title: Vector Stores
parent: API Reference
nav_order: 6
---

# Vector Stores

The vector stores API allows you to create and manage vector stores for semantic search and retrieval. Vector stores contain files that are chunked, embedded, and indexed for efficient similarity search.

All vector store endpoints are fully compatible with the [OpenAI Vector Stores API](https://platform.openai.com/docs/api-reference/vector-stores), which powers semantic search for the Retrieval API and the `file_search` tool in the Responses and Assistants APIs.

## HTTP Requests

### Vector Stores

```http
POST /vector_stores
POST /v1/vector_stores
Host: your-aqueduct-domain.com
Authorization: Bearer YOUR_AQUEDUCT_TOKEN
```

```http
GET /vector_stores
GET /v1/vector_stores
Host: your-aqueduct-domain.com
Authorization: Bearer YOUR_AQUEDUCT_TOKEN
```

```http
GET /vector_stores/{vector_store_id}
GET /v1/vector_stores/{vector_store_id}
Host: your-aqueduct-domain.com
Authorization: Bearer YOUR_AQUEDUCT_TOKEN
```

```http
POST /vector_stores/{vector_store_id}
POST /v1/vector_stores/{vector_store_id}
Host: your-aqueduct-domain.com
Authorization: Bearer YOUR_AQUEDUCT_TOKEN
```

```http
DELETE /vector_stores/{vector_store_id}
DELETE /v1/vector_stores/{vector_store_id}
Host: your-aqueduct-domain.com
Authorization: Bearer YOUR_AQUEDUCT_TOKEN
```

```http
POST /vector_stores/{vector_store_id}/search
POST /v1/vector_stores/{vector_store_id}/search
Host: your-aqueduct-domain.com
Authorization: Bearer YOUR_AQUEDUCT_TOKEN
```

### Vector Store Files

```http
POST /vector_stores/{vector_store_id}/files
POST /v1/vector_stores/{vector_store_id}/files
Host: your-aqueduct-domain.com
Authorization: Bearer YOUR_AQUEDUCT_TOKEN
```

```http
GET /vector_stores/{vector_store_id}/files
GET /v1/vector_stores/{vector_store_id}/files
Host: your-aqueduct-domain.com
Authorization: Bearer YOUR_AQUEDUCT_TOKEN
```

```http
GET /vector_stores/{vector_store_id}/files/{file_id}
GET /v1/vector_stores/{vector_store_id}/files/{file_id}
Host: your-aqueduct-domain.com
Authorization: Bearer YOUR_AQUEDUCT_TOKEN
```

```http
POST /vector_stores/{vector_store_id}/files/{file_id}
POST /v1/vector_stores/{vector_store_id}/files/{file_id}
Host: your-aqueduct-domain.com
Authorization: Bearer YOUR_AQUEDUCT_TOKEN
```

```http
DELETE /vector_stores/{vector_store_id}/files/{file_id}
DELETE /v1/vector_stores/{vector_store_id}/files/{file_id}
Host: your-aqueduct-domain.com
Authorization: Bearer YOUR_AQUEDUCT_TOKEN
```

```http
GET /vector_stores/{vector_store_id}/files/{file_id}/content
GET /v1/vector_stores/{vector_store_id}/files/{file_id}/content
Host: your-aqueduct-domain.com
Authorization: Bearer YOUR_AQUEDUCT_TOKEN
```

### Vector Store File Batches

```http
POST /vector_stores/{vector_store_id}/file_batches
POST /v1/vector_stores/{vector_store_id}/file_batches
Host: your-aqueduct-domain.com
Authorization: Bearer YOUR_AQUEDUCT_TOKEN
```

```http
GET /vector_stores/{vector_store_id}/file_batches/{batch_id}
GET /v1/vector_stores/{vector_store_id}/file_batches/{batch_id}
Host: your-aqueduct-domain.com
Authorization: Bearer YOUR_AQUEDUCT_TOKEN
```

```http
POST /vector_stores/{vector_store_id}/file_batches/{batch_id}/cancel
POST /v1/vector_stores/{vector_store_id}/file_batches/{batch_id}/cancel
Host: your-aqueduct-domain.com
Authorization: Bearer YOUR_AQUEDUCT_TOKEN
```

```http
GET /vector_stores/{vector_store_id}/file_batches/{batch_id}/files
GET /v1/vector_stores/{vector_store_id}/file_batches/{batch_id}/files
Host: your-aqueduct-domain.com
Authorization: Bearer YOUR_AQUEDUCT_TOKEN
```

## Vector Store Parameters

Create vector store:

| Parameter           | Type    | Description                                                                                              |
| ------------------- | ------- | -------------------------------------------------------------------------------------------------------- |
| `name`              | string  | The name of the vector store (optional).                                                                 |
| `description`       | string  | A description for the vector store. Can be used to describe the vector store's purpose (optional).       |
| `file_ids`          | array   | A list of File IDs that the vector store should use (optional).                                         |
| `expires_after`     | object  | The expiration policy for a vector store (optional).                                                     |
| `chunking_strategy` | object  | The chunking strategy used to chunk the file(s). If not set, will use the `auto` strategy (optional).    |
| `metadata`          | map     | A map of up to 16 key-value pairs (optional).                                                            |

Modify vector store:

| Parameter       | Type    | Description                                                              |
| --------------- | ------- | ------------------------------------------------------------------------ |
| `name`          | string  | A new name for the vector store (optional).                              |
| `description`   | string  | A description for the vector store (optional).                           |
| `expires_after` | object  | The expiration policy for a vector store. Can be `null` to clear (optional). |
| `metadata`      | map     | A map of up to 16 key-value pairs (optional).                            |

Search vector store:

| Parameter       | Type    | Description                                                              |
| --------------- | ------- | ------------------------------------------------------------------------ |
| `query`         | string or array | A query string for a search (required).                            |
| `filters`       | object  | A filter to apply based on file attributes (optional).                   |
| `max_num_results` | integer | The maximum number of results to return, between 1 and 50 (optional, default: 10). |
| `ranking_options` | object | Ranking options for search (optional).                                  |
| `rewrite_query` | boolean | Whether to rewrite the natural language query for vector search (optional, default: false). |

## Vector Store File Parameters

Create file:

| Parameter          | Type    | Description                                                                 |
| ------------------ | ------- | --------------------------------------------------------------------------- |
| `file_id`          | string  | The File ID to attach to the vector store. If chunking_strategy is not set, will use the `auto` strategy (required). |
| `chunking_strategy` | object | The chunking strategy used to chunk the file(s) (optional).                 |
| `attributes`       | map     | A map of up to 16 key-value pairs (optional).                              |

Update file attributes:

| Parameter    | Type | Description                                                              |
| ------------ | ---- | ------------------------------------------------------------------------ |
| `attributes` | map  | A map of up to 16 key-value pairs (required).                            |

## Vector Store File Batch Parameters

Create batch:

| Parameter          | Type           | Description                                                                     |
| ------------------ | -------------- | ------------------------------------------------------------------------------- |
| `file_ids`         | array          | A list of File IDs. If `attributes` or `chunking_strategy` are provided, they will be applied to all files in the batch (optional, mutually exclusive with `files`). |
| `files`            | array          | A list of objects that each include a `file_id` plus optional `attributes` or `chunking_strategy`. Use this when you need to override metadata for specific files (optional, mutually exclusive with `file_ids`). |
| `chunking_strategy` | object         | The chunking strategy to use for all files in the batch (optional).         |
| `attributes`       | map            | A map of up to 16 key-value pairs to attach to all files in the batch (optional). |

Note: Either `file_ids` or `files` must be provided, but not both.

## Examples

### cURL Examples

Create a vector store:

```bash
curl https://your-aqueduct-domain.com/vector_stores \
  -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Support FAQ"
  }'
```

List vector stores:

```bash
curl https://your-aqueduct-domain.com/vector_stores \
  -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN"
```

Retrieve a vector store:

```bash
curl https://your-aqueduct-domain.com/vector_stores/{vector_store_id} \
  -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN"
```

Modify a vector store:

```bash
curl https://your-aqueduct-domain.com/vector_stores/{vector_store_id} \
  -X POST \
  -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Updated Documentation Name",
    "description": "Updated product documentation"
  }'
```

Delete a vector store:

```bash
curl https://your-aqueduct-domain.com/vector_stores/{vector_store_id} \
  -X DELETE \
  -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN"
```

Search a vector store:

```bash
curl -X POST https://your-aqueduct-domain.com/vector_stores/{vector_store_id}/search \
  -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I reset my password?",
    "filters": {"category": "support"}
  }'
```

Add a file to a vector store:

```bash
curl https://your-aqueduct-domain.com/vector_stores/{vector_store_id}/files \
  -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "file_abc123"
  }'
```

List files in a vector store:

```bash
curl https://your-aqueduct-domain.com/vector_stores/{vector_store_id}/files \
  -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN"
```

Retrieve a file in a vector store:

```bash
curl https://your-aqueduct-domain.com/vector_stores/{vector_store_id}/files/{file_id} \
  -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN"
```

Delete a file from a vector store:

```bash
curl https://your-aqueduct-domain.com/vector_stores/{vector_store_id}/files/{file_id} \
  -X DELETE \
  -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN"
```

Create a file batch with per-file attributes:

```bash
curl https://your-aqueduct-domain.com/vector_stores/{vector_store_id}/file_batches \
  -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "files": [
      {
        "file_id": "file_abc123",
        "attributes": {"category": "getting-started"}
      },
      {
        "file_id": "file_def456",
        "chunking_strategy": {
          "type": "static",
          "static": {
            "max_chunk_size_tokens": 1200,
            "chunk_overlap_tokens": 200
          }
        }
      }
    ]
  }'
```

Cancel a file batch:

```bash
curl https://your-aqueduct-domain.com/vector_stores/{vector_store_id}/file_batches/{batch_id}/cancel \
  -X POST \
  -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN"
```

List files in a batch:

```bash
curl https://your-aqueduct-domain.com/vector_stores/{vector_store_id}/file_batches/{batch_id}/files \
  -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN"
```

### Python Example (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://your-aqueduct-domain.com",
    api_key="YOUR_AQUEDUCT_TOKEN",
)

# Create a vector store
vector_store = client.vector_stores.create(name="Support FAQ")
print(f"Created vector store: {vector_store.id}")

# Create with files attached
vector_store_with_files = client.vector_stores.create(
    name="Documentation",
    file_ids=["file_abc123", "file_def456"]
)
print(f"Created vector store with files: {vector_store_with_files.id}")

# List vector stores
vector_stores = client.vector_stores.list(limit=20)
for vs in vector_stores.data:
    print(f"{vs.id}: {vs.name}")

# Retrieve a vector store
vs = client.vector_stores.retrieve(vector_store.id)
print(f"Vector store status: {vs.status}")
print(f"File counts: {vs.file_counts}")

# Modify a vector store
vs = client.vector_stores.update(
    vector_store.id,
    name="Updated Product Documentation",
    description="Comprehensive product documentation",
    metadata={"category": "support"}
)
print(f"Updated name: {vs.name}")

# Search a vector store
results = client.vector_stores.search(
    vector_store.id,
    query="How do I reset my password?",
    max_num_results=10,
    rewrite_query=True
)
print(f"Search query: {results.search_query}")
for result in results.data:
    print(f"  File: {result.filename} (score: {result.score})")

# Add a file to a vector store
vs_file = client.vector_stores.files.create(
    vector_store.id,
    file_id="file_abc123"
)
print(f"Added file with status: {vs_file.status}")

# List files in a vector store
files = client.vector_stores.files.list(
    vector_store.id,
    limit=20,
    filter="completed"
)
for file in files.data:
    print(f"{file.id}: status={file.status}")

# Retrieve a file in a vector store
vs_file = client.vector_stores.files.retrieve(
    vector_store.id,
    vs_file.id
)
print(f"File status: {vs_file.status}, usage_bytes: {vs_file.usage_bytes}")

# Update file attributes
vs_file = client.vector_stores.files.update(
    vector_store.id,
    vs_file.id,
    attributes={"category": "finance", "priority": "high"}
)
print(f"Updated file attributes: {vs_file.attributes}")

# Create a file batch with per-file attributes
batch = client.vector_stores.file_batches.create(
    vector_store.id,
    files=[
        {
            "file_id": "file_abc123",
            "attributes": {"category": "getting-started"}
        },
        {
            "file_id": "file_def456",
            "chunking_strategy": {
                "type": "static",
                "static": {
                    "max_chunk_size_tokens": 1200,
                    "chunk_overlap_tokens": 200
                }
            }
        }
    ]
)
print(f"Batch status: {batch.status}")
print(f"File counts: {batch.file_counts}")

# Retrieve batch status
batch = client.vector_stores.file_batches.retrieve(
    vector_store.id,
    batch.id
)
print(f"Batch file_counts: {batch.file_counts}")

# Cancel a batch
batch = client.vector_stores.file_batches.cancel(
    vector_store.id,
    batch.id
)
print(f"Batch cancelled: {batch.status}")

# List files in a batch
batch_files = client.vector_stores.file_batches.list_files(
    vector_store.id,
    batch.id
)
for file in batch_files.data:
    print(f"Batch file: {file.id}")

# Delete a file from vector store
client.vector_stores.files.delete(
    vector_store.id,
    vs_file.id
)
print("File deleted from vector store")

# Delete a vector store
client.vector_stores.delete(vector_store.id)
print("Vector store deleted")
```

## Response Objects

### Vector Store Object

```json
{
  "id": "vs_123",
  "object": "vector_store",
  "name": "my_vector_store",
  "description": null,
  "created_at": 1698107661,
  "usage_bytes": 123456,
  "file_counts": {
    "in_progress": 0,
    "completed": 100,
    "cancelled": 0,
    "failed": 0,
    "total": 100
  },
  "last_active_at": 1698107661,
  "expires_after": null,
  "expires_at": null,
  "last_used_at": 1698107661,
  "status": "completed",
  "metadata": {}
}
```

### Vector Store File Object

```json
{
  "id": "file-abc123",
  "object": "vector_store.file",
  "usage_bytes": 1234,
  "created_at": 1698107661,
  "vector_store_id": "vs_abc123",
  "status": "completed",
  "last_error": null,
  "chunking_strategy": {
    "type": "static",
    "static": {
      "max_chunk_size_tokens": 800,
      "chunk_overlap_tokens": 400
    }
  },
  "attributes": {}
}
```

### Search Results Object

```json
{
  "object": "vector_store.search_results.page",
  "search_query": "What is the return policy?",
  "data": [
    {
      "file_id": "file_123",
      "filename": "document.pdf",
      "score": 0.95,
      "attributes": {
        "author": "John Doe",
        "date": "2023-01-01"
      },
      "content": [
        {
          "type": "text",
          "text": "Relevant chunk"
        }
      ]
    }
  ],
  "has_more": false,
  "next_page": null
}
```

## Rate Limits

Vector store creation is limited to prevent resource exhaustion:

- **Users**: Maximum 10 vector stores per user (configurable via `MAX_USER_VECTOR_STORES`)
- **Teams**: Maximum 50 vector stores per team (configurable via `MAX_TEAM_VECTOR_STORES`)
- **Vector Store Files**: Maximum 1000 files per vector store (configurable via `MAX_VECTOR_STORE_FILES`)

When the limit is reached, the API returns a `403 Forbidden` error.

## Error Responses

| Status Code | Description                                                                 |
| ----------- | --------------------------------------------------------------------------- |
| 400         | Bad request (missing required parameters, invalid parameters).              |
| 403         | Vector store limit reached or file limit reached.                           |
| 404         | Resource not found (vector store, file, or batch).                          |
| 502         | Failed to communicate with upstream vector store service.                   |
| 503         | Vector Store API not configured.                                            |

## See Also

- [OpenAI Vector Stores API Reference](https://platform.openai.com/docs/api-reference/vector-stores)
- [OpenAI Vector Store Files API Reference](https://platform.openai.com/docs/api-reference/vector-stores-files)
- [OpenAI Vector Store File Batches API Reference](https://platform.openai.com/docs/api-reference/vector-stores-file-batches)
- [File Search](https://platform.openai.com/docs/assistants/tools/file-search) - Related guide
- [Files](files.md) - Files API for uploading documents