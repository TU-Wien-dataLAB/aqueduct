---
title: Batches
parent: API Reference
nav_order: 6
---

# Batches

The `/batches` endpoints allow you to submit and manage batch processing jobs using uploaded JSONL files from
the [Files](files.md) API.

## HTTP Requests

```http
POST /batches
POST /v1/batches
Host: your-aqueduct-domain.com
Authorization: Bearer YOUR_AQUEDUCT_TOKEN
Content-Type: application/json
```

```http
GET /batches
GET /v1/batches
Host: your-aqueduct-domain.com
Authorization: Bearer YOUR_AQUEDUCT_TOKEN
```

```http
GET /batches/{batch_id}
GET /v1/batches/{batch_id}
Host: your-aqueduct-domain.com
Authorization: Bearer YOUR_AQUEDUCT_TOKEN
```

```http
POST /batches/{batch_id}/cancel
POST /v1/batches/{batch_id}/cancel
Host: your-aqueduct-domain.com
Authorization: Bearer YOUR_AQUEDUCT_TOKEN
Content-Type: application/json
```

## Create a Batch
> **Note**: The `completion_window` parameter is currently ignored by the server and has no effect, but is still required.
The request body should be a JSON object compatible with the
OpenAI [BatchCreateParams](https://platform.openai.com/docs/api-reference/batch/create) schema.

| Parameter           | Type   | Description                                                                                                   |
|---------------------|--------|---------------------------------------------------------------------------------------------------------------|
| `input_file_id`     | string | The ID of a file uploaded via the Files API.                                                                  |
| `completion_window` | string | The time frame within which the batch should be processed (e.g., `"24h"`).                                    |
| `endpoint`          | string | The OpenAI API endpoint for each request (e.g., `/v1/completions`, `/v1/chat/completions`, `/v1/embeddings`). |
| `metadata`          | object | Optional metadata to attach to the batch.                                                                     |

See the OpenAI documentation for full
details: [Batch Create API](https://platform.openai.com/docs/api-reference/batch/create).

## List Batches

Retrieve a list of batch jobs for your token.

## Retrieve a Batch

Get the current status and metadata of a specific batch.

## Cancel a Batch

Request cancellation of an in-progress or validating batch.

## Examples

### cURL Example

```bash
# Create a batch
curl https://your-aqueduct-domain.com/batches \
  -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "input_file_id": "file-id",
    "completion_window": "24h",
    "endpoint": "/v1/chat/completions"
  }'
```

```bash
# Get batch status
curl https://your-aqueduct-domain.com/batches/{batch_id} \
  -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN"
```

```bash
# Cancel a batch
curl -X POST https://your-aqueduct-domain.com/batches/{batch_id}/cancel \
  -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN"
```

### Python Example (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://your-aqueduct-domain.com",
    api_key="YOUR_AQUEDUCT_TOKEN",
)

# Create batch
batch = client.batches.create(
    input_file_id="file-id",
    completion_window="24h",
    endpoint="/v1/chat/completions",
)
print(batch.id, batch.status)

# Retrieve batch
batch = client.batches.retrieve(batch.id)
print(batch.status, batch.request_counts)

#  Retrieve batch results
file_response = client.files.content(batch.output_file_id)
print(file_response.text)

# Cancel batch
batch = client.batches.cancel(batch.id)
print(batch.status)
```

## Error Responses

Same as [Completions](completions.md) endpoint, with the following batch-specific cases:

| Status Code | Description                                                                                 |
|-------------|---------------------------------------------------------------------------------------------|
| 400         | Bad request (invalid JSON, missing required fields, input file must have purpose `"batch"`) |
| 404         | Input file not found or batch not found                                                     |
