---
title: Files
parent: API Reference
nav_order: 5
---

# Files

The `/files` endpoints allow you to upload and manage files for batch processing.
Uploaded files must be newline-delimited JSON (`.jsonl`) and are used with the [Batches](batches.md) API.

## HTTP Requests

```http
POST /files
POST /v1/files
Host: your-aqueduct-domain.com
Authorization: Bearer YOUR_AQUEDUCT_TOKEN
Content-Type: multipart/form-data
```

```http
GET /files
GET /v1/files
Host: your-aqueduct-domain.com
Authorization: Bearer YOUR_AQUEDUCT_TOKEN
```

```http
GET /files/{file_id}
GET /v1/files/{file_id}
Host: your-aqueduct-domain.com
Authorization: Bearer YOUR_AQUEDUCT_TOKEN
```

```http
DELETE /files/{file_id}
DELETE /v1/files/{file_id}
Host: your-aqueduct-domain.com
Authorization: Bearer YOUR_AQUEDUCT_TOKEN
```

```http
GET /files/{file_id}/content
GET /v1/files/{file_id}/content
Host: your-aqueduct-domain.com
Authorization: Bearer YOUR_AQUEDUCT_TOKEN
```

## Upload a File

The request body must be sent as `multipart/form-data` with the following fields:

| Parameter  | Type | Description                                               |
| ---------- | ---- | --------------------------------------------------------- |
| `file`     | file | A newline-delimited JSON (`.jsonl`) file to upload.       |
| `purpose`  | string | Must be set to `batch`.                                 |

Files are given an expiry date, after which they are deleted. By default, this expiry is 7 days.

See the OpenAI documentation for file uploads: [File Upload API](https://platform.openai.com/docs/api-reference/files/upload).

## List Files

Retrieve a list of uploaded files for your user.

## Retrieve File Metadata

Get metadata for a specific file by its ID.

## Download File Content

Download the raw contents of a specific file by its ID.

## Delete a File

Delete a specific file by its ID.

## Examples

### cURL Example

```bash
# Upload a file
curl https://your-aqueduct-domain.com/files \
  -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN" \
  -F "purpose=batch" \
  -F "file=@data.jsonl"
```

```bash
# List files
curl https://your-aqueduct-domain.com/files \
  -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN"
```

```bash
# Download file content
curl https://your-aqueduct-domain.com/files/{file_id}/content \
  -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN"
```

```bash
# Delete file
curl https://your-aqueduct-domain.com/files/{file_id} \
  -X DELETE \
  -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN"
```

### Python Example (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://your-aqueduct-domain.com",
    api_key="YOUR_AQUEDUCT_TOKEN",
)

# Upload file
with open("data.jsonl", "rb") as f:
    response = client.files.create(file=f, purpose="batch")
print(response.id)

# List files
files = client.files.list()
for file in files.data:
    print(file.id, file.filename)

# Download content
content = client.files.content(response.id)
print(content)

# Delete file
client.files.delete(response.id)
```

## Error Responses

Same as [Completions](completions.md) endpoint, with the following file-specific cases:

| Status Code | Description                                                                 |
| ----------- | --------------------------------------------------------------------------- |
| 400         | Bad request (missing `file` or `purpose`, unsupported `purpose`, invalid file extension, file too large) |
| 404         | File not found (for metadata, download, or delete)                         |
