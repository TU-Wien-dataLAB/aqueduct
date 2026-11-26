---
title: Image Edit
parent: API Reference
nav_order: 12
---

# Image Edit

The `/images/edits` endpoint edits an input image based on a text prompt. It is the image-editing
companion to [`/images/generations`](image_generation.md) and accepts a **multipart/form-data**
request with one or more uploaded images.

## HTTP Request

```http
POST /images/edits
POST /v1/images/edits
Host: your-aqueduct-domain.com
Authorization: Bearer YOUR_AQUEDUCT_TOKEN
Content-Type: multipart/form-data
```

## Request Body

The request is sent as `multipart/form-data`, with the file fields containing the image(s) to edit.
The request body is compatible with the OpenAI
[image edit](https://platform.openai.com/docs/api-reference/images/edit) schema, plus the
additional `seed` parameter below.

| Parameter         | Type   | Description                                                                   |
|-------------------|--------|-------------------------------------------------------------------------------|
| `model`           | string | (Required) The model to use for image editing. You can also use a [model alias](../user-guide/models.md). |
| `prompt`          | string | (Required) The text description of the edit to apply.                         |
| `image`           | file   | (Required) The image to edit. To edit multiple images, send each as a separate `image[]` field. |
| `mask`            | file   | (Optional) An additional image whose fully opaque areas indicate where `image` should be edited. |
| `n`               | int    | (Optional) The number of edited images to return. Default: `1`.               |
| `size`            | string | (Optional) The size of the output images.                                     |
| `response_format` | string | (Optional) The format of the returned images. Only `b64_json` is supported.   |
| `seed`            | int    | (Optional) Seed forwarded to providers that support reproducible generation.  |
| `user`            | string | (Optional) A unique identifier representing your end-user.                    |

> **Note**: Aqueduct does not support image streaming. Requests with `stream: true` are rejected
> with a `400 Bad Request`.

> **Note**: Only the `b64_json` response format is supported; requesting `url` returns a
> `400 Bad Request`.

## Examples

### cURL Example

```bash
curl https://your-aqueduct-domain.com/images/edits \
  -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN" \
  -F "model=dall-e-2" \
  -F "prompt=Add a tree to the background" \
  -F "image=@/path/to/image.png"
```

### Python Example (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="https://your-aqueduct-domain.com/v1", api_key="YOUR_AQUEDUCT_TOKEN")

response = client.images.edit(
    model="dall-e-2",
    image=open("/path/to/image.png", "rb"),
    prompt="Add a tree to the background",
    n=1,
)

print(response.data[0].b64_json)
```

## Sample Response

```json
{
  "created": 1710000000,
  "data": [
    {
      "b64_json": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    }
  ]
}
```

## Error Responses

Same as [Completions](completions.md) endpoint.
