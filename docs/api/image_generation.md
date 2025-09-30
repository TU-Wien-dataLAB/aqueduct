---
title: Image Generation
parent: API Reference
nav_order: 8
---

# Image Generation

The `/images/generations` endpoint generates images from text prompts using text-to-image models.
See the OpenAI documentation for a more comprehensive [guide](https://platform.openai.com/docs/guides/images).

## HTTP Request

```http
POST /images/generations
POST /v1/images/generations
Host: your-aqueduct-domain.com
Authorization: Bearer YOUR_AQUEDUCT_TOKEN
Content-Type: application/json
```

## Request Body

The request body should be a JSON object compatible with the OpenAI [ImageGenerateParams](https://platform.openai.com/docs/api-reference/images/create) schema.

| Parameter         | Type   | Description                                                                   |
|-------------------|--------|-------------------------------------------------------------------------------|
| `model`           | string | (Optional) The model to use for image generation.                             |
| `prompt`          | string | The text description of the desired image(s).                                 |
| `n`               | int    | (Optional) The number of images to generate. Default: `1`.                    |
| `size`            | string | (Optional) The size of the generated images.                                  |
| `quality`         | string | (Optional) The quality of the generated images.                               |
| `style`           | string | (Optional) The style of the generated images.                                 |
| `response_format` | string | (Optional) The format to return the generated images in (`url`or `b64_json`). |
| `user`            | string | (Optional) A unique identifier representing your end-user.                    |

Note that the default values may be different for different models, and some parameters are only supported
by specific models.

See the OpenAI documentation for a full list of parameters.

## Examples

### cURL Example

```bash
curl https://your-aqueduct-domain.com/images/generations \
  -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
      "model": "dall-e-3",
      "prompt": "A cute baby sea otter",
      "n": 1,
      "size": "1024x1024"
  }'
```

### Python Example (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://your-aqueduct-domain.com/v1",
    api_key="YOUR_AQUEDUCT_TOKEN"
)

response = client.images.generate(
    model="dall-e-3",
    prompt="A cute baby sea otter",
    n=1,
    size="1024x1024"
)

print(response.data[0].url)
```

## Error Responses

Same as [Completions](completions.md) endpoint.
