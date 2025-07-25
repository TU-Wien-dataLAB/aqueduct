---
title: Examples
parent: User Guide
nav_order: 8
---

# Examples
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## OpenAI SDK Chat Completion

Aqueduct supports OpenAI-compatible APIs. You can use the official OpenAI Python SDK to interact with Aqueduct
endpoints. Below is an example of how to make a call using your Aqueduct token:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://your-aqueduct-domain.com/vllm",  # Replace with your Aqueduct endpoint
    api_key="YOUR_AQUEDUCT_TOKEN",  # Replace with your Aqueduct token
)

response = client.chat.completions.create(
    model="your-model-name",  # Replace with your model name
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, who are you?"}
    ]
)

print(response.choices[0].message.content)
```

Replace `YOUR_AQUEDUCT_TOKEN`, `https://your-aqueduct-domain.com/vllm`, and `your-model-name` with your actual token,
Aqueduct endpoint, and model name.

---

## cURL Example: Chat Completion

You can also use `curl` to make a chat completion request:

```bash
curl https://your-aqueduct-domain.com/vllm/chat/completions \
  -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Write me a short poem!"}
    ]
  }'
```

---

## cURL Example: List Models

To list available models, use the following `curl` command:

```bash
curl https://your-aqueduct-domain.com/vllm/models \
  -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN"
```

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
