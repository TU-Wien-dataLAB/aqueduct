---
title: Examples
parent: User Guide
nav_order: 5
---

# Examples
{: .no_toc }

## Table of contents

{: .no_toc .text-delta }

1. TOC
   {:toc}

---

# OpenAI SDK Chat Completion

Aqueduct supports OpenAI-compatible APIs. You can use the official OpenAI Python SDK to interact with Aqueduct
endpoints. Below is an example of how to make a call using your Aqueduct token:

```python
import openai

openai.api_key = "YOUR_AQUEDUCT_TOKEN"
openai.api_base = "https://your-aqueduct-domain.com/vllm"  # Replace with your Aqueduct endpoint

response = openai.ChatCompletion.create(
    model="your-model-name",  # Replace with your model name
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, who are you?"}
    ]
)

print(response.choices[0].message["content"])
```

Replace `YOUR_AQUEDUCT_TOKEN`, `https://your-aqueduct-domain.com/vllm`, and `your-model-name` with your actual token,
Aqueduct endpoint, and model name.
