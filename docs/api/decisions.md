---
title: Decisions
parent: API Reference
nav_order: 11
---

# Decisions

The `/decisions` endpoint evaluates structured **classification and scoring questions** against a
shared piece of evidence (text, images, or both). It is a thin proxy to OpenAI's Decisions API:
Aqueduct forwards the `input` and `questions` to the configured decision model and returns the
scored `answers` unchanged.

Unlike the other proxied endpoints, this endpoint is **not** part of the standard OpenAI
chat/embedding surface, but it *is* served through the OpenAI Python SDK's
[`client.decisions.create`](https://platform.openai.com/docs/api-reference/decisions/create) method
and the `/v1/decisions` path, so existing OpenAI clients can call it directly.

## HTTP Request

```http
POST /decisions
POST /v1/decisions
Host: your-aqueduct-domain.com
Authorization: Bearer YOUR_AQUEDUCT_TOKEN
Content-Type: application/json
```

## Request Body

The request body should be a JSON object compatible with the OpenAI
[`DecisionCreateParams`](https://platform.openai.com/docs/api-reference/decisions/create) schema.

| Parameter           | Type                       | Description                                                                                   |
|---------------------|----------------------------|-----------------------------------------------------------------------------------------------|
| `model`             | string                     | (Required) The decision model to use. You can also use a [model alias](../user-guide/models.md). |
| `input`             | string or [object]         | (Required) The shared evidence. A string, or an array of user messages containing text and inline images. |
| `questions`         | [object]                   | (Required) The questions to evaluate against the input.                                       |
| `safety_identifier` | string                     | (Optional) An opaque caller-provided end-user identifier, scoped by the verified org.          |

Each entry in `questions` is one of three question types, discriminated by its `type` field:

### `predicate`

A yes/no question answered with a probability.

| Parameter      | Type   | Description                  |
|----------------|--------|------------------------------|
| `type`         | string | Always `predicate`.          |
| `instructions` | string | (Required) The question.     |
| `name`         | string | (Optional) A name for the answer. |

### `choice`

A multiple-choice question answered with the chosen value and its confidence.

| Parameter      | Type   | Description                  |
|----------------|--------|------------------------------|
| `type`         | string | Always `choice`.             |
| `instructions` | string | (Required) The question.     |
| `choices`      | [object] | (Required) The allowed options. |
| `name`         | string | (Optional) A name for the answer. |

Each entry in `choices` has a `value` (string or boolean, required) and an optional `description`.

### `score`

A rating question answered on a defined numeric scale.

| Parameter      | Type   | Description                  |
|----------------|--------|------------------------------|
| `type`         | string | Always `score`.              |
| `instructions` | string | (Required) The question.     |
| `levels`       | [object] | (Required) The score levels. |
| `name`         | string | (Optional) A name for the answer. |

Each entry in `levels` has a `label` (string, required) and an optional `description`.

## Examples

### cURL Example

```bash
curl https://your-aqueduct-domain.com/decisions \
  -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
      "model": "LiquidAI/d1-3B",
      "input": "Inspect the product in this photo.",
      "questions": [
          {
              "type": "predicate",
              "name": "visible_damage",
              "instructions": "Does the product have visible damage?"
          }
      ]
  }'
```

### Python Example (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="https://your-aqueduct-domain.com/v1", api_key="YOUR_AQUEDUCT_TOKEN")

decision = client.decisions.create(
    model="LiquidAI/d1-3B",
    input="Inspect the product in this photo.",
    questions=[
        {
            "type": "predicate",
            "name": "visible_damage",
            "instructions": "Does the product have visible damage?",
        }
    ],
)

print(decision.answers[0].name, decision.answers[0].probability)
```

## Sample Response

The response is passed through unchanged from the backing model. It contains an `answers` array
(one entry per question), the resolved `model`, and token `usage`:

```json
{
  "answers": [
    {
      "name": "visible_damage",
      "probability": 0.92,
      "type": "predicate"
    }
  ],
  "model": "LiquidAI/d1-3B",
  "usage": {
    "input_tokens": 15,
    "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
    "output_tokens": 9,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 24
  }
}
```

Depending on the question type, an answer takes one of these shapes:

- `predicate`: `name`, `probability`, `type`
- `choice`: `name`, `choice`, `confidence`, `probabilities`, `type`
- `score`: `name`, `score`, `confidence`, `probabilities`, `type`
- `refusal`: `name`, `type` (the model declined the question)

## Provider Notes

The request is relayed directly to the deployment's OpenAI-compatible endpoint, so the backing
decision model is configured in the [router config](../user-guide/models.md) like any other model.
A typical decision model is configured with an `id` and an alias, e.g. `decision`:

```yaml
model_list:
  - model_name: LiquidAI/d1-3B
    litellm_params:
      model: openai/LiquidAI/d1-3B
      api_key: "os.environ/OPENAI_API_KEY"
    model_info:
      id: LiquidAI/d1-3B
      aliases: ["decision", "d1-3b"]
```

## Error Responses

Same as [Completions](completions.md) endpoint.
