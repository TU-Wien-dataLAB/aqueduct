---
title: Audio Speech
parent: API Reference
nav_order: 7
---

# Audio Speech

The `/audio/speech` endpoint generates audio from text using text-to-speech models.
See the OpenAI documentation for a more comprehensive [guide](https://platform.openai.com/docs/guides/text-to-speech).

## HTTP Request

```http
POST /audio/speech
POST /v1/audio/speech
Host: your-aqueduct-domain.com
Authorization: Bearer YOUR_AQUEDUCT_TOKEN
Content-Type: application/json
```

## Request Body

The request body should be a JSON object compatible with the OpenAI [SpeechCreateParams](https://platform.openai.com/docs/api-reference/audio/createSpeech) schema.

| Parameter         | Type   | Description                                                   |
|-------------------| ------ |---------------------------------------------------------------|
| `model`           | string | The name of the text-to-speech model to use.                  |
| `voice`           | string | The voice to use for generation.                              |
| `input`           | string | The text to generate audio for.                               |
| `instructions`    | string | (Optional) Instructions for how to generate the audio.        |
| `response_format` | string | (Optional) The format to return the audio in. Default: `mp3`. |
| `speed`           | number | (Optional) The speed of the generated audio. Default: `1.0`.  |

See the OpenAI documentation for a full list of parameters.

## Examples

### cURL Example

```bash
curl https://your-aqueduct-domain.com/audio/speech \
  -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
      "model": "kokoro",
      "voice": "af_alloy",
      "input": "Today is a wonderful day to build something people really like!",
      "instructions": "Speak in a cheerful and positive tone."
  }' --output speech.mp3
```

### Python Example (OpenAI SDK)

```python
from pathlib import Path
from openai import OpenAI

client = OpenAI(
    base_url="https://your-aqueduct-domain.com/v1",
    api_key="YOUR_AQUEDUCT_TOKEN"
)

speech_file_path = Path("/tmp") / "speech.mp3"

with client.audio.speech.with_streaming_response.create(
    model="kokoro",
    voice="af_alloy",
    input="Today is a wonderful day to build something people really like!",
    instructions="Speak in a cheerful and positive tone.",
) as response:
    response.stream_to_file(speech_file_path)
```

## Error Responses

Same as [Completions](completions.md) endpoint.
