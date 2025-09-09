---
title: Audio Transcriptions
parent: API Reference
nav_order: 8
---

# Audio Transcriptions

The `/audio/transcriptions` endpoint transcribes audio files to text using speech-to-text models.
See the OpenAI documentation for a more comprehensive [guide](https://platform.openai.com/docs/guides/speech-to-text).


## HTTP Request

```http
POST /audio/transcriptions
POST /v1/audio/transcriptions
Host: your-aqueduct-domain.com
Authorization: Bearer YOUR_AQUEDUCT_TOKEN
Content-Type: multipart/form-data
```

## Request Body

The request should be sent as `multipart/form-data` and is compatible with the OpenAI [TranscriptionCreateParams](https://platform.openai.com/docs/api-reference/audio/createTranscription) schema.

| Parameter             | Type   | Description                                                      |
| --------------------- | ------ | ---------------------------------------------------------------- |
| `file`                | file   | The audio file to transcribe.                                    |
| `model`               | string | The name of the speech-to-text model to use.                     |
| `language`            | string | (Optional) The language of the audio file.                       |
| `prompt`              | string | (Optional) Text to guide the model's style.                      |
| `response_format`     | string | (Optional) The format of the output. Default: `json`.            |
| `temperature`         | number | (Optional) The sampling temperature. Default: `0`.               |
| `timestamp_granularities[]` | array | (Optional) The timestamp granularities to include.               |

See the OpenAI documentation for a full list of parameters.

## Examples

### cURL Example

```bash
curl https://your-aqueduct-domain.com/audio/transcriptions \
  -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN" \
  -F file="@audio.mp3" \
  -F model="whisper-1" \
  -F language="en" \
  -F response_format="verbose_json"
```

### Python Example (OpenAI SDK)

```python
from pathlib import Path
from openai import OpenAI

client = OpenAI(
    base_url="https://your-aqueduct-domain.com/v1",
    api_key="YOUR_AQUEDUCT_TOKEN"
)

audio_file = Path("/path/to/audio.mp3")

transcription = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file,
    language="en",
    response_format="verbose_json"
)

print(transcription.text)
```

## Error Responses

Same as [Completions](completions.md) endpoint.
