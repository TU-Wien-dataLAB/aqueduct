#!/usr/bin/env -S uv run --with python-dotenv --with openai --script

import io
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from a .env file one directory up
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

# Retrieve the gateway access token and base URL
endpoint_access_token = os.getenv("AQUEDUCT_GATEWAY_ACCESS_TOKEN")
# Use v1 prefix for batch endpoints
BASE_URL = os.getenv("AQUEDUCT_GATEWAY_BASE_URL", "http://localhost:8000/v1")

# Define the model and completion parameters
MODEL = os.getenv("MODEL_NAME", "Qwen-32B")
COMPLETION_WINDOW = "24h"

# Initialize the OpenAI client
client = OpenAI(base_url=BASE_URL, api_key=endpoint_access_token)

print("Preparing batch input JSONL payload...")
# Example chat-completion entries
records = [
    {
        "custom_id": f"request-{i}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
        },
    }
    for i in range(20)
]
payload = "\n".join(json.dumps(r, separators=(",", ":")) for r in records) + "\n"
stream = io.BytesIO(payload.encode("utf-8"))
stream.name = "batch_input.jsonl"

print('Uploading input payload as purpose "batch"...')
upload_resp = client.files.create(file=stream, purpose="batch")
print(f"Uploaded file ID: {upload_resp.id}")

print("Reading back uploaded file contents via SDK for verification...")
content = client.files.content(upload_resp.id)
# content may be bytes or string depending on SDK; decode if needed
if isinstance(content, (bytes, bytearray)):
    print(content.decode("utf-8"))
else:
    print(content)

print("Creating batch...")
batch_resp = client.batches.create(
    completion_window=COMPLETION_WINDOW,
    endpoint="/v1/chat/completions",
    input_file_id=upload_resp.id,
)

print("Batch created:")
print(batch_resp)
