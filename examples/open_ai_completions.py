#!/usr/bin/env -S uv run --with python-dotenv --with openai --script

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))
endpoint_access_token = os.getenv("AQUEDUCT_GATEWAY_ACCESS_TOKEN")

# BASE_URL = "https://lite-llm.ai.datalab.tuwien.ac.at"
BASE_URL = "http://localhost:8000/v1"
MODEL = "qwen-32b"
PROMPT = "Write me a short poem!"
MAX_TOKENS = 150

client = OpenAI(
    base_url=BASE_URL,
    api_key=endpoint_access_token,
)

response = client.completions.create(
    model=MODEL,
    prompt=PROMPT,
    max_tokens=MAX_TOKENS,
)

if response.choices:
    print(response.choices[0].text.strip())