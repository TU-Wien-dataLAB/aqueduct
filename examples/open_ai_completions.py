#!/usr/bin/env -S uv run --with python-dotenv --with openai --script

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from a .env file located one directory up
load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

# Get the access token from environment variables
endpoint_access_token = os.getenv("AQUEDUCT_GATEWAY_ACCESS_TOKEN")
endpoint_access_token = "sk-w7MoIVQQ46yPPvpA3m9mMqr5HsFI1_pwXgkR1v663SY"

# Define the base URL for the API endpoint
# BASE_URL = "https://lite-llm.ai.datalab.tuwien.ac.at"
BASE_URL = "http://localhost:8000/vllm"  # Using localhost as in the original script
BASE_URL = "https://aqueduct.ai.datalab.tuwien.ac.at/vllm"

# Define the model name
# MODEL = "qwen-32b"
MODEL = "Qwen-32B"  # Using Qwen-32B as in the original script

# Define the prompt for the AI
PROMPT = "Write me a short poem!"

# Define the maximum number of tokens for the response
MAX_TOKENS = 150

# Initialize the OpenAI client
# Use the base_url and api_key for the custom endpoint
client = OpenAI(
    base_url=BASE_URL,
    api_key=endpoint_access_token,
)

# Send the request to the API with streaming enabled
print(f"Sending request to {BASE_URL} with model {MODEL}...")
print("--- Streaming Chat Response ---")

response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": PROMPT}],
    max_tokens=MAX_TOKENS,
    stream=True  # Streaming is enabled
)

try:
    for chunk in response:
        if chunk.choices:
            delta = chunk.choices[0].delta
            content_chunk = delta.content  # <-- access as attribute, not .get()

            if content_chunk:
                print(content_chunk, end='', flush=True)

    print("\n--- End of Stream ---")

except Exception as e:
    print(f"\nAn error occurred: {e}")
