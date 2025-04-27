#!/usr/bin/env -S uv run --with python-dotenv --with openai --script

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from a .env file located one directory up
load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

# Get the access token from environment variables
endpoint_access_token = os.getenv("AQUEDUCT_GATEWAY_ACCESS_TOKEN")

# Define the base URL for the API endpoint
# BASE_URL = "https://lite-llm.ai.datalab.tuwien.ac.at"
BASE_URL = "http://localhost:8000/vllm" # Using localhost as in the original script

# Define the model name
# MODEL = "qwen-32b"
MODEL = "Qwen-32B" # Using Qwen-32B as in the original script

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
print("--- Streaming Response ---")
response = client.completions.create(
    model=MODEL,
    prompt=PROMPT,
    max_tokens=MAX_TOKENS,
    stream=True # This is already set correctly for streaming
)

# Iterate over the streaming response and print chunks as they arrive
try:
    # The response object is an iterator when stream=True
    for chunk in response:
        # Each chunk contains a 'choices' list
        if chunk.choices:
            # Access the text from the first choice
            # In streaming, text comes in pieces
            text_chunk = chunk.choices[0].text
            if text_chunk:
                # Print the chunk without a newline at the end
                # This allows the text to build up on a single line or flow naturally
                print(text_chunk, end='', flush=True)

    # Print a final newline after the stream ends to ensure the next prompt is on a new line
    print("\n--- End of Stream ---")

except Exception as e:
    print(f"\nAn error occurred: {e}")