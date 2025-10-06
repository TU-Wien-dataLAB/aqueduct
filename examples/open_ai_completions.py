#!/usr/bin/env -S uv run --with python-dotenv --with openai --script

import os

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from a .env file located one directory up
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

# Get the access token from environment variables
endpoint_access_token = os.getenv("AQUEDUCT_GATEWAY_ACCESS_TOKEN")

# Define the base URL for the API endpoint
# BASE_URL = "https://lite-llm.ai.datalab.tuwien.ac.at"
BASE_URL = "http://localhost:8000/v1"  # Using localhost as in the original script

# Define the model name
# MODEL = "qwen-32b"
MODEL = "Qwen-32B"  # Using Qwen-32B as in the original script

# Define the prompt for the AI
PROMPT = "Write me a short poem!"

# Define the maximum number of tokens for the response
MAX_TOKENS = 150

# Initialize the OpenAI client
# Use the base_url and api_key for the custom endpoint
client = OpenAI(base_url=BASE_URL, api_key=endpoint_access_token)

# File input example using 2024ltr.pdf
print("--- File Input Example ---")
try:
    # Read the PDF file and encode it as base64
    pdf_path = os.path.join(os.path.dirname(__file__), "2024ltr.pdf")
    with open(pdf_path, "rb") as pdf_file:
        import base64

        pdf_base64 = base64.b64encode(pdf_file.read()).decode("utf-8")

    # Create a chat completion request with file content
    file_response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze the letter and provide a summary of the key points.",
                    },
                    {
                        "type": "file",
                        "file": {
                            "filename": "2024ltr.pdf",
                            "file_data": f"data:application/pdf;base64,{pdf_base64}",
                            # "file_data": pdf_base64
                        },
                    },
                ],
            }
        ],
        max_completion_tokens=500,
        stream=False,
    )

    if file_response.choices:
        file_content = file_response.choices[0].message.content
        print(f"File analysis result: {file_content}")
    print("--- End of File Input Example ---")

except Exception as e:
    print(f"An error occurred in file input example: {e}")
    print("--- End of File Input Example ---")

# Send the request to the API with streaming enabled
print(f"Sending request to {BASE_URL} with model {MODEL}...")
# --- Embedding Example ---
print("--- Embedding Example ---")
embedding_response = client.embeddings.create(model="openai-embedding", input=PROMPT)
if embedding_response.data:
    embedding_vector = embedding_response.data[0].embedding
    print(f"Embedding vector (first 10 values): {embedding_vector[:10]}")
print("--- End of Embedding Example ---")

print("--- Streaming Chat Response ---")

response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": PROMPT}],
    max_completion_tokens=MAX_TOKENS,
    stream=True,  # Streaming is enabled
)

try:
    for chunk in response:
        if chunk.choices:
            delta = chunk.choices[0].delta
            content_chunk = delta.content  # <-- access as attribute, not .get()

            if content_chunk:
                print(content_chunk, end="", flush=True)

    print("\n--- End of Stream ---")

except Exception as e:
    print(f"\nAn error occurred: {e}")

print("\n--- Non-Streaming Chat Response ---")
try:
    non_stream_response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        max_completion_tokens=MAX_TOKENS,
        stream=False,  # Streaming is disabled
    )
    if non_stream_response.choices:
        content = non_stream_response.choices[0].message.content
        print(content)
    print("--- End of Non-Streaming Response ---")
except Exception as e:
    print(f"An error occurred in non-streaming request: {e}")

# Add a completion (non-chat) response afterwards
print("\n--- Non-Chat Completion Response ---")
try:
    completion_response = client.completions.create(
        model=MODEL, prompt=PROMPT, max_completion_tokens=MAX_TOKENS, stream=False
    )
    if completion_response.choices:
        completion_content = completion_response.choices[0].text
        print(completion_content)
    print("--- End of Non-Chat Completion Response ---")
except Exception as e:
    print(f"An error occurred in non-chat completion request: {e}")

print("\n--- Streaming Non-Chat Completion Response ---")
try:
    stream_response = client.completions.create(
        model=MODEL, prompt=PROMPT, max_completion_tokens=MAX_TOKENS, stream=True
    )
    for chunk in stream_response:
        if chunk.choices:
            text_chunk = chunk.choices[0].text
            if text_chunk:
                print(text_chunk, end="", flush=True)
    print("\n--- End of Streaming Non-Chat Completion Response ---")
except Exception as e:
    print(f"An error occurred in streaming non-chat completion request: {e}")
