#!/usr/bin/env -S uv run --with python-dotenv --with openai --script

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from a .env file one directory up
load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

# Retrieve the gateway access token and base URL
endpoint_access_token = os.getenv('AQUEDUCT_GATEWAY_ACCESS_TOKEN')
# Use v1 prefix for batch endpoints
BASE_URL = os.getenv('AQUEDUCT_GATEWAY_BASE_URL', 'http://localhost:8000/v1')

# Define the model and completion parameters
MODEL = os.getenv('MODEL_NAME', 'Qwen-32B')
COMPLETION_WINDOW = '24h'

# Initialize the OpenAI client
client = OpenAI(
    base_url=BASE_URL,
    api_key=endpoint_access_token,
)

batches = client.batches.list()
for batch in batches:
    if batch.status == 'completed':
        print(f'\t{batch.id}')
        output_content = client.files.content(batch.output_file_id)
        print(f'\t{batch.output_file_id}')
        print(f'\t{output_content}')
        if isinstance(output_content, (bytes, bytearray)):
            print(f'\t{output_content.decode('utf-8')}')
        else:
            print(f'\t{output_content.content}')
        break
