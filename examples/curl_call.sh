#!/bin/bash

# This script sources a ../.env file and calls an OpenAI-compatible completions endpoint.

# Source the environment file
if [ -f "../.env" ]; then
  source "../.env"
else
  echo "Error: ../.env file not found."
  exit 1
fi

# Check if ENDPOINT_ACCESS_TOKEN is set
if [ -z "$AQUEDUCT_GATEWAY_ACCESS_TOKEN" ]; then
  echo "Error: AQUEDUCT_GATEWAY_ACCESS_TOKEN is not set in ../.env"
  exit 1
fi

# --- Configuration ---
# Set the URL of your OpenAI-compatible server's completions endpoint
# Replace with your server's address and the correct path if different
ENDPOINT_URL="http://localhost:8000/v1/completions"

# Set the model to use
MODEL="qwen-32b" # Replace with the model name your server supports

# Set the prompt for the completion
PROMPT="Write me a short poem!"

# Set the maximum number of tokens to generate
MAX_TOKENS=250

# --- API Call ---
# Use curl to make the POST request to the completions endpoint
curl -X POST "$ENDPOINT_URL" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AQUEDUCT_GATEWAY_ACCESS_TOKEN" \
  -d '{
    "model": "'"$MODEL"'",
    "prompt": "'"$PROMPT"'",
    "max_tokens": '"$MAX_TOKENS"'
  }'

# Add a newline at the end of the output
echo