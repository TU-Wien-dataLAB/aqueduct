#!/bin/bash

# This script sources a ../.env file and lists existing batch jobs
# using an OpenAI-compatible API's /v1/batches endpoint.

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
# Set the base URL of your OpenAI-compatible server
# Assumes it supports /v1/batches endpoint
BASE_URL="http://localhost:8000"

# --- List Batch Jobs ---
echo "Listing models..."
curl -s -X GET "$BASE_URL/models" \
  -H "Authorization: Bearer $AQUEDUCT_GATEWAY_ACCESS_TOKEN"

# Add a newline at the end of the output for better formatting
echo

# Note: The output will be JSON. You might want to pipe it to jq for better readability:
# ./curl_call.sh | jq
