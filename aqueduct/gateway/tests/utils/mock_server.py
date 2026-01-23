import argparse
import re
import subprocess
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional
from unittest.mock import patch

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.requests import Request
from starlette.status import HTTP_404_NOT_FOUND

from gateway.tests.utils.helpers import get_available_port

app = FastAPI(debug=True)


class MockConfig(BaseModel):
    status_code: int = 200
    response_data: Dict[str, Any] = {}
    headers: Dict[str, str] = {"Content-Type": "application/json"}


class MockStreamingConfig(MockConfig):
    response_data: list[bytes] = []  # ??
    headers: Dict[str, str] = {"Content-Type": "text/event-stream"}


default_mock_post_responses = {
    # OK
    "audio/speech": MockStreamingConfig(response_data=[b"mock", b"audio", b"data"]),
    # OK
    "audio/transcriptions": MockConfig(
        response_data={
            "text": "This is a mock transcription",
            "usage": {"type": "duration", "seconds": 60},
        }
    ),
    # Batches are already mocked with a mock_router; TODO: check the response data
    "batches": MockConfig(
        response_data={
            "id": "batch_123456789",
            "object": "batch",
            "endpoint": "/v1/chat/completions",
            "errors": None,
            "input_file_id": "file-123456789",
            "completion_window": "24h",
            "status": "validating",
            "created_at": 1694268190,
            "in_progress_at": None,
            "expires_at": 1694354590,
            "finalizing_at": None,
            "completed_at": None,
            "failed_at": None,
            "cancelling_at": None,
            "cancelled_at": None,
            "error_file_id": None,
            "results_file_id": None,
            "metadata": {"custom_id": "my-batch"},
        }
    ),
    # TODO: check this!
    "completions": MockConfig(
        response_data={
            "id": "cmpl-123456789",
            "object": "text_completion",
            "created": 1694268190,
            "model": "text-davinci-003",
            "choices": [
                {
                    "text": "This is a mock completion response.",
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17},
        }
    ),
    # OK
    "chat/completions": MockConfig(
        response_data={
            "id": "chatcmpl-123456789",
            "object": "chat.completion",
            "created": 1694268190,
            "model": "gpt-3.5-turbo",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "This is a mock chat completion response.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 9,
                "total_tokens": 24,
                "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
                "completion_tokens_details": {
                    "reasoning_tokens": 0,
                    "audio_tokens": 0,
                    "accepted_prediction_tokens": 0,
                    "rejected_prediction_tokens": 0,
                },
            },
        }
    ),
    # OK
    "embeddings": MockConfig(
        response_data={
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": [0.1234, -0.5678, 0.9012, -0.3456], "index": 0}
            ],
            "model": "text-embedding-ada-002",
            "usage": {"prompt_tokens": 8, "total_tokens": 8},
        }
    ),
    # TODO: check this!
    "files": MockConfig(
        response_data={
            "id": "file-123456789",
            "object": "file",
            "bytes": 1024,
            "created_at": 1694268190,
            "filename": "uploaded_file.json",
            "purpose": "fine-tune",
        }
    ),
    # OK
    "images/generations": MockConfig(
        response_data={
            "created": 1713833628,
            "data": [
                {
                    "b64_json": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
                }
            ],
            "usage": {
                "total_tokens": 100,
                "input_tokens": 50,
                "output_tokens": 50,
                "input_tokens_details": {"text_tokens": 10, "image_tokens": 40},
            },
        }
    ),
    #
    "responses": MockConfig(
        response_data={
            "created_at": 1741476542,
            "completed_at": 1741476543,
            "id": "resp_67cb71b3c2b0819084d481baaaf148f206981a8637e6bc44",
            "max_output_tokens": 50,
            "model": "gpt-4.1-nano-2025-04-14",
            "object": "response",
            "output": [
                {
                    "type": "message",
                    "id": "msg_67cb71b3c2b0819084d481baaaf148f206981a8637e6bc44",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Hello! I'm doing well, thank you. How can I assist you today?",
                            "annotations": [],
                        }
                    ],
                }
            ],
            "parallel_tool_calls": True,
            "reasoning": {},
            "status": "completed",
            "store": True,
            "temperature": 1.0,
            "text": {"format": {"type": "text"}, "verbosity": "medium"},
            "tool_choice": "auto",
            "tools": [],
            "top_p": 1.0,
            "truncation": "disabled",
            "usage": {
                "input_tokens": 13,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 17,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 30,
            },
            "metadata": {},
        }
    ),
}

default_mock_get_responses = {
    "responses/id": MockConfig(
        response_data={
            "completed_at": 1769125419,
            "created_at": 1769125418.0,
            "id": "resp_67cb71b3c2b0819084d481baaaf148f206981a8637e6bc44",
            "max_output_tokens": 50,
            "metadata": {},
            "model": "gpt-4.1-nano-2025-04-14",
            "object": "response",
            "output": [
                {
                    "content": [
                        {
                            "annotations": [],
                            "text": "Hello! I'm doing well, thank you. How are "
                            "you today? How can I assist you?",
                            "type": "output_text",
                        }
                    ],
                    "id": "msg_67cb71b3c2b0819084d481baaaf148f206981a8637e6bc44",
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": True,
            "reasoning": {},
            "status": "completed",
            "store": True,
            "temperature": 1.0,
            "text": {"format": {"type": "text"}, "verbosity": "medium"},
            "tool_choice": "auto",
            "tools": [],
            "truncation": "disabled",
            "usage": {
                "input_tokens": 13,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 21,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 34,
            },
        }
    ),
    "responses/id/input_items": MockConfig(
        response_data={
            "data": [
                {
                    "content": [{"text": "Hello, how are you?", "type": "input_text"}],
                    "id": "msg_67cb71b3c2b0819084d481baaaf148f206981a8637e6bc44",
                    "role": "user",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "first_id": "msg_67cb71b3c2b0819084d481baaaf148f206981a8637e6bc44",
            "has_more": False,
            "last_id": "msg_67cb71b3c2b0819084d481baaaf148f206981a8637e6bc44",
            "object": "list",
        }
    ),
}

special_mock_responses: dict[str, MockConfig] = {}


@app.get("/health")
async def health_check():
    return JSONResponse({"status": "ok"})


@app.post("/configure/{path:path}")
async def configure_endpoint(path: str, config: MockConfig | MockStreamingConfig):
    """
    Configure a special mock response for a specific endpoint.

    This endpoint allows to dynamically configure a mock response for any endpoint,
    regardless of the request method (GET, POST). It can be used when the default
    response from `default_mock_post_responses` is not what one needs.
    """
    special_mock_responses[path] = config
    return {"message": f"Configured a special mock response for {path}"}


@app.post("/reset/{path:path}")
async def reset_endpoint(path: str):
    """
    Reset the special mock response for a specific endpoint to its default behavior.

    A request to this endpoint should be sent after a test with a special mock response
    finishes, to prevent tests from interfering with one another.
    """
    del special_mock_responses[path]
    return {"message": f"Reset the special mock response for {path}"}


@app.get("/{path:path}")
@app.post("/{path:path}")
async def mock_endpoint(path: str, request: Request):
    path = path.strip("/").removeprefix("v1/")

    try:
        if path in special_mock_responses:
            config = special_mock_responses[path]
        elif request.method == "POST":
            config = default_mock_post_responses[path]
        else:  # request.method == "GET"
            if re.match("^responses/.+/input_items$", path):
                config = default_mock_get_responses["responses/id/input_items"]
            elif re.match("responses/.+$", path):
                config = default_mock_get_responses["responses/id"]
            else:
                config = default_mock_get_responses[path]
    except KeyError:
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND, detail=f"No mock configured for this endpoint: {path}"
        )

    if isinstance(config, MockStreamingConfig):
        return StreamingResponse(
            content=config.response_data, status_code=config.status_code, headers=config.headers
        )
    return JSONResponse(
        content=config.response_data, status_code=config.status_code, headers=config.headers
    )


class MockAPIServer:
    def __init__(self, port: int = None, log_level: str = "error") -> None:
        self.host: str = "localhost"
        self.port: int = port or get_available_port()
        self.base_url: str = f"http://{self.host}:{self.port}"
        self.process: Optional[subprocess.Popen] = None
        self.log_level: str = log_level

    def start(self) -> None:
        """Start the uvicorn mock server in a subprocess"""
        print(f"\nStarting a mock uvicorn server on port {self.port}...")
        cmd = [
            "uvicorn",
            "gateway.tests.utils.mock_server:app",
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--log-level",
            self.log_level,
        ]
        self.process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        print(f"Waiting for the mock server to accept connections on port {self.port}...")
        start_time = time.time()
        timeout = 20

        while True:
            try:
                response = requests.get(f"{self.base_url}/health", timeout=0.5)
                if response.status_code == 200:
                    print(f"✓ Mock server started successfully on port {self.port}.")
                    break
            except requests.RequestException as err:
                if time.time() - start_time < timeout:
                    time.sleep(0.5)
                else:
                    raise RuntimeError(
                        f"Mock server failed to start within {timeout} s. Last error: {err}"
                    )

    def stop(self):
        """Stop the mock server"""
        if self.process:
            print("\nStopping mock server...")
            self.process.terminate()
            try:
                self.process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                outs, errs = self.process.communicate()
                print("Process did not terminate gracefully. Force killed.")
            self.process = None

    def configure_endpoint(self, path: str, config: MockConfig) -> None:
        normalized_path = path.strip("/").removeprefix("v1/")
        url = f"{self.base_url}/configure/{normalized_path}"
        if isinstance(config, MockConfig):
            response = requests.post(url, json=config.model_dump(mode="json"), timeout=1)
        else:
            response = requests.post(url, ...)  # TODO!
        response.raise_for_status()

    def reset_endpoint_config(self, path: str) -> None:
        normalized_path = path.strip("/").removeprefix("v1/")
        url = f"{self.base_url}/reset/{normalized_path}"
        response = requests.post(url, timeout=1)
        response.raise_for_status()

    @contextmanager
    def patch_external_api(self, url: str | None = None, config: MockConfig | None = None):
        """Context manager to temporarily patch external OpenAI API calls.

        If neither `url` nor `config` are provided, the default response from
        `default_mock_post_responses` will be used. If both args are provided,
        the response will be based on the `config`. Providing only one of the args
        raises `ValueError`.

        Args:
            url: The path to patch, e.g. "chat/completions"
            config: The mock configuration to use for creating the response
        """
        if (url is None) != (config is None):
            raise ValueError("Both 'url' and 'config' must be provided - or neither of them.")

        if config is not None:
            self.configure_endpoint(url, config)

        # TODO: make it more robust? Now it relies on the fact that AsyncOpenAI client
        #  tries to get the url from the env; maybe mock get_openai_client? or Router? like in `get_mock_router`?
        with patch.dict(
            "os.environ", {"OPENAI_BASE_URL": self.base_url, "OPENAI_API_KEY": "fake_openai_key"}
        ):
            try:
                yield
            finally:
                if config is not None:
                    self.reset_endpoint_config(url)
                else:
                    pass
                # Further cleanup handled by tearDown


def main():
    """Run the mock uvicorn server as a standalone script."""
    parser = argparse.ArgumentParser(description="Run a mock API server")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to run the server on. If not provided, a random unused port is chosen",
    )
    parser.add_argument("--log-level", type=str, default="error", help="Log level for uvicorn")
    # TODO: add option to add delays to responses
    # parser.add_argument("--delays", action="store_true", help="Add delays to responses")
    args = parser.parse_args()
    mock_server = MockAPIServer(port=args.port, log_level=args.log_level)
    try:
        mock_server.start()
        print(f"Server running on {mock_server.base_url}. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass  # Do not print out the traceback here, just exit cleanly
    finally:
        mock_server.stop()
        print("Server stopped.")


if __name__ == "__main__":
    main()
