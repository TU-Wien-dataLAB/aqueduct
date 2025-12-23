import argparse
import subprocess
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional
from unittest.mock import patch

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from gateway.tests.utils.helpers import get_available_port

app = FastAPI(debug=True)


class MockConfig(BaseModel):
    status_code: int = 200
    response_data: Dict[str, Any] | bytes = {}
    headers: Dict[str, str] = {"Content-Type": "application/json"}


class MockStreamingConfig(BaseModel):
    status_code: int = 200
    response_streaming_data: list[bytes] = []  # ??
    headers: Dict[str, str] = {"Content-Type": "text/event-stream"}


mock_responses = {
    # TODO: test them all, they were generated or guessed
    "audio/speech": MockStreamingConfig(response_streaming_data=[b"mock", b"audio", b"data"]),
    "audio/transcriptions": MockConfig(
        response_data={
            "text": "This is a mock transcription",
            "usage": {"type": "duration", "seconds": 60},
        }
    ),
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
}


@app.get("/health")
async def health_check():
    return JSONResponse({"status": "ok"})


@app.post("/configure/{method}/{path:path}")
async def configure_endpoint(method: str, path: str, config: MockConfig) -> Dict[str, str]:
    key = f"{method}:{path}"
    mock_responses[key] = config
    return {"message": f"Configured {method} {path}"}


@app.get("/{path:path}")
@app.post("/{path:path}")
@app.delete("/{path:path}")
async def mock_endpoint(path: str, request: Request):
    # key = f"{request.method}:{path}"
    path = path.strip("/").removeprefix("v1/")
    if path not in mock_responses:
        raise HTTPException(status_code=404, detail=f"No mock configured for this endpoint: {path}")

    config = mock_responses[path]

    return JSONResponse(
        content=config.response_data, status_code=config.status_code, headers=config.headers
    )
    # except Exception as e:
    #     import traceback
    #     logger.error(f"Error processing transcription: {str(e)}")
    #     logger.error(f"Traceback: {traceback.format_exc()}")
    #     return JSONResponse(
    #         status_code=500,
    #         content={
    #             "error": str(e),
    #             "detail": traceback.format_exc(),
    #             "request_data": request.body()
    #         }
    #     )


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

    def configure_endpoint(
        self,
        method: str,
        path: str,
        status_code: int = 200,
        response_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        url = f"{self.base_url}/configure/{method}/{path}"
        config = {
            "status_code": status_code,
            "response_data": response_data or {},
            "headers": headers,
        }
        response = requests.post(url, json=config, timeout=1)
        response.raise_for_status()

    @contextmanager
    def patch_external_api(self):
        # TODO: make it more robust? Now it relies on the fact that AsyncOpenAI client
        #  tries to get the url from the env; maybe mock get_openai_client? or Router? like in `get_mock_router`?
        with patch.dict(
            "os.environ", {"OPENAI_BASE_URL": self.base_url, "OPENAI_API_KEY": "fake_openai_key"}
        ):
            try:
                yield
            finally:
                pass  # Cleanup handled by tearDown


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
