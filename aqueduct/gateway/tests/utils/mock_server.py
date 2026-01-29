import argparse
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from json import JSONDecodeError
from typing import Optional
from unittest.mock import patch

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.requests import Request
from starlette.status import HTTP_404_NOT_FOUND

from gateway.tests.utils.helpers import get_available_port
from gateway.tests.utils.mock_configs import (
    MockConfig,
    MockPlainTextConfig,
    MockStreamingConfig,
    default_delete_configs,
    default_get_configs,
    default_post_configs,
    default_post_stream_configs,
    special_configs,
)

app = FastAPI(debug=True)


@app.get("/health")
async def health_check():
    return JSONResponse({"status": "ok"})


@app.post("/configure/{path:path}")
async def configure_endpoint(
    path: str, config: MockConfig | MockStreamingConfig | MockPlainTextConfig
):
    """
    Configure a special mock response for a specific endpoint.

    This endpoint allows to dynamically configure a mock response for any endpoint,
    regardless of the request method (GET, POST, DELETE). It can be used when the default
    response from `default_{method}_configs` is not what one needs.
    """
    special_configs[path] = config
    return {"message": f"Configured a special mock response for {path}"}


@app.post("/reset/{path:path}")
async def reset_endpoint(path: str):
    """
    Reset the special mock response for a specific endpoint to its default behavior.

    A request to this endpoint should be sent after a test with a special mock response
    finishes, to prevent tests from interfering with one another.
    """
    del special_configs[path]
    return {"message": f"Reset the special mock response for {path}"}


# TODO: Do we need to mock any other GET/DELETE endpoints? Refactor the response
#   matching to get rid of those if..elifs. Maybe create separate endpoints for
#   GET and DELETE?
@app.delete("/{path:path}")
@app.get("/{path:path}")
@app.post("/{path:path}")
async def mock_endpoint(path: str, request: Request):
    path = path.strip("/").removeprefix("v1/")

    try:
        if path in special_configs:
            # Special config can be a streaming, a JSON, or a plain text one.
            config = special_configs[path]
        elif await _should_stream(request):
            # Some streaming configs are stored in a separate dict, because some
            # endpoints can return both a streaming or a JSON response.
            # Note: streaming responses can only be returned for POST requests.
            config = default_post_stream_configs[path]
        elif request.method == "POST":
            config = default_post_configs[path]
        elif request.method == "GET":
            if re.match("^responses/.+/input_items$", path):
                config = default_get_configs["responses/id/input_items"]
            elif re.match("responses/.+$", path):
                config = default_get_configs["responses/id"]
            else:
                config = default_get_configs[path]
        elif request.method == "DELETE":
            if re.match("^responses/.+$", path):
                config = default_delete_configs["responses/id"]
            else:
                config = default_delete_configs[path]
    except KeyError:
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND, detail=f"No mock configured for this endpoint: {path}"
        )

    if isinstance(config, MockStreamingConfig):
        return StreamingResponse(
            content=config.response_data, status_code=config.status_code, headers=config.headers
        )
    elif isinstance(config, MockPlainTextConfig):
        return PlainTextResponse(
            content=config.response_data, status_code=config.status_code, headers=config.headers
        )
    return JSONResponse(
        content=config.response_data, status_code=config.status_code, headers=config.headers
    )


async def _should_stream(request: Request) -> bool:
    try:
        request_data = await request.json()
        should_stream = request_data.get("stream") is True
    except (AttributeError, JSONDecodeError, UnicodeDecodeError):
        should_stream = False
    return should_stream


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

        # Set PYTHONPATH to include current directory so uvicorn can import the gateway module
        # (necessary for the github pipeline)
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(sys.path)
        self.process = subprocess.Popen(cmd, text=True, env=env)

        print(f"Waiting for the mock server to accept connections on port {self.port}...")
        start_time = time.time()
        timeout = 20

        while True:
            try:
                response = requests.get(f"{self.base_url}/health", timeout=0.5)
                if response.status_code == 200:
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
        self.process.terminate()
        try:
            self.process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.communicate()
            print("Process did not terminate gracefully. Force killed.")
        self.process = None

    def configure_endpoint(self, path: str, config: MockConfig) -> None:
        normalized_path = path.strip("/").removeprefix("v1/")
        url = f"{self.base_url}/configure/{normalized_path}"
        response = requests.post(url, json=config.model_dump(mode="json"), timeout=1)
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
