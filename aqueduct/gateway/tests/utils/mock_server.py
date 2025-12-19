import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import patch

import requests
from django.test import TestCase
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from gateway.tests.utils.helpers import get_available_port

ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent

app = FastAPI()


class MockConfig(BaseModel):
    status_code: int = 200
    response_data: Dict[str, Any] = {}
    headers: Optional[Dict[str, str]] = {"Content-Type": "application/json"}


mock_responses = {
    "POST:images/generations": MockConfig(
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
    )
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
async def mock_endpoint(path: str, request: Request, body: None | BaseModel = None):
    key = f"{request.method}:{path}"
    if key not in mock_responses:
        raise HTTPException(status_code=404, detail=f"No mock configured for this endpoint: {key}")

    config = mock_responses[key]

    return JSONResponse(
        content=config.response_data, status_code=config.status_code, headers=config.headers
    )


class MockAPIServer:
    def __init__(self) -> None:
        self.host: str = "localhost"
        self.port: int = get_available_port()
        self.base_url: str = f"http://{self.host}:{self.port}"
        self.process: Optional[subprocess.Popen] = None

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
            "error",
        ]
        self.process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        print(f"Waiting for the mock server to accept connections on port {self.port}...")
        start_time = time.time()
        timeout = 20
        last_error = None

        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}/health", timeout=0.5)
                if response.status_code == 200:
                    print(f"✓ Mock server started successfully on port {self.port}")
                    return
            except requests.RequestException as e:
                last_error = e
                time.sleep(0.5)

        raise RuntimeError(
            f"Mock server failed to start within {timeout} s. Last error: {last_error}"
        )

    def stop(self):
        """Stop the mock server"""
        if self.process:
            self.process.terminate()
            try:
                self.process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                outs, errs = self.process.communicate()
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


class OpenAITestCase(TestCase):
    """A test case running the mock server for external OpenAI requests."""

    fixtures = ["gateway_data.json"]
    mock_server = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.mock_server = MockAPIServer()

        try:
            cls.mock_server.start()
        except RuntimeError as err:
            print(err)
            print(f"Failed to connect to the mock server! Interrupting the {cls.__name__}")
            # In case of any errors during setup, `tearDownClass` is not called, which means
            # the uvicorn server subprocess is *not* terminated and continues to run
            # in the background even after the test process exists.
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.mock_server.stop()
        super().tearDownClass()
