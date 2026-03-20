import argparse
import logging
import logging.config
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from http import HTTPStatus
from unittest.mock import patch

import requests

from mock_api.helpers import get_available_port
from mock_api.mock_configs import MockConfig

# Since the mock server is running as a standalone application and not within the Django app,
# we need to separately configure logging for it.
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler", "formatter": "standard",
        },
    },
    "loggers": {
        "mock_server": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
    },
}  # fmt: skip
logging.config.dictConfig(LOGGING_CONFIG)

logger = logging.getLogger("mock_server")


class MockAPIServer:
    def __init__(
        self, host: str = "localhost", port: int | None = None, delays: bool = True, log_level: str = "error"
    ) -> None:
        self.host: str = host
        self.port: int = port or get_available_port()
        self.base_url: str = f"http://{self.host}:{self.port}"
        self.delays: bool = delays
        self.process: subprocess.Popen | None = None
        self.log_level: str = log_level
        self.logger = logging.getLogger("mock_server")

    def start(self) -> None:
        """Start the uvicorn mock server in a subprocess"""
        self.logger.info("Starting a mock uvicorn server on port %s", self.port)
        cmd = [
            "uvicorn",
            "mock_api.endpoints:app",
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--log-level",
            self.log_level,
        ]

        # Set PYTHONPATH to include current directory so uvicorn can import the mock_api module
        # (necessary for the github pipeline)
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(sys.path)
        env["MOCK_API_DELAYS"] = "true" if self.delays else "false"
        self.process = subprocess.Popen(cmd, text=True, env=env)  # noqa: S603

        self.logger.debug("Waiting for the mock server to accept connections on port %s", self.port)
        start_time = time.time()
        timeout = 20

        while True:
            try:
                response = requests.get(f"{self.base_url}/health", timeout=0.5)
                if response.status_code == HTTPStatus.OK:
                    break
            except requests.RequestException as err:
                if time.time() - start_time < timeout:
                    time.sleep(0.5)
                else:
                    self.logger.exception("Mock server failed to start within %s s. Last error: %s", timeout, err)
                    raise RuntimeError(f"Mock server failed to start within {timeout} s. Last error: {err}") from err

    def stop(self):
        """Stop the mock server"""
        self.process.terminate()
        try:
            self.process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.communicate()
            self.logger.warning("Process did not terminate gracefully. Force killed.")
        self.process = None

    def configure_endpoint(self, path: str, config: MockConfig) -> None:
        normalized_path = path.strip("/").removeprefix("v1/")
        url = f"{self.base_url}/configure/{normalized_path}"
        logger.debug("Configuring the %s endpoint", normalized_path)
        response = requests.post(url, json=config.model_dump(mode="json"), timeout=1)
        response.raise_for_status()

    def reset_endpoint_config(self, path: str) -> None:
        normalized_path = path.strip("/").removeprefix("v1/")
        url = f"{self.base_url}/reset/{normalized_path}"
        logger.debug("Resetting the %s endpoint", normalized_path)
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
            url (optional): The path to patch, e.g. "chat/completions"
            config (optional): The mock configuration to use for creating the response
        """
        if (url is None) != (config is None):
            raise ValueError("Both 'url' and 'config' must be provided - or neither of them.")

        if config is not None:
            self.configure_endpoint(url, config)

        with patch.dict("os.environ", {"OPENAI_BASE_URL": self.base_url, "OPENAI_API_KEY": "fake_openai_key"}):
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
        "--host",
        type=str,
        default="localhost",
        help="Host to bind the server to (use '0.0.0.0' for Docker, 'localhost' when running Django tests)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to run the server on. If not provided, a random unused port is chosen",
    )
    parser.add_argument("--delays", action="store_true", help="Add delays to responses")
    parser.add_argument("--log-level", type=str, default="error", help="Log level for uvicorn")
    args = parser.parse_args()
    mock_server = MockAPIServer(host=args.host, port=args.port, delays=args.delays, log_level=args.log_level)
    try:
        mock_server.start()
        logger.info("Mock server running on %s. Press Ctrl+C to stop.", mock_server.base_url)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass  # Do not print out the traceback here, just exit cleanly
    finally:
        mock_server.stop()
        logger.info("Mock server stopped.")


if __name__ == "__main__":
    main()
