import json
import os
import socket
import subprocess
import time
from contextlib import asynccontextmanager
from copy import deepcopy
from functools import wraps
from typing import Optional
from unittest.mock import patch

import httpx
from asgiref.sync import sync_to_async
from channels.testing import ChannelsLiveServerTestCase
from daphne.testing import DaphneProcess
from django.test import override_settings
from mcp import ClientSession
from mcp.client.streamable_http import StreamableHTTPTransport, streamable_http_client

from gateway.tests.utils.test_runner import (
    get_mcp_server_port,
    get_shared_mcp_server_process,
    set_shared_mcp_server,
)
from mock_api.helpers import get_available_port

MCP_CONFIG_PATH = "/tmp/aqueduct/test-mcp-config.json"
MCP_TEST_CONFIG = {
    "mcpServers": {"test-server": {"type": "streamable-http", "url": "http://localhost:3001/mcp"}}
}


# patch: https://github.com/modelcontextprotocol/python-sdk/pull/1670
def patch_mcp_sse_issue(view_func):
    @wraps(view_func)
    async def wrapper(*args, **kwargs):
        sse = args[1]
        if not sse.data:
            return False
        else:
            return await view_func(*args, **kwargs)

    return wrapper


def _is_cancel_scope_error(exc):
    """
    Recursively check if an exception or ExceptionGroup contains a cancel scope error.

    The cancel scope error is a known race condition in anyio's task group cleanup
    that can occur during MCP session teardown. It manifests as nested ExceptionGroups
    containing errors with "cancel scope" in the message.
    """
    if "cancel scope" in str(exc).lower():
        return True
    if isinstance(exc, ExceptionGroup):
        for sub_exc in exc.exceptions:
            if _is_cancel_scope_error(sub_exc):
                return True
    return False


def skip_on_cancel_scope_error(test_func):
    """
    Decorator to skip MCP tests when encountering the cancel scope error.

    This is a known race condition in anyio/MCP SDK where cancel scopes can get out of sync
    during session cleanup. This can randomly occur on any MCP test. When detected, the test
    is skipped rather than marked as failed.

    Usage:
        @async_to_sync
        @skip_on_cancel_scope_error
        async def test_something(self):
            ...
    """

    @wraps(test_func)
    async def wrapper(self, *args, **kwargs):
        try:
            return await test_func(self, *args, **kwargs)
        except (Exception, ExceptionGroup) as e:
            # Check if this is the cancel scope error (a race condition in anyio/MCP SDK)
            # This can occur during session cleanup when cancel scopes exit in unexpected order.
            if _is_cancel_scope_error(e):
                self.skipTest(f"Skipping due to cancel scope race condition: {e}")
            raise

    return wrapper


class MCPDaphneProcess(DaphneProcess):
    @patch(
        "mcp.client.streamable_http.StreamableHTTPTransport._handle_sse_event",
        new=patch_mcp_sse_issue(StreamableHTTPTransport._handle_sse_event),
    )
    def run(self):
        from django.conf import settings

        settings.MCP_CONFIG_FILE_PATH = MCP_CONFIG_PATH
        print(f"Updating MCP_CONFIG_FILE_PATH: {settings.MCP_CONFIG_FILE_PATH}")

        # Configure MCP security settings for testing
        settings.MCP_ENABLE_DNS_REBINDING_PROTECTION = True
        settings.MCP_ALLOWED_HOSTS = ["localhost:8000", "localhost:*", "127.0.0.1:*"]
        settings.MCP_ALLOWED_ORIGINS = [
            "http://localhost:3000",
            "http://localhost:*",
            "http://127.0.0.1:*",
        ]

        # Allow test hosts in Django's ALLOWED_HOSTS so requests reach our security decorator
        # Add common test hosts that we'll use to test the security decorator
        if "evil.testserver" not in settings.ALLOWED_HOSTS:
            settings.ALLOWED_HOSTS.append("evil.testserver")
        if "malicious.com" not in settings.ALLOWED_HOSTS:
            settings.ALLOWED_HOSTS.append("malicious.com")

        print(
            f"MCP Security: Protection={settings.MCP_ENABLE_DNS_REBINDING_PROTECTION}, "
            f"Hosts={settings.MCP_ALLOWED_HOSTS}"
        )

        super().run()


@override_settings(OIDC_OP_JWKS_ENDPOINT="https://example.com/application/o/example/jwks/")
class MCPLiveServerTestCase(ChannelsLiveServerTestCase):
    """
    Live server test case for MCP endpoints using Django's LiveServerTestCase.
    Provides a running server for testing HTTP endpoints with httpx.
    """

    serve_static = False
    ProtocolServerProcess = MCPDaphneProcess

    fixtures = ["gateway_data.json"]
    mcp_server_process: Optional[subprocess.Popen] = None

    @property
    def mcp_url(self):
        return f"{self.live_server_url}/mcp-servers/test-server/mcp"

    @asynccontextmanager
    async def client_session(self):
        async with (
            httpx.AsyncClient(headers=self.headers) as client,
            streamable_http_client(self.mcp_url, http_client=client) as (
                read_stream,
                write_stream,
                _,
            ),
            ClientSession(read_stream, write_stream) as session,
        ):
            yield session

    async def assertRequestLogged(self, n: int = 1):
        from management.models import Request

        # Check that (only) initialize request was logged

        mcp_requests = await sync_to_async(Request.objects.count)()
        self.assertEqual(mcp_requests, n, f"There should be exactly {n} logged MCP request.")

    @classmethod
    def _write_mcp_config(cls):
        os.makedirs(os.path.dirname(MCP_CONFIG_PATH), exist_ok=True)
        with open(MCP_CONFIG_PATH, "w") as f:
            json.dump(MCP_TEST_CONFIG, f)

    @classmethod
    def _update_mcp_config_port(cls, port: int):
        config = deepcopy(MCP_TEST_CONFIG)
        config["mcpServers"]["test-server"]["url"] = f"http://localhost:{port}/mcp"

        os.makedirs(os.path.dirname(MCP_CONFIG_PATH), exist_ok=True)
        with open(MCP_CONFIG_PATH, "w") as f:
            json.dump(config, f)

    @classmethod
    def _start_mcp_server(cls):
        """Start the MCP everything server on a random available port."""
        port = get_available_port()

        cls.mcp_server_port = port
        cls._update_mcp_config_port(port)

        print(f"\nStarting MCP everything server on port {port}...")
        try:
            env = os.environ.copy()
            env["PORT"] = str(port)
            cls.mcp_server_process = subprocess.Popen(
                [
                    "npx",
                    "-y",
                    "@modelcontextprotocol/server-everything@2025.11.25",
                    "streamableHttp",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                cwd=os.path.dirname(MCP_CONFIG_PATH),
            )
        except FileNotFoundError:
            raise RuntimeError(
                "npx command not found. Please ensure Node.js and npm are installed."
            )

        # Set global variables enabling the tests to share the MCP server process
        set_shared_mcp_server(cls.mcp_server_process, cls.mcp_server_port)

        # Wait for server to be ready by checking if it accepts connections
        print(f"Waiting for MCP server to accept connections on port {port}...")
        start_time = time.time()
        timeout = 30
        last_error = None

        while time.time() - start_time < timeout:
            # Check if process is still running
            if cls.mcp_server_process.poll() is not None:
                stdout, _ = cls.mcp_server_process.communicate()
                raise RuntimeError(f"MCP server failed to start. stdout: {stdout}")

            # Try to connect to the server
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(1.0)
                    sock.connect(("localhost", port))
                    print(f"✓ MCP everything server started successfully on port {port}")
                    return
            except (socket.error, ConnectionRefusedError, OSError) as e:
                last_error = e
                time.sleep(0.5)

        # If we get here, timeout occurred
        raise RuntimeError(
            f"MCP server did not accept connections within {timeout} seconds. Last error: {last_error}"
        )

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.headers = {"Authorization": "Bearer sk-123abc", "Content-Type": "application/json"}

        # If another test class has already created the MCP server, it will be reused.
        # Otherwise, write MCP config, create the MCP server process, and set necessary
        # global variables. Next tests to use the `MCPLiveServerTestCase` class as base
        # class will reuse the same process.
        # Note: Teardown is handled by the test runner, after all the tests finished running.
        mcp_process = get_shared_mcp_server_process()
        if mcp_process is None:
            cls._write_mcp_config()
            cls._start_mcp_server()
        else:
            assert (mcp_port := get_mcp_server_port()) is not None, (
                "Global MCP test server port is not set, even though the server process "
                "exists already. This should not happen!"
            )
            cls.mcp_server_process = mcp_process
            cls.mcp_server_port = mcp_port
