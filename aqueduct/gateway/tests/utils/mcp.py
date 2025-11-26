import json
import os
import socket
import subprocess
import time
from contextlib import asynccontextmanager
from copy import deepcopy
from functools import wraps
from unittest.mock import patch

from asgiref.sync import sync_to_async
from channels.testing import ChannelsLiveServerTestCase
from daphne.testing import DaphneProcess
from mcp import ClientSession
from mcp.client.streamable_http import StreamableHTTPTransport, streamablehttp_client

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
            f"MCP Security: Protection={settings.MCP_ENABLE_DNS_REBINDING_PROTECTION}, Hosts={settings.MCP_ALLOWED_HOSTS}"
        )

        super().run()


class MCPLiveServerTestCase(ChannelsLiveServerTestCase):
    """
    Live server test case for MCP endpoints using Django's LiveServerTestCase.
    Provides a running server for testing HTTP endpoints with httpx.
    """

    serve_static = False
    ProtocolServerProcess = MCPDaphneProcess

    fixtures = ["gateway_data.json"]
    mcp_server_process = None

    @property
    def headers(self):
        # ChannelsLiveServerTestCase raises AppRegistryNotReady if I try to access model classes (through import)
        # -> so we hard-code the headers here again
        return {"Authorization": "Bearer sk-123abc", "Content-Type": "application/json"}

    @property
    def mcp_url(self):
        return f"{self.live_server_url}/mcp-servers/test-server/mcp"

    @asynccontextmanager
    async def client_session(self):
        async with streamablehttp_client(self.mcp_url, headers=self.headers) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                yield session

    async def assertRequestLogged(self, n: int = 1):
        # Check that (only) initialize request was logged
        from management.models import Request

        mcp_requests = await sync_to_async(list)(Request.objects.all())
        self.assertEqual(len(mcp_requests), n, f"There should be exactly {n} logged MCP request.")

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
        # Find a random available port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            port = s.getsockname()[1]

        cls.mcp_server_port = port
        cls._update_mcp_config_port(port)

        print(f"\nStarting MCP everything server on port {port}...")
        try:
            env = os.environ.copy()
            env["PORT"] = str(port)
            cls.mcp_server_process = subprocess.Popen(
                ["npx", "@modelcontextprotocol/server-everything@2025.11.25", "streamableHttp"],
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
    def _stop_mcp_server(cls):
        """Stop the MCP everything server."""
        if cls.mcp_server_process and cls.mcp_server_process.poll() is None:
            print("Stopping MCP everything server...")
            cls.mcp_server_process.terminate()
            try:
                cls.mcp_server_process.wait(timeout=10)
                print("MCP server stopped successfully")
            except subprocess.TimeoutExpired:
                print("MCP server did not stop gracefully, forcing kill")
                cls.mcp_server_process.kill()
                cls.mcp_server_process.wait()
            cls.mcp_server_process = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._write_mcp_config()
        cls._start_mcp_server()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        cls._stop_mcp_server()
        if os.path.exists(MCP_CONFIG_PATH):
            os.remove(MCP_CONFIG_PATH)
