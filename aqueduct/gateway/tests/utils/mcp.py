import fcntl
import os
import socket
import subprocess
import time
from contextlib import asynccontextmanager

from channels.testing import ChannelsLiveServerTestCase
from daphne.testing import DaphneProcess
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

MCP_CONFIG_PATH = "/tmp/aqueduct/test-mcp-config.json"
MCP_TEST_CONFIG = {
    "mcpServers": {"test-server": {"type": "streamable-http", "url": "http://localhost:3001/mcp"}}
}


class MCPDaphneProcess(DaphneProcess):
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
        # ChannelsLiveServerTestCase really hates if I try to access model classes (through import)
        # -> stops raising AppRegistryNotReady
        return {"Authorization": "Bearer sk-123abc"}

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

    @classmethod
    def _write_mcp_config(cls):
        import json

        os.makedirs(os.path.dirname(MCP_CONFIG_PATH), exist_ok=True)
        with open(MCP_CONFIG_PATH, "w") as f:
            json.dump(MCP_TEST_CONFIG, f)

    @classmethod
    def _update_mcp_config_port(cls, port: int):
        import json

        config = MCP_TEST_CONFIG.copy()
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
                ["npx", "@modelcontextprotocol/server-everything", "streamableHttp"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                cwd=os.path.dirname(MCP_CONFIG_PATH),
            )

            # Set stdout to non-blocking mode
            if cls.mcp_server_process.stdout:
                fd = cls.mcp_server_process.stdout.fileno()
                flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

            # Wait for server to be ready by reading output
            start_time = time.time()
            timeout = 10  # 10 second timeout
            output_buffer = ""

            while time.time() - start_time < timeout:
                # Check if process is still running
                if cls.mcp_server_process.poll() is not None:
                    stdout, _ = cls.mcp_server_process.communicate()
                    raise RuntimeError(f"MCP server failed to start. stdout: {stdout}")

                # Try to read some output
                try:
                    if cls.mcp_server_process.stdout:
                        chunk = cls.mcp_server_process.stdout.read()
                        if chunk is not None and chunk:
                            output_buffer += chunk
                            # Process complete lines
                            while "\n" in output_buffer:
                                line, output_buffer = output_buffer.split("\n", 1)
                                if line.strip():
                                    print(f"MCP Server: {line.strip()}")
                                    if "MCP Streamable HTTP Server listening on port" in line:
                                        print(
                                            f"✓ MCP everything server started successfully on port {port}"
                                        )
                                        return
                except (OSError, BlockingIOError, TypeError):
                    pass

                time.sleep(0.1)

            # If we get here, timeout occurred
            raise RuntimeError(
                f"MCP server did not start within {timeout} seconds. Last output: {output_buffer}"
            )
        except FileNotFoundError:
            raise RuntimeError(
                "npx command not found. Please ensure Node.js and npm are installed."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to start MCP server: {e}") from e

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
