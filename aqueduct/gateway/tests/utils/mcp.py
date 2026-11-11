"""Live MCP test infrastructure; the upstream is the real npm reference server."""

import json
import logging
import os
import signal
import subprocess
import time
from contextlib import asynccontextmanager, suppress
from http import HTTPStatus
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ClassVar

import httpx2
from channels.testing import ChannelsLiveServerTestCase
from daphne.testing import DaphneProcess
from django.test import override_settings
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

from gateway.tests.utils.test_runner import get_shared_mcp_server
from mock_api.helpers import get_available_port

logger = logging.getLogger(__name__)

MCP_PROTOCOL_VERSION = "2026-07-28"
MCP_PACKAGE = "@modelcontextprotocol/server-everything@2.0.0"
TASKS_EXTENSION = "io.modelcontextprotocol/tasks"
ROUTER_CONFIG_PATH = Path(__file__).resolve().parents[4] / "example_router_config.yaml"


def modern_params(params=None, capabilities=None):
    """The modern protocol's per-request identity/capabilities envelope."""
    return {
        **(params or {}),
        "_meta": {
            "io.modelcontextprotocol/protocolVersion": MCP_PROTOCOL_VERSION,
            "io.modelcontextprotocol/clientInfo": {"name": "aqueduct-tests", "version": "1"},
            "io.modelcontextprotocol/clientCapabilities": capabilities or {},
        },
    }


class MCPEverythingServer:
    """One lazy, suite-wide npm subprocess, with bounded startup and group cleanup."""

    def __init__(self):
        self.directory = TemporaryDirectory(prefix="aqueduct-mcp-")
        self.config_path = Path(self.directory.name) / "config.json"
        self.log_path = Path(self.directory.name) / "server.log"
        self.port = get_available_port()
        self.url = f"http://127.0.0.1:{self.port}/mcp"
        self.process = None
        self.log_file = None
        self.config_path.write_text(
            json.dumps(
                {"mcpServers": {"test-server": {"type": "streamable-http", "url": self.url}}}
            )
        )

    def start(self):
        self.log_file = self.log_path.open("w")
        try:
            self.process = subprocess.Popen(
                ["npx", "--prefer-offline", "-y", MCP_PACKAGE, "streamableHttp"],
                env={**os.environ, "PORT": str(self.port)},
                cwd=self.directory.name,
                stdout=self.log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = time.monotonic() + 60
            with httpx2.Client(timeout=1, trust_env=False) as client:
                while time.monotonic() < deadline:
                    if self.process.poll() is not None:
                        raise RuntimeError("MCP reference server exited during startup")
                    try:
                        response = client.post(
                            self.url,
                            headers={
                                "Accept": "application/json, text/event-stream",
                                "Mcp-Method": "server/discover",
                                "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
                            },
                            json={
                                "jsonrpc": "2.0",
                                "id": 1,
                                "method": "server/discover",
                                "params": modern_params(),
                            },
                        )
                    except httpx2.TransportError:
                        time.sleep(0.1)
                        continue
                    if response.status_code != HTTPStatus.OK or "result" not in response.json():
                        raise RuntimeError(f"MCP discovery failed: {response.text}")
                    logger.info("MCP reference server ready at %s", self.url)
                    return
            raise RuntimeError("MCP reference server was not ready within 60 seconds")
        except Exception as error:
            output = self.log_path.read_text()
            self.stop()
            raise RuntimeError(f"Unable to start {MCP_PACKAGE}: {error}\n{output}") from error

    def stop(self):
        try:
            if self.process is not None:
                # npx spawns node: terminate the whole new session, even if npx exited first.
                with suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    with suppress(ProcessLookupError):
                        os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait(timeout=5)
                # The npm launcher can exit before a stubborn child does.
                with suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGKILL)
        finally:
            self.process = None
            if self.log_file is not None:
                self.log_file.close()
                self.log_file = None
            self.directory.cleanup()


class MCPDaphneProcess(DaphneProcess):
    def __init__(self, *args, **kwargs):
        # Capture in the parent: macOS multiprocessing uses spawn, not fork.
        self.mcp_config_path = str(get_shared_mcp_server().config_path)
        super().__init__(*args, **kwargs)

    def start(self):
        super().start()
        deadline = time.monotonic() + 30
        while not self.ready.wait(timeout=0.1):
            if not self.is_alive() or time.monotonic() >= deadline:
                self.terminate()
                self.join(timeout=5)
                if self.is_alive():
                    self.kill()
                    self.join(timeout=5)
                raise RuntimeError("Live Django ASGI server failed to start within 30 seconds")

    def run(self):
        from django.conf import settings

        settings.MCP_CONFIG_FILE_PATH = self.mcp_config_path
        settings.LITELLM_ROUTER_CONFIG_FILE_PATH = str(ROUTER_CONFIG_PATH)
        settings.AUTHENTICATION_BACKENDS = ["gateway.authentication.TokenAuthenticationBackend"]
        settings.MCP_ENABLE_DNS_REBINDING_PROTECTION = True
        settings.MCP_ALLOWED_HOSTS = ["localhost:*", "127.0.0.1:*"]
        settings.MCP_ALLOWED_ORIGINS = ["http://localhost:*", "http://127.0.0.1:*"]
        # Let deliberately bad hosts reach the MCP transport-security decorator.
        settings.ALLOWED_HOSTS = ["*"]
        super().run()


@override_settings(OIDC_OP_JWKS_ENDPOINT="https://example.com/application/o/example/jwks/")
class MCPLiveServerTestCase(ChannelsLiveServerTestCase):
    """Real ASGI gateway + real reference server; no mocked transport or relay."""

    serve_static = False
    ProtocolServerProcess = MCPDaphneProcess
    fixtures: ClassVar[list[str]] = ["gateway_data.json"]

    @classmethod
    def setUpClass(cls):
        cls.upstream = get_shared_mcp_server()
        cls.headers = {"Authorization": "Bearer sk-123abc"}
        try:
            super().setUpClass()
        except BaseException:
            # Channels only restores this setting on a successful class teardown.
            modified_settings = getattr(cls, "_live_server_modified_settings", None)
            if modified_settings is not None:
                modified_settings.disable()
            raise

    @property
    def mcp_url(self):
        return f"{self.live_server_url}/mcp-servers/test-server/mcp"

    @asynccontextmanager
    async def mcp_client(self, **kwargs):
        # SDK v2 uses httpx2 (not the gateway's httpx), and Client negotiates
        # server/discover automatically. Disable its cache so calls hit the relay.
        async with (
            httpx2.AsyncClient(headers=self.headers, timeout=10, trust_env=False) as http_client,
            Client(
                streamable_http_client(self.mcp_url, http_client=http_client),
                read_timeout_seconds=10,
                cache=None,
                **kwargs,
            ) as client,
        ):
            yield client

    async def rpc(
        self, method, params=None, *, headers=None, capabilities=None, request_id=1
    ) -> tuple[httpx2.Response, dict[str, Any]]:
        """Send JSON-RPC without the SDK; return the HTTP response and parsed JSON/SSE payload."""
        params = modern_params(params, capabilities)
        full_headers = {
            **self.headers,
            "Accept": "application/json, text/event-stream",
            "Mcp-Method": method,
            "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
        }
        name = params.get("uri" if method == "resources/read" else "name")
        if method in {"tools/call", "prompts/get", "resources/read"} and name is not None:
            full_headers["Mcp-Name"] = name
        # The tasks extension declares taskId as its name parameter. The relay
        # need not know that extension: its client supplies Mcp-Name explicitly.
        if method in {"tasks/get", "tasks/update", "tasks/cancel"}:
            full_headers["Mcp-Name"] = params["taskId"]
        full_headers.update(headers or {})
        async with httpx2.AsyncClient(timeout=10, trust_env=False) as client:
            response = await client.post(
                self.mcp_url,
                headers=full_headers,
                json={"jsonrpc": "2.0", "id": request_id, "method": method, "params": params},
            )
        if "text/event-stream" in response.headers.get("content-type", ""):
            messages = [
                json.loads(line[5:])
                for line in response.text.splitlines()
                if line.startswith("data:")
            ]
            finals = [message for message in messages if message.get("id") == request_id]
            self.assertEqual(len(finals), 1, response.text)
            payload = finals[0]
        else:
            payload = response.json()
        if "jsonrpc" in payload:
            self.assertEqual(payload["id"], request_id)
        return response, payload

    async def assert_requests_logged(self, count):
        from management.models import Request

        requests = Request.objects.filter(path="/mcp-servers/test-server/mcp")
        self.assertEqual(await requests.acount(), count)
        self.assertEqual(await requests.exclude(status_code=200).acount(), 0)
