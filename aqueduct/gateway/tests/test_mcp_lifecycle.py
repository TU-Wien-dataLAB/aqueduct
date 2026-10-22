"""Relay resource ownership at the HTTP endpoint and upstream transport boundary."""

import asyncio
import json
from typing import ClassVar
from unittest.mock import patch

import httpx
from asgiref.testing import ApplicationCommunicator
from django.core.asgi import get_asgi_application
from django.test import TransactionTestCase, override_settings
from django.urls import reverse

from gateway.tests.test_mcp import MCP_BODY, MCP_CONFIG, MCP_HEADERS, MCP_SERVER
from gateway.tests.utils.base import ROUTER_CONFIG_PATH, GatewayIntegrationTestCase
from gateway.tests.utils.mcp_transport import upstream


class FailingStream(httpx.AsyncByteStream):
    def __init__(self):
        self.closed = False

    async def __aiter__(self):
        yield b"partial response"
        raise httpx.ReadError("upstream disconnected")

    async def aclose(self):
        self.closed = True


class BlockingStream(httpx.AsyncByteStream):
    def __init__(self):
        self.waiting = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        yield b'data: {"jsonrpc":"2.0","method":"notifications/progress"}\n\n'
        self.waiting.set()
        await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True


@override_settings(
    MCP_ENABLE_DNS_REBINDING_PROTECTION=False,
    AUTHENTICATION_BACKENDS=["gateway.authentication.TokenAuthenticationBackend"],
    LITELLM_ROUTER_CONFIG_FILE_PATH=ROUTER_CONFIG_PATH,
)
class MCPRelayLifecycleTest(TransactionTestCase):
    fixtures = GatewayIntegrationTestCase.fixtures
    headers: ClassVar[dict[str, str]] = {"Authorization": "Bearer sk-123abc"}

    def setUp(self):
        super().setUp()
        self.url = reverse("gateway:mcp_server", kwargs={"name": MCP_SERVER})
        patcher = patch("gateway.views.mcp.get_mcp_config", return_value=MCP_CONFIG)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def post(self):
        return await self.async_client.post(
            self.url,
            data=json.dumps(MCP_BODY),
            content_type="application/json",
            headers={**self.headers, **MCP_HEADERS},
        )

    def communicator(self):
        headers = {
            **self.headers,
            **MCP_HEADERS,
            "Content-Type": "application/json",
            "Host": "testserver",
        }
        return ApplicationCommunicator(
            get_asgi_application(),
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": self.url,
                "query_string": b"",
                "headers": [
                    (key.lower().encode(), value.encode()) for key, value in headers.items()
                ],
                "server": ("testserver", 80),
                "client": ("127.0.0.1", 1234),
            },
        )

    async def test_client_disconnect_cancels_upstream_stream(self):
        stream = BlockingStream()
        response = httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
        with upstream(lambda request: response) as clients:
            connection = self.communicator()
            try:
                await connection.send_input(
                    {"type": "http.request", "body": json.dumps(MCP_BODY).encode()}
                )
                start = await connection.receive_output(timeout=2)
                self.assertEqual(start["status"], 200)
                chunk = await connection.receive_output(timeout=2)
                self.assertIn(b"notifications/progress", chunk["body"])
                await asyncio.wait_for(stream.waiting.wait(), timeout=2)
                await connection.send_input({"type": "http.disconnect"})
                await connection.wait(timeout=2)
                self.assertTrue(stream.closed)
                self.assertTrue(response.is_closed)
                self.assertTrue(clients[0].is_closed)
            finally:
                connection.stop()

    async def test_disconnect_before_upstream_headers_closes_client(self):
        waiting = asyncio.Event()

        async def pending_request(request):
            waiting.set()
            await asyncio.Event().wait()

        with upstream(pending_request) as clients:
            connection = self.communicator()
            try:
                await connection.send_input(
                    {"type": "http.request", "body": json.dumps(MCP_BODY).encode()}
                )
                await asyncio.wait_for(waiting.wait(), timeout=2)
                await connection.send_input({"type": "http.disconnect"})
                await connection.wait(timeout=2)
                self.assertTrue(clients[0].is_closed)
            finally:
                connection.stop()

    async def test_connection_failure_closes_upstream_client(self):
        def connection_failure(request):
            raise httpx.ConnectError("upstream unavailable", request=request)

        with upstream(connection_failure) as clients:
            with self.assertRaises(httpx.ConnectError):
                await self.post()
            self.assertTrue(clients[0].is_closed)

    async def test_sse_read_failure_closes_upstream_response_and_client(self):
        stream = FailingStream()
        response = httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
        with upstream(lambda request: response) as clients:
            downstream = await self.post()
            chunks = aiter(downstream)
            self.assertEqual(await anext(chunks), b"partial response")
            with self.assertRaises(httpx.ReadError):
                await anext(chunks)
            self.assertTrue(stream.closed)
            self.assertTrue(response.is_closed)
            self.assertTrue(clients[0].is_closed)

    async def test_json_read_failure_closes_upstream_response_and_client(self):
        stream = FailingStream()
        response = httpx.Response(200, headers={"content-type": "application/json"}, stream=stream)
        with upstream(lambda request: response) as clients:
            with self.assertRaises(httpx.ReadError):
                await self.post()
            self.assertTrue(stream.closed)
            self.assertTrue(response.is_closed)
            self.assertTrue(clients[0].is_closed)
