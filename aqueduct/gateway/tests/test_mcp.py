"""Gateway tests for the stateless (2026-07-28) MCP endpoint.

These tests exercise the Aqueduct gateway's MCP behavior: authentication, the
required ``Mcp-Method`` header, header/routing metadata on the relay, response
passthrough (JSON and SSE), transport security, and exclusion. The upstream MCP
server is mocked with ``httpx.MockTransport`` because real servers still speak
the old handshake-based protocol; what matters here is how the gateway relays
each self-contained request.
"""

import json
from unittest.mock import patch

import httpx
from django.test import override_settings
from django.urls import reverse
from mcp.types import jsonrpc_message_adapter

from gateway.tests.utils.base import GatewayIntegrationTestCase
from management.models import Org

MCP_SERVER = "test-server"
MCP_URL = f"http://upstream.example/{MCP_SERVER}/mcp"
MCP_CONFIG = {MCP_SERVER: {"type": "streamable-http", "url": MCP_URL}}

# The 2026-07-28 required transport headers.
MCP_HEADERS = {"Mcp-Method": "tools/call", "Mcp-Name": "echo", "MCP-Protocol-Version": "2026-07-28"}

MCP_BODY = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {"name": "echo", "arguments": {"message": "hi"}},
}


def _json_rpc_response_handler(request: httpx.Request) -> httpx.Response:
    """Return a JSON-RPC response echoing the Mcp-* headers the relay sent."""
    received = jsonrpc_message_adapter.validate_json(request.content)
    return httpx.Response(
        200,
        json={
            "jsonrpc": "2.0",
            "id": getattr(received, "id", None),
            "result": {
                "relayed": {
                    "method": request.headers.get("mcp-method"),
                    "name": request.headers.get("mcp-name"),
                    "protocol_version": request.headers.get("mcp-protocol-version"),
                    "body_params": getattr(received, "params", None),
                }
            },
        },
    )


def _sse_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=b'data: {"jsonrpc":"2.0","id":1,"result":{"ok":true}}\n\n',
    )


def _patch_client(handler) -> patch:
    """Patch the relay's httpx.AsyncClient to use an in-memory MockTransport."""
    from gateway.views import mcp as mcp_views

    real_async_client = httpx.AsyncClient  # capture the unpatched class

    def factory(*args, **kwargs):
        return real_async_client(transport=httpx.MockTransport(handler))

    return patch.object(mcp_views.httpx, "AsyncClient", new=factory)


@override_settings(MCP_ENABLE_DNS_REBINDING_PROTECTION=False)
class MCPRelayTest(GatewayIntegrationTestCase):
    """Stateless relay: required header, forwarding, passthrough, independence."""

    def setUp(self):
        super().setUp()
        self.url = reverse("gateway:mcp_server", kwargs={"name": MCP_SERVER})
        patcher = patch("gateway.views.mcp.get_mcp_config", return_value=MCP_CONFIG)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _post(self, body=None, headers=None, content_type="application/json", include_method=True):
        full = {**self.headers, "Content-Type": content_type}
        if include_method:
            full["Mcp-Method"] = MCP_HEADERS["Mcp-Method"]
        if headers:
            full.update(headers)
        return self.client.post(
            self.url, data=json.dumps(body or MCP_BODY), content_type=content_type, headers=full
        )

    def test_requires_mcp_method_header(self):
        # No Mcp-Method header -> 400 (the relay requires self-describing headers).
        resp = self._post(include_method=False)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Mcp-Method", resp.json()["error"]["message"])

    def test_stateless_relay_forwards_headers_and_returns_response(self):
        with _patch_client(_json_rpc_response_handler):
            resp = self._post(headers=MCP_HEADERS)
        self.assertEqual(resp.status_code, 200)
        result = resp.json()["result"]["relayed"]
        self.assertEqual(result["method"], "tools/call")
        self.assertEqual(result["name"], "echo")
        self.assertEqual(result["protocol_version"], "2026-07-28")
        self.assertEqual(result["body_params"]["name"], "echo")

    def test_stateless_requests_are_independent(self):
        # Two identical calls produce identical results (no shared session state).
        with _patch_client(_json_rpc_response_handler):
            r1 = self._post(headers=MCP_HEADERS)
            r2 = self._post(headers=MCP_HEADERS)
        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r1.json(), r2.json())

    def test_relay_streams_sse_response(self):
        from asgiref.sync import async_to_sync

        async def collect() -> bytes:
            return b"".join([chunk async for chunk in resp.streaming_content])

        with _patch_client(_sse_handler):
            resp = self._post(headers=MCP_HEADERS)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp["content-type"], "text/event-stream")
        self.assertIn(b"jsonrpc", async_to_sync(collect)())

    def test_get_rejected(self):
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 405)

    def test_delete_rejected(self):
        resp = self.client.delete(self.url)
        self.assertEqual(resp.status_code, 405)

    def test_server_requires_auth(self):
        # No Authorization header -> 401.
        resp = self.client.post(
            self.url,
            data=json.dumps(MCP_BODY),
            content_type="application/json",
            headers={"Content-Type": "application/json", **MCP_HEADERS},
        )
        self.assertEqual(resp.status_code, 401)


class MCPNotFoundTest(GatewayIntegrationTestCase):
    """Server config lookup errors."""

    @override_settings(MCP_ENABLE_DNS_REBINDING_PROTECTION=False)
    def test_nonexistent_server_returns_404(self):
        url = reverse("gateway:mcp_server", kwargs={"name": "ghost"})
        with patch("gateway.views.mcp.get_mcp_config", return_value={}):
            resp = self.client.post(
                url,
                data=json.dumps(MCP_BODY),
                content_type="application/json",
                headers={**self.headers, **MCP_HEADERS},
            )
        self.assertEqual(resp.status_code, 404)


@override_settings(
    # Let Django accept the Host header so our `mcp_transport_security` decorator runs
    # and does the host/origin enforcement itself (the test client sends no Host by default).
    ALLOWED_HOSTS=["*"],
    MCP_ENABLE_DNS_REBINDING_PROTECTION=True,
    MCP_ALLOWED_HOSTS=["testserver", "localhost:*"],
    MCP_ALLOWED_ORIGINS=["https://allowed.example"],
)
class MCPTransportSecurityTest(GatewayIntegrationTestCase):
    """Transport security (DNS rebinding / origin / content-type validation)."""

    def setUp(self):
        super().setUp()
        self.url = reverse("gateway:mcp_server", kwargs={"name": MCP_SERVER})
        patcher = patch("gateway.views.mcp.get_mcp_config", return_value=MCP_CONFIG)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_invalid_host_rejected(self):
        headers = {**self.headers, **MCP_HEADERS}
        resp = self.client.post(
            self.url,
            data=json.dumps(MCP_BODY),
            content_type="application/json",
            headers=headers,
            HTTP_HOST="evil.testserver:8000",
        )
        self.assertEqual(resp.status_code, 421)
        self.assertIn("Invalid Host header", resp.json()["error"]["message"])

    def test_invalid_origin_rejected(self):
        headers = {**self.headers, **MCP_HEADERS, "Origin": "https://evil.com"}
        resp = self.client.post(
            self.url,
            data=json.dumps(MCP_BODY),
            content_type="application/json",
            headers=headers,
            HTTP_HOST="testserver",
        )
        self.assertEqual(resp.status_code, 403)
        self.assertIn("Invalid Origin header", resp.json()["error"]["message"])

    def test_invalid_content_type_rejected(self):
        # `parse_body` rejects unsupported content types with 415 before the relay.
        resp = self.client.post(
            self.url,
            data="not json",
            content_type="text/plain",
            headers={**self.headers, **MCP_HEADERS, "Content-Type": "text/plain"},
        )
        self.assertEqual(resp.status_code, 415)

    def test_valid_host_and_missing_origin_allowed(self):
        with _patch_client(_json_rpc_response_handler):
            resp = self.client.post(
                self.url,
                data=json.dumps(MCP_BODY),
                content_type="application/json",
                headers={**self.headers, **MCP_HEADERS},
                HTTP_HOST="testserver",
            )
        self.assertEqual(resp.status_code, 200)


class MCPServerExclusionTest(GatewayIntegrationTestCase):
    """MCP server exclusion (org/team/user) still enforced before relay."""

    @override_settings(MCP_ENABLE_DNS_REBINDING_PROTECTION=False)
    def test_org_excluded_mcp_server(self):
        org = Org.objects.get(name="E060")
        org.add_excluded_mcp_server(MCP_SERVER)
        self.addCleanup(lambda: org.remove_excluded_mcp_server(MCP_SERVER))

        url = reverse("gateway:mcp_server", kwargs={"name": MCP_SERVER})
        resp = self.client.post(
            url,
            data=json.dumps(MCP_BODY),
            content_type="application/json",
            headers={**self.headers, **MCP_HEADERS},
        )
        self.assertEqual(resp.status_code, 404)

    @override_settings(MCP_ENABLE_DNS_REBINDING_PROTECTION=False)
    def test_mcp_server_access_allowed(self):
        url = reverse("gateway:mcp_server", kwargs={"name": MCP_SERVER})
        with (
            patch("gateway.views.mcp.get_mcp_config", return_value=MCP_CONFIG),
            _patch_client(_json_rpc_response_handler),
        ):
            resp = self.client.post(
                url,
                data=json.dumps(MCP_BODY),
                content_type="application/json",
                headers={**self.headers, **MCP_HEADERS},
            )
        self.assertEqual(resp.status_code, 200)
