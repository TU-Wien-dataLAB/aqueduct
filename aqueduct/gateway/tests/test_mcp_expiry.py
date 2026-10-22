"""Real npm upstream expiry, with only its Date.now system-clock boundary shifted."""

import base64
import gzip
import json
import os
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

from django.test import AsyncClient, TestCase, override_settings
from django.urls import reverse

from gateway.tests.utils.mcp import (
    MCP_PROTOCOL_VERSION,
    ROUTER_CONFIG_PATH,
    TASKS_EXTENSION,
    MCPEverythingServer,
    modern_params,
)


@override_settings(
    OIDC_OP_JWKS_ENDPOINT="https://example.com/application/o/example/jwks/",
    LITELLM_ROUTER_CONFIG_FILE_PATH=str(ROUTER_CONFIG_PATH),
    AUTHENTICATION_BACKENDS=["gateway.authentication.TokenAuthenticationBackend"],
    MCP_ENABLE_DNS_REBINDING_PROTECTION=False,
)
class MCPExpiryTest(TestCase):
    fixtures: ClassVar[list[str]] = ["gateway_data.json"]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Never use the suite-wide live server: advancing its clock would expire
        # other tests' handles. Cleanup is registered before any startup work.
        cls.upstream = MCPEverythingServer()
        cls.addClassCleanup(cls.upstream.stop)
        cls.offset_path = Path(cls.upstream.directory.name) / "offset-ms"
        cls.offset_path.write_text("0")
        preload = Path(cls.upstream.directory.name) / "clock.cjs"
        preload.write_text(
            "const fs = require('node:fs');\n"
            "const realNow = Date.now.bind(Date);\n"
            "Date.now = () => realNow() + Number(fs.readFileSync("
            + json.dumps(str(cls.offset_path))
            + ", 'utf8'));\n"
        )
        # Preserve the launcher's existing options; only this subprocess inherits
        # the preload. No reference package/source or gateway clock is patched.
        options = f"--require={preload} {os.environ.get('NODE_OPTIONS', '')}".strip()
        with patch.dict(os.environ, {"NODE_OPTIONS": options}):
            cls.upstream.start()

    def setUp(self):
        super().setUp()
        self.set_clock_offset(0)
        self.addCleanup(self.set_clock_offset, 0)
        self.enterContext(
            patch(
                "gateway.views.mcp.get_mcp_config",
                return_value={"test-server": {"url": self.upstream.url}},
            )
        )
        self.gateway = AsyncClient()
        self.url = reverse("gateway:mcp_server", kwargs={"name": "test-server"})

    @classmethod
    def set_clock_offset(cls, milliseconds):
        # Atomic replacement keeps Date.now from observing a partially written file.
        pending = cls.offset_path.with_suffix(".pending")
        pending.write_text(str(milliseconds))
        pending.replace(cls.offset_path)

    async def rpc(self, method, params, *, capabilities=None):
        headers = {
            "Authorization": "Bearer sk-123abc",
            "Accept": "application/json, text/event-stream",
            "Mcp-Method": method,
            "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
        }
        headers["Mcp-Name"] = params[
            "uri" if method == "resources/read" else "taskId" if method == "tasks/get" else "name"
        ]
        response = await self.gateway.post(
            self.url,
            data=json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": method,
                    "params": modern_params(params, capabilities),
                }
            ),
            content_type="application/json",
            headers=headers,
        )
        if response.streaming:
            body = b"".join([chunk async for chunk in response.streaming_content]).decode()
        else:
            body = response.content.decode()
        self.assertEqual(response.status_code, 200, body)
        if "text/event-stream" in response.headers.get("Content-Type", ""):
            messages = [
                json.loads(line[5:]) for line in body.splitlines() if line.startswith("data:")
            ]
            finals = [message for message in messages if message.get("id") == 1]
            self.assertEqual(len(finals), 1, body)
            payload = finals[0]
        else:
            payload = json.loads(body)
        self.assertEqual(payload["id"], 1)
        return payload

    async def test_signed_request_state_is_valid_then_expires(self):
        capabilities = {"elicitation": {"form": {}}}
        call = {"name": "trigger-multi-step-elicitation", "arguments": {}}
        payload = await self.rpc("tools/call", call, capabilities=capabilities)
        first = payload["result"]
        self.assertEqual(first["resultType"], "input_required")
        self.assertIn("name", first["inputRequests"])
        state = first["requestState"]
        self.assertIsInstance(state, str)
        self.assertTrue(state)
        continuation = {
            **call,
            "requestState": state,
            "inputResponses": {"name": {"action": "accept", "content": {"name": "Ada"}}},
        }
        payload = await self.rpc("tools/call", continuation, capabilities=capabilities)
        second = payload["result"]
        self.assertEqual(second["resultType"], "input_required")
        self.assertEqual(
            second["inputRequests"]["color"]["params"]["message"],
            "Thanks, Ada. Which color do you like best?",
        )

        # Replay the identical, genuinely signed state and answer, not a forged
        # token. The process-wide signing key and all request fields stay intact.
        self.set_clock_offset(601_000)
        payload = await self.rpc("tools/call", continuation, capabilities=capabilities)
        self.assertEqual(payload["error"]["code"], -32602)
        self.assertNotIn("result", payload)

    async def test_minted_gzip_resource_is_readable_then_expires(self):
        payload = await self.rpc(
            "tools/call",
            {
                "name": "gzip-file-as-resource",
                "arguments": {
                    "name": "expiry.txt.gz",
                    "data": "data:text/plain;base64,SGVsbG8sIE1DUCE=",
                    "outputType": "resourceLink",
                },
            },
        )
        result = payload["result"]
        self.assertFalse(result.get("isError", False))
        link = result["content"][0]
        self.assertEqual(link["type"], "resource_link")
        uri = link["uri"]
        self.assertTrue(uri.startswith("demo://resource/minted/"))
        payload = await self.rpc("resources/read", {"uri": uri})
        resource = payload["result"]["contents"][0]
        self.assertEqual(resource["uri"], uri)
        self.assertEqual(gzip.decompress(base64.b64decode(resource["blob"])), b"Hello, MCP!")

        self.set_clock_offset(601_000)
        payload = await self.rpc("resources/read", {"uri": uri})
        self.assertEqual(payload["error"]["code"], -32602)
        self.assertEqual(payload["error"]["data"]["uri"], uri)
        self.assertNotIn("result", payload)

    async def test_task_is_valid_then_expires_after_its_ttl(self):
        capabilities = {"extensions": {TASKS_EXTENSION: {}}}
        payload = await self.rpc(
            "tools/call",
            {"name": "simulate-research-query", "arguments": {"topic": "Expiry coverage"}},
            capabilities=capabilities,
        )
        task = payload["result"]
        self.assertEqual(task["resultType"], "task")
        task_id = task["taskId"]
        ttl = task["ttlMs"]
        self.assertEqual(ttl, 300_000)
        payload = await self.rpc("tasks/get", {"taskId": task_id}, capabilities=capabilities)
        self.assertEqual(payload["result"]["taskId"], task_id)
        self.assertEqual(payload["result"]["ttlMs"], ttl)
        self.assertIn(payload["result"]["status"], ("working", "completed"))

        self.set_clock_offset(ttl + 1_000)
        payload = await self.rpc("tasks/get", {"taskId": task_id}, capabilities=capabilities)
        self.assertEqual(payload["error"]["code"], -32602)
        self.assertIn("Unknown task", payload["error"]["message"])
        self.assertNotIn("result", payload)
