"""Mock relay regressions and live 2026-07-28 server-everything integration tests."""

import asyncio
import base64
import gzip
import json
from unittest.mock import patch

import httpx
from asgiref.sync import async_to_sync, sync_to_async
from django.test import override_settings
from django.urls import reverse
from mcp import MCPError
from mcp.client.subscriptions import (
    PromptsListChanged,
    ResourcesListChanged,
    ResourceUpdated,
    ToolsListChanged,
)
from mcp.types import (
    ElicitResult,
    PromptReference,
    ResourceTemplateReference,
    jsonrpc_message_adapter,
)

from gateway.tests.utils.base import GatewayIntegrationTestCase
from gateway.tests.utils.mcp import TASKS_EXTENSION, MCPLiveServerTestCase
from management.models import Org, ServiceAccount, Team, Token, UserProfile

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
                    "parameter_headers": {
                        key: value
                        for key, value in request.headers.items()
                        if key.startswith("mcp-param-")
                    },
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


class MCPLiveClientTest(MCPLiveServerTestCase):
    async def test_discovery_and_stateless_tool_call(self):
        async with self.mcp_client() as client:
            self.assertEqual(client.protocol_version, "2026-07-28")
            self.assertEqual(client.server_info.name, "mcp-servers/everything")
            self.assertEqual(client.server_info.version, "2.0.0")
            result = await client.call_tool("echo", {"message": "through the live gateway"})
            self.assertFalse(result.is_error)
            self.assertEqual(result.content[0].text, "Echo: through the live gateway")

    async def test_discovery_advertises_modern_capabilities_and_cache_hints(self):
        response, payload = await self.rpc("server/discover")
        self.assertEqual(response.status_code, 200)
        result = payload["result"]
        self.assertIn("2026-07-28", result["supportedVersions"])
        for capability in ("tools", "resources", "prompts", "completions"):
            self.assertIn(capability, result["capabilities"])
        self.assertIn(TASKS_EXTENSION, result["capabilities"]["extensions"])
        self.assertEqual(result["ttlMs"], 300000)
        self.assertEqual(result["cacheScope"], "public")
        self.assertTrue(result["instructions"])
        self.assertNotIn("mcp-session-id", response.headers)
        await self.assert_requests_logged(1)

    async def test_requests_work_without_discovery_or_shared_session(self):
        for message in ("first", "second"):
            response, payload = await self.rpc(
                "tools/call", {"name": "echo", "arguments": {"message": message}}
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(payload["result"]["content"][0]["text"], f"Echo: {message}")
            self.assertNotIn("mcp-session-id", response.headers)
        await self.assert_requests_logged(2)

    async def test_lists_include_tools_and_json_schema_header_annotations(self):
        async with self.mcp_client() as client:
            tools = await client.list_tools()
            names = {tool.name for tool in tools.tools}
            # Exact v2.0.0 modern catalog: notice additions when upgrading the fixture.
            self.assertEqual(
                names,
                {
                    "echo",
                    "get-annotated-message",
                    "get-env",
                    "get-header-params",
                    "get-resource-links",
                    "get-resource-reference",
                    "get-roots-list",
                    "get-structured-content",
                    "get-sum",
                    "get-tiny-image",
                    "gzip-file-as-resource",
                    "simulate-research-query",
                    "trigger-elicitation-request",
                    "trigger-elicitation-request-async",
                    "trigger-list-changed",
                    "trigger-log-messages",
                    "trigger-long-running-operation",
                    "trigger-multi-step-elicitation",
                    "trigger-resource-update",
                    "trigger-sampling-request",
                    "trigger-sampling-request-async",
                    "trigger-url-elicitation",
                },
            )
            header_tool = next(tool for tool in tools.tools if tool.name == "get-header-params")
            self.assertEqual(
                header_tool.input_schema["$schema"], "https://json-schema.org/draft/2020-12/schema"
            )
            self.assertEqual(
                header_tool.input_schema["properties"]["region"]["x-mcp-header"], "Region"
            )
            self.assertEqual(tools.ttl_ms, 60000)
            self.assertEqual(tools.cache_scope, "public")

    async def test_sum_and_structured_tool_results(self):
        async with self.mcp_client() as client:
            result = await client.call_tool("get-sum", {"a": 2, "b": 3})
            self.assertFalse(result.is_error)
            self.assertEqual(result.content[0].text, "The sum of 2 and 3 is 5.")
            weather = await client.call_tool("get-structured-content", {"location": "New York"})
            self.assertFalse(weather.is_error)
            self.assertEqual(
                weather.structured_content,
                {"temperature": 33, "conditions": "Cloudy", "humidity": 82},
            )
            self.assertEqual(json.loads(weather.content[0].text), weather.structured_content)

    async def test_image_and_resource_link_tool_content(self):
        async with self.mcp_client() as client:
            image = await client.call_tool("get-tiny-image")
            image_block = next(block for block in image.content if block.type == "image")
            self.assertEqual(image_block.mime_type, "image/png")
            self.assertTrue(base64.b64decode(image_block.data).startswith(b"\x89PNG\r\n\x1a\n"))
            links = await client.call_tool("get-resource-links", {"count": 2})
            resources = [block for block in links.content if block.type == "resource_link"]
            self.assertEqual(len(resources), 2)
            for resource in resources:
                read = await client.read_resource(str(resource.uri))
                self.assertEqual(str(read.contents[0].uri), str(resource.uri))

    async def test_header_parameter_plain_and_encoded_values(self):
        async with self.mcp_client() as client:
            # The SDK must see x-mcp-header annotations before it can mirror arguments.
            await client.list_tools()
            for region, header in (
                ("eu-west-1", "eu-west-1"),
                ("Hello, 世界", "=?base64?SGVsbG8sIOS4lueVjA==?="),
            ):
                with self.subTest(region=region):
                    result = await client.call_tool(
                        "get-header-params", {"region": region, "repeat": 2}
                    )
                    self.assertFalse(result.is_error)
                    self.assertEqual(
                        result.structured_content,
                        {"region": region, "regionHeader": header, "repeated": [region, region]},
                    )

    async def test_resources_and_static_document_cache_hints(self):
        async with self.mcp_client() as client:
            resources = await client.list_resources()
            self.assertGreater(len(resources.resources), 0)
            resource = next(
                resource
                for resource in resources.resources
                if str(resource.uri).endswith("/instructions.md")
            )
            result = await client.read_resource(str(resource.uri))
            self.assertIn("Everything", result.contents[0].text)
            self.assertEqual(str(result.contents[0].uri), str(resource.uri))
            self.assertEqual(result.ttl_ms, 3600000)
            self.assertEqual(result.cache_scope, "public")

    async def test_dynamic_text_and_blob_resource_templates(self):
        async with self.mcp_client() as client:
            templates = await client.list_resource_templates()
            uris = {template.uri_template for template in templates.resource_templates}
            self.assertIn("demo://resource/dynamic/text/{resourceId}", uris)
            self.assertIn("demo://resource/dynamic/blob/{resourceId}", uris)
            text = await client.read_resource("demo://resource/dynamic/text/7")
            self.assertTrue(
                text.contents[0].text.startswith("Resource 7: This is a plaintext resource")
            )
            blob = await client.read_resource("demo://resource/dynamic/blob/8")
            self.assertEqual(blob.contents[0].mime_type, "application/octet-stream")
            self.assertTrue(base64.b64decode(blob.contents[0].blob).startswith(b"Resource 8:"))

    async def test_prompts_with_arguments_and_embedded_resources(self):
        async with self.mcp_client() as client:
            prompts = await client.list_prompts()
            self.assertEqual(
                {prompt.name for prompt in prompts.prompts},
                {"simple-prompt", "args-prompt", "completable-prompt", "resource-prompt"},
            )
            simple = await client.get_prompt("simple-prompt")
            self.assertEqual(simple.messages[0].role, "user")
            self.assertEqual(
                simple.messages[0].content.text, "This is a simple prompt without arguments."
            )
            args = await client.get_prompt("args-prompt", {"city": "Boston", "state": "MA"})
            self.assertEqual(args.messages[0].content.text, "What's weather in Boston, MA?")
            embedded = await client.get_prompt(
                "resource-prompt", {"resourceType": "Text", "resourceId": "9"}
            )
            resource = next(
                message.content
                for message in embedded.messages
                if message.content.type == "resource"
            )
            self.assertEqual(str(resource.resource.uri), "demo://resource/dynamic/text/9")
            self.assertIn("Resource 9:", resource.resource.text)

    async def test_prompt_completions_use_argument_context(self):
        async with self.mcp_client() as client:
            ref = PromptReference(type="ref/prompt", name="completable-prompt")
            departments = await client.complete(ref, {"name": "department", "value": "E"})
            self.assertEqual(departments.completion.values, ["Engineering"])
            names = await client.complete(
                ref, {"name": "name", "value": "A"}, {"department": "Engineering"}
            )
            self.assertEqual(names.completion.values, ["Alice"])
            no_names = await client.complete(
                ref, {"name": "name", "value": "A"}, {"department": "Sales"}
            )
            self.assertEqual(no_names.completion.values, [])

    async def test_resource_template_completions(self):
        async with self.mcp_client() as client:
            ref = ResourceTemplateReference(
                type="ref/resource", uri="demo://resource/dynamic/text/{resourceId}"
            )
            for value, expected in (("12", ["12"]), ("0", []), ("not-an-id", [])):
                with self.subTest(value=value):
                    result = await client.complete(ref, {"name": "resourceId", "value": value})
                    self.assertEqual(result.completion.values, expected)

    async def test_reference_server_lists_are_terminal_with_or_without_cursor(self):
        # v2.0.0's McpServer lists are unpaginated. Exercise the cursor parameter
        # explicitly, not an `if nextCursor` branch that silently tests nothing.
        async with self.mcp_client() as client:
            for method, field in (
                (client.list_tools, "tools"),
                (client.list_resources, "resources"),
                (client.list_resource_templates, "resource_templates"),
                (client.list_prompts, "prompts"),
            ):
                with self.subTest(method=method.__name__):
                    first = await method()
                    page = await method(cursor="opaque-client-cursor")
                    self.assertIsNone(first.next_cursor)
                    self.assertIsNone(page.next_cursor)
                    self.assertGreater(len(getattr(first, field)), 0)
                    self.assertEqual(getattr(page, field), getattr(first, field))

    async def test_progress_notifications_precede_final_result(self):
        updates = []

        async def progress(progress, total, message):
            updates.append((progress, total, message))

        async with self.mcp_client() as client:
            result = await client.call_tool(
                "trigger-long-running-operation",
                {"duration": 0.12, "steps": 3},
                progress_callback=progress,
                read_timeout_seconds=5,
            )
            self.assertFalse(result.is_error)
            self.assertEqual(updates, [(1, 3, None), (2, 3, None), (3, 3, None)])
            self.assertEqual(
                result.content[0].text,
                "Long running operation completed. Duration: 0.12 seconds, Steps: 3.",
            )

    async def test_per_request_logging_filters_and_does_not_leak(self):
        messages = []

        async def log_message(params):
            messages.append(params)

        async with self.mcp_client(logging_callback=log_message) as client:
            await client.call_tool(
                "trigger-log-messages", meta={"io.modelcontextprotocol/logLevel": "warning"}
            )
            self.assertEqual(
                [message.level for message in messages],
                ["warning", "error", "critical", "alert", "emergency"],
            )
            messages.clear()
            await client.call_tool("trigger-log-messages")
            self.assertEqual(messages, [])

    async def test_multi_round_trip_elicitation_carries_request_state(self):
        questions = []

        async def elicit(context, params):
            questions.append(params.message)
            content = {"name": "Ada"} if len(questions) == 1 else {"color": "blue"}
            return ElicitResult(action="accept", content=content)

        async with self.mcp_client(elicitation_callback=elicit) as client:
            result = await client.call_tool("trigger-multi-step-elicitation")
            self.assertEqual(
                questions, ["What is your name?", "Thanks, Ada. Which color do you like best?"]
            )
            self.assertFalse(result.is_error)
            self.assertEqual(result.structured_content, {"name": "Ada", "color": "blue"})
            self.assertEqual(result.content[0].text, "Ada's favorite color is blue.")

    async def test_multi_round_trip_rejects_tampered_request_state(self):
        capabilities = {"elicitation": {"form": {}}}
        response, payload = await self.rpc(
            "tools/call",
            {"name": "trigger-multi-step-elicitation", "arguments": {}},
            capabilities=capabilities,
        )
        self.assertEqual(response.status_code, 200)
        result = payload["result"]
        self.assertEqual(result["resultType"], "input_required")
        self.assertIn("name", result["inputRequests"])
        state = result["requestState"]
        response, rejected = await self.rpc(
            "tools/call",
            {
                "name": "trigger-multi-step-elicitation",
                "arguments": {},
                "requestState": state + "tampered",
                "inputResponses": {"name": {"action": "accept", "content": {"name": "Ada"}}},
            },
            capabilities=capabilities,
        )
        self.assertEqual(rejected["error"]["code"], -32602)

    async def test_missing_client_capability_is_a_protocol_error(self):
        _response, payload = await self.rpc(
            "tools/call", {"name": "trigger-multi-step-elicitation", "arguments": {}}
        )
        self.assertEqual(payload["error"]["code"], -32021)
        self.assertIn("elicitation", payload["error"]["data"]["requiredCapabilities"])

    async def test_subscription_list_changes_and_resource_updates(self):
        uri = "demo://resource/dynamic/text/1"
        async with (
            self.mcp_client() as client,
            client.listen(
                tools_list_changed=True,
                prompts_list_changed=True,
                resources_list_changed=True,
                resource_subscriptions=[uri],
            ) as subscription,
        ):
            self.assertTrue(subscription.honored.tools_list_changed)
            self.assertEqual(subscription.honored.resource_subscriptions, [uri])
            for name, event_type in (
                ("tools", ToolsListChanged),
                ("prompts", PromptsListChanged),
                ("resources", ResourcesListChanged),
            ):
                await client.call_tool("trigger-list-changed", {"list": name})
                event = await asyncio.wait_for(anext(subscription), timeout=5)
                self.assertIsInstance(event, event_type)
            await client.call_tool("trigger-resource-update", {"uri": uri})
            event = await asyncio.wait_for(anext(subscription), timeout=5)
            self.assertIsInstance(event, ResourceUpdated)
            self.assertEqual(str(event.uri), uri)

    async def test_tasks_extension_create_poll_and_cancel(self):
        capabilities = {"extensions": {TASKS_EXTENSION: {}}}
        response, payload = await self.rpc(
            "tools/call",
            {"name": "simulate-research-query", "arguments": {"topic": "MCP"}},
            capabilities=capabilities,
        )
        self.assertEqual(response.status_code, 200)
        task = payload["result"]
        self.assertEqual(task["resultType"], "task")
        self.assertEqual(task["status"], "working")
        task_id = task["taskId"]
        response, payload = await self.rpc(
            "tasks/get", {"taskId": task_id}, capabilities=capabilities
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(payload["result"]["taskId"], task_id)
        self.assertEqual(payload["result"]["status"], "working")
        response, payload = await self.rpc(
            "tasks/cancel", {"taskId": task_id}, capabilities=capabilities
        )
        self.assertEqual(response.status_code, 200)
        response, payload = await self.rpc(
            "tasks/get", {"taskId": task_id}, capabilities=capabilities
        )
        self.assertEqual(payload["result"]["status"], "cancelled")

    async def test_tasks_extension_input_update_completes_task(self):
        capabilities = {"extensions": {TASKS_EXTENSION: {}}, "elicitation": {"form": {}}}
        response, payload = await self.rpc(
            "tools/call",
            {"name": "trigger-elicitation-request-async", "arguments": {}},
            capabilities=capabilities,
        )
        self.assertEqual(response.status_code, 200)
        task = payload["result"]
        self.assertEqual(task["resultType"], "task")
        self.assertEqual(task["status"], "input_required")
        task_id = task["taskId"]
        key, request = next(iter(task["inputRequests"].items()))
        self.assertEqual(request["method"], "elicitation/create")
        response, payload = await self.rpc(
            "tasks/update",
            {
                "taskId": task_id,
                "inputResponses": {key: {"action": "accept", "content": {"name": "Ada"}}},
            },
            capabilities=capabilities,
        )
        self.assertEqual(response.status_code, 200, response.text)
        response, payload = await self.rpc(
            "tasks/get", {"taskId": task_id}, capabilities=capabilities
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["result"]["status"], "completed")
        self.assertIn("Name: Ada", payload["result"]["result"]["content"][0]["text"])

    async def test_minted_resource_can_be_read_by_an_independent_client(self):
        async with self.mcp_client() as client:
            result = await client.call_tool(
                "gzip-file-as-resource",
                {"name": "greeting.gz", "data": "data:text/plain;base64,SGVsbG8sIE1DUCE="},
            )
            self.assertFalse(result.is_error)
            link = next(block for block in result.content if block.type == "resource_link")
            self.assertTrue(str(link.uri).startswith("demo://resource/minted/"))
        async with self.mcp_client() as other_client:
            resource = await other_client.read_resource(str(link.uri))
            self.assertEqual(resource.contents[0].mime_type, "application/gzip")
            self.assertEqual(
                gzip.decompress(base64.b64decode(resource.contents[0].blob)), b"Hello, MCP!"
            )

    async def test_errors_for_unknown_tools_resources_prompts_and_invalid_arguments(self):
        async with self.mcp_client() as client:
            with self.assertRaises(MCPError) as error:
                await client.call_tool("nonexistent-tool")
            self.assertEqual(error.exception.code, -32602)
            self.assertIn("nonexistent-tool", error.exception.message)
            with self.assertRaises(MCPError) as error:
                await client.read_resource("demo://resource/dynamic/text/0")
            self.assertEqual(error.exception.code, -32602)
            self.assertIn("demo://resource/dynamic/text/0", str(error.exception.data))
            with self.assertRaises(MCPError) as error:
                await client.get_prompt("nonexistent-prompt")
            self.assertEqual(error.exception.code, -32602)
            with self.assertRaises(MCPError) as error:
                await client.get_prompt("args-prompt")
            self.assertEqual(error.exception.code, -32602)
            invalid = await client.call_tool("get-sum", {"a": "not a number", "b": 1})
            self.assertTrue(invalid.is_error)
            self.assertIn("a", invalid.content[0].text)

    async def test_header_mismatches_and_unknown_method_errors_pass_through(self):
        for headers in ({"Mcp-Name": "wrong"}, {"Mcp-Method": "tools/list"}):
            with self.subTest(headers=headers):
                response, payload = await self.rpc(
                    "tools/call",
                    {"name": "echo", "arguments": {"message": "test"}},
                    headers=headers,
                )
                self.assertEqual(response.status_code, 400)
                self.assertEqual(payload["error"]["code"], -32020)
        response, payload = await self.rpc("not/a-method")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(payload["error"]["code"], -32601)


class MCPLivePolicyTest(MCPLiveServerTestCase):
    async def test_authentication_is_required(self):
        response, _payload = await self.rpc("server/discover", headers={"Authorization": ""})
        self.assertEqual(response.status_code, 401)

    async def test_host_and_origin_security_over_real_http(self):
        for headers, status, text in (
            ({"Host": "evil.testserver:8000"}, 421, "Invalid Host header"),
            ({"Origin": "https://evil.example"}, 403, "Invalid Origin header"),
        ):
            with self.subTest(headers=headers):
                response, payload = await self.rpc("server/discover", headers=headers)
                self.assertEqual(response.status_code, status)
                self.assertIn(text, payload["error"]["message"])
        await self.assert_requests_logged(0)
        response, payload = await self.rpc(
            "server/discover", headers={"Origin": self.live_server_url}
        )
        self.assertEqual(response.status_code, 200)

    async def test_get_and_delete_are_not_supported(self):
        async with httpx.AsyncClient(headers=self.headers, timeout=5, trust_env=False) as client:
            for method in ("GET", "DELETE"):
                response = await client.request(method, self.mcp_url)
                self.assertEqual(response.status_code, 405)
        await self.assert_requests_logged(0)

    async def test_org_exclusion_is_enforced(self):
        org = await Org.objects.aget(name="E060")
        await sync_to_async(org.add_excluded_mcp_server)("test-server")
        response, _payload = await self.rpc("server/discover")
        self.assertEqual(response.status_code, 404)
        await self.assert_requests_logged(0)

    async def test_team_exclusion_is_enforced(self):
        team = await Team.objects.aget(name="Whale")
        await sync_to_async(team.add_excluded_mcp_server)("test-server")
        account = await ServiceAccount.objects.acreate(team=team, name="MCP test service account")
        token = await Token.objects.aget(key_hash=Token._hash_key("sk-123abc"))
        token.service_account = account
        await token.asave()
        response, _payload = await self.rpc("server/discover")
        self.assertEqual(response.status_code, 404)
        await self.assert_requests_logged(0)

    async def test_user_exclusion_is_enforced(self):
        profile = await UserProfile.objects.aget(user__username="Me")
        await sync_to_async(profile.add_excluded_mcp_server)("test-server")
        response, _payload = await self.rpc("server/discover")
        self.assertEqual(response.status_code, 404)
        await self.assert_requests_logged(0)

    async def test_user_can_disable_inherited_exclusions(self):
        org = await Org.objects.aget(name="E060")
        await sync_to_async(org.add_excluded_mcp_server)("test-server")
        profile = await UserProfile.objects.aget(user__username="Me")
        profile.merge_mcp_server_exclusion_lists = False
        await profile.asave()
        response, _payload = await self.rpc("server/discover")
        self.assertEqual(response.status_code, 200)
        await self.assert_requests_logged(1)


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

    def test_incoming_protocol_version_is_preserved(self):
        with _patch_client(_json_rpc_response_handler):
            resp = self._post(headers={**MCP_HEADERS, "MCP-Protocol-Version": "2099-01-01"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["result"]["relayed"]["protocol_version"], "2099-01-01")

    def test_custom_parameter_headers_are_forwarded_unchanged(self):
        parameters = {
            "Mcp-Param-Region": "eu-west-1",
            "mCp-PaRaM-Greeting": "=?base64?SGVsbG8sIOS4lueVjA==?=",
        }
        with _patch_client(_json_rpc_response_handler):
            resp = self._post(headers={**MCP_HEADERS, **parameters})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(
            resp.json()["result"]["relayed"]["parameter_headers"],
            {
                "mcp-param-region": "eu-west-1",
                "mcp-param-greeting": "=?base64?SGVsbG8sIOS4lueVjA==?=",
            },
        )

    def test_resource_read_derives_name_from_uri(self):
        body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "resources/read",
            "params": {"uri": "demo://resource/dynamic/text/1"},
        }
        with _patch_client(_json_rpc_response_handler):
            resp = self._post(body, headers={"Mcp-Method": "resources/read"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["result"]["relayed"]["name"], "demo://resource/dynamic/text/1")

    def test_only_standard_name_bearing_methods_derive_name(self):
        for method in (
            "tools/get",
            "resources/templates/list",
            "completion/complete",
            "tasks/get",
            "tasks/list",
            "tools/list",
            "server/discover",
        ):
            with self.subTest(method=method), _patch_client(_json_rpc_response_handler):
                body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": {"name": "unused"}}
                resp = self._post(body, headers={"Mcp-Method": method})
                self.assertEqual(resp.status_code, 200)
                self.assertIsNone(resp.json()["result"]["relayed"]["name"])

    def test_paginated_list_preserves_opaque_cursor_and_result(self):
        cursors = []

        def upstream(request):
            cursor = json.loads(request.content)["params"].get("cursor")
            cursors.append(cursor)
            result = {"tools": [{"name": "first"}], "nextCursor": "opaque+/=cursor"}
            if cursor == "opaque+/=cursor":
                result = {"tools": [{"name": "second"}]}
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": result})

        with _patch_client(upstream):
            body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
            first = self._post(body, headers={"Mcp-Method": "tools/list"})
            self.assertEqual(first.status_code, 200)
            body["params"]["cursor"] = first.json()["result"]["nextCursor"]
            second = self._post(body, headers={"Mcp-Method": "tools/list"})
        self.assertEqual(second.status_code, 200)
        self.assertEqual(cursors, [None, "opaque+/=cursor"])
        self.assertEqual(first.json()["result"]["tools"], [{"name": "first"}])
        self.assertEqual(second.json()["result"], {"tools": [{"name": "second"}]})

    def test_stateless_requests_are_independent(self):
        # Two identical calls produce identical results (no shared session state).
        with _patch_client(_json_rpc_response_handler):
            r1 = self._post(headers=MCP_HEADERS)
            r2 = self._post(headers=MCP_HEADERS)
        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r1.json(), r2.json())

    def test_relay_streams_sse_response(self):
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
