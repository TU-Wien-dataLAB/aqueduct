"""Additional public-API edges of the real server-everything v2.0.0 upstream."""

import asyncio
import base64
import gzip
import json

import httpx
from mcp.client.subscriptions import ResourceUpdated
from mcp.types import ElicitResult

from gateway.tests.utils.mcp import (
    MCP_PROTOCOL_VERSION,
    TASKS_EXTENSION,
    MCPLiveServerTestCase,
    modern_params,
)

TASK_CAPABILITIES = {"extensions": {TASKS_EXTENSION: {}}}
FORM_CAPABILITIES = {**TASK_CAPABILITIES, "elicitation": {"form": {}}}


class MCPLiveEdgesTest(MCPLiveServerTestCase):
    async def test_subscription_filters_exclude_unrequested_lists_and_resource_uris(self):
        uri = "demo://resource/dynamic/text/1"
        notifications = {"toolsListChanged": True, "resourceSubscriptions": [uri]}
        # Read raw frames: the SDK itself filters list events, which could hide
        # an upstream filter regression. The acknowledged stream is our barrier.
        async with (
            self.mcp_client() as client,
            httpx.AsyncClient(timeout=5, trust_env=False) as transport,
            transport.stream(
                "POST",
                self.mcp_url,
                headers={
                    **self.headers,
                    "Accept": "application/json, text/event-stream",
                    "Mcp-Method": "subscriptions/listen",
                    "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
                },
                json={
                    "jsonrpc": "2.0",
                    "id": "edge-filter",
                    "method": "subscriptions/listen",
                    "params": modern_params({"notifications": notifications}),
                },
            ) as response,
        ):
            self.assertEqual(response.status_code, 200)
            lines = response.aiter_lines()

            async def next_frame():
                async with asyncio.timeout(5):
                    async for line in lines:
                        if line.startswith("data:") and line[5:].strip():
                            return json.loads(line[5:])
                raise AssertionError("Listen stream ended before the expected frame")

            ack = await next_frame()
            self.assertEqual(ack["method"], "notifications/subscriptions/acknowledged")
            self.assertEqual(ack["params"]["notifications"], notifications)
            for name in ("prompts", "resources"):
                result = await client.call_tool("trigger-list-changed", {"list": name})
                self.assertFalse(result.is_error)
            result = await client.call_tool(
                "trigger-resource-update", {"uri": "demo://resource/dynamic/text/2"}
            )
            self.assertFalse(result.is_error)
            # A selected event, published after all excluded events, is a
            # deterministic fence on this same ordered SSE stream. No quiet sleep.
            await client.call_tool("trigger-list-changed", {"list": "tools"})
            frame = await next_frame()
            self.assertEqual(frame["method"], "notifications/tools/list_changed")
            await client.call_tool("trigger-resource-update", {"uri": uri})
            frame = await next_frame()
            self.assertEqual(frame["method"], "notifications/resources/updated")
            self.assertEqual(frame["params"]["uri"], uri)

    async def test_concurrent_clients_are_independent_and_one_can_close(self):
        first_uri = "demo://resource/dynamic/text/11"
        second_uri = "demo://resource/dynamic/text/12"
        async with (
            self.mcp_client() as survivor,
            survivor.listen(resource_subscriptions=[second_uri]) as surviving_subscription,
        ):
            async with (
                self.mcp_client() as first,
                first.listen(resource_subscriptions=[first_uri]) as first_subscription,
            ):
                # Both listen contexts have received their acknowledgments. Start
                # real overlapping requests, without relying on scheduling sleeps.
                async with asyncio.timeout(10):
                    results = await asyncio.gather(
                        first.call_tool("echo", {"message": "first client"}),
                        survivor.call_tool("echo", {"message": "second client"}),
                    )
                self.assertEqual(
                    [result.content[0].text for result in results],
                    ["Echo: first client", "Echo: second client"],
                )
                await first.call_tool("trigger-resource-update", {"uri": first_uri})
                await survivor.call_tool("trigger-resource-update", {"uri": second_uri})
                async with asyncio.timeout(5):
                    first_event, second_event = await asyncio.gather(
                        anext(first_subscription), anext(surviving_subscription)
                    )
                self.assertIsInstance(first_event, ResourceUpdated)
                self.assertEqual(str(first_event.uri), first_uri)
                self.assertIsInstance(second_event, ResourceUpdated)
                self.assertEqual(str(second_event.uri), second_uri)
            # Closing both the other subscription and its client must not end
            # this client's existing stream or its ability to make new calls.
            result = await survivor.call_tool("echo", {"message": "still healthy"})
            self.assertEqual(result.content[0].text, "Echo: still healthy")
            await survivor.call_tool("trigger-resource-update", {"uri": second_uri})
            event = await asyncio.wait_for(anext(surviving_subscription), timeout=5)
            self.assertIsInstance(event, ResourceUpdated)
            self.assertEqual(str(event.uri), second_uri)

    async def test_gzip_inline_resource_contains_decompressible_output(self):
        async with self.mcp_client() as client:
            result = await client.call_tool(
                "gzip-file-as-resource",
                {
                    "name": "inline.gz",
                    "data": "data:text/plain;base64,SGVsbG8sIE1DUCE=",
                    "outputType": "resource",
                },
            )
            self.assertFalse(result.is_error)
            self.assertEqual(len(result.content), 1)
            self.assertEqual(result.content[0].type, "resource")
            resource = result.content[0].resource
            self.assertEqual(resource.mime_type, "application/gzip")
            self.assertEqual(gzip.decompress(base64.b64decode(resource.blob)), b"Hello, MCP!")

    async def test_gzip_rejects_invalid_payloads_without_external_fetches(self):
        async with self.mcp_client() as client:
            for data in ("not a URL", "ftp://example.com/file", "data:text/plain;base64,%%%"):
                with self.subTest(data=data):
                    result = await client.call_tool("gzip-file-as-resource", {"data": data})
                    self.assertTrue(result.is_error)
                    self.assertTrue(result.content[0].text)
                    self.assertFalse(any(block.type == "resource_link" for block in result.content))

    async def test_blob_resource_prompt_and_completable_prompt_get(self):
        async with self.mcp_client() as client:
            prompt = await client.get_prompt(
                "resource-prompt", {"resourceType": "Blob", "resourceId": "8"}
            )
            embedded = next(
                message.content.resource
                for message in prompt.messages
                if message.content.type == "resource"
            )
            self.assertEqual(str(embedded.uri), "demo://resource/dynamic/blob/8")
            self.assertEqual(embedded.mime_type, "application/octet-stream")
            self.assertTrue(base64.b64decode(embedded.blob).startswith(b"Resource 8:"))
            prompt = await client.get_prompt(
                "completable-prompt", {"department": "Engineering", "name": "Alice"}
            )
            self.assertEqual(prompt.messages[0].role, "user")
            self.assertEqual(
                prompt.messages[0].content.text,
                "Please promote Alice to the head of the Engineering team.",
            )

    async def test_multi_round_elicitation_retries_blank_name_and_invalid_color(self):
        questions = []
        answers = iter(({"name": "  "}, {"name": "Ada"}, {"color": "mauve"}, {"color": "red"}))

        async def elicit(context, params):
            questions.append(params.message)
            return ElicitResult(action="accept", content=next(answers))

        async with self.mcp_client(elicitation_callback=elicit) as client:
            async with asyncio.timeout(10):
                result = await client.call_tool("trigger-multi-step-elicitation")
            self.assertFalse(result.is_error)
            self.assertEqual(
                questions,
                [
                    "What is your name?",
                    "A name is required. What is your name?",
                    "Thanks, Ada. Which color do you like best?",
                    "Thanks, Ada. Which color do you like best?",
                ],
            )
            self.assertEqual(result.structured_content, {"name": "Ada", "color": "red"})

    async def test_multi_round_elicitation_rejects_url_only_capability(self):
        _response, payload = await self.rpc(
            "tools/call",
            {"name": "trigger-multi-step-elicitation", "arguments": {}},
            capabilities={"elicitation": {"url": {}}},
        )
        self.assertEqual(payload["error"]["code"], -32021)
        self.assertEqual(
            payload["error"]["data"]["requiredCapabilities"], {"elicitation": {"form": {}}}
        )

    async def test_research_task_completes_and_cancellation_preserves_finished_result(self):
        _response, payload = await self.rpc(
            "tools/call",
            {"name": "simulate-research-query", "arguments": {"topic": "MCP edge coverage"}},
            capabilities=TASK_CAPABILITIES,
        )
        task = payload["result"]
        self.assertEqual(task["resultType"], "task")
        self.assertEqual(task["status"], "working")
        task_id = task["taskId"]
        try:
            # Upstream's four one-second stages are real; poll with a deadline,
            # never advance its clock or assume a particular intermediate stage.
            async with asyncio.timeout(10):
                while task["status"] != "completed":
                    await asyncio.sleep(0.2)
                    response, payload = await self.rpc(
                        "tasks/get", {"taskId": task_id}, capabilities=TASK_CAPABILITIES
                    )
                    self.assertEqual(response.status_code, 200)
                    task = payload["result"]
                    self.assertIn(task["status"], ("working", "completed"))
            self.assertEqual(task["resultType"], "complete")
            report = task["result"]["content"][0]["text"]
            self.assertTrue(report.startswith("# Research Report: MCP edge coverage\n"))
            self.assertIn("Stage 4: Generating report ✓", report)
            response, payload = await self.rpc(
                "tasks/cancel", {"taskId": task_id}, capabilities=TASK_CAPABILITIES
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(payload["result"]["resultType"], "complete")
            _response, payload = await self.rpc(
                "tasks/get", {"taskId": task_id}, capabilities=TASK_CAPABILITIES
            )
            self.assertEqual(payload["result"], task)
        finally:
            await self.rpc("tasks/cancel", {"taskId": task_id}, capabilities=TASK_CAPABILITIES)

    async def test_unknown_task_handles_and_missing_extension(self):
        for method, extra in (
            ("tasks/get", {}),
            ("tasks/update", {"inputResponses": {}}),
            ("tasks/cancel", {}),
        ):
            for capabilities, code in ((TASK_CAPABILITIES, -32602), ({}, -32021)):
                with self.subTest(method=method, capabilities=capabilities):
                    _response, payload = await self.rpc(
                        method,
                        {"taskId": "nonexistent-edge-task", **extra},
                        capabilities=capabilities,
                    )
                    self.assertEqual(payload["error"]["code"], code)
                    if not capabilities:
                        self.assertEqual(
                            payload["error"]["data"]["requiredCapabilities"], TASK_CAPABILITIES
                        )
                    else:
                        self.assertIn("Unknown task", payload["error"]["message"])
        _response, payload = await self.rpc(
            "tools/call", {"name": "simulate-research-query", "arguments": {"topic": "MCP"}}
        )
        self.assertEqual(payload["error"]["code"], -32021)
        self.assertEqual(payload["error"]["data"]["requiredCapabilities"], TASK_CAPABILITIES)

    async def test_task_update_validation_is_atomic_and_unexpected_answers_are_ignored(self):
        _response, payload = await self.rpc(
            "tools/call",
            {"name": "trigger-elicitation-request-async", "arguments": {}},
            capabilities=FORM_CAPABILITIES,
        )
        task = payload["result"]
        self.assertEqual(task["status"], "input_required")
        task_id = task["taskId"]
        key = next(iter(task["inputRequests"]))
        try:
            # Compare get to get: tools/call adds its own content envelope.
            _response, payload = await self.rpc(
                "tasks/get", {"taskId": task_id}, capabilities=FORM_CAPABILITIES
            )
            task = payload["result"]
            for extra, message in (
                ({}, "inputResponses is required"),
                (
                    {
                        "inputResponses": {
                            key: {"action": "accept", "content": {"name": "x" * 65537}}
                        }
                    },
                    "65536-byte limit",
                ),
            ):
                with self.subTest(message=message):
                    _response, payload = await self.rpc(
                        "tasks/update", {"taskId": task_id, **extra}, capabilities=FORM_CAPABILITIES
                    )
                    self.assertEqual(payload["error"]["code"], -32602)
                    self.assertIn(message, payload["error"]["message"])
                    _response, payload = await self.rpc(
                        "tasks/get", {"taskId": task_id}, capabilities=FORM_CAPABILITIES
                    )
                    self.assertEqual(payload["result"], task)
            # Unknown answers are ignored even if their payload exceeds the
            # retained-answer limit. The outstanding request stays answerable.
            response, payload = await self.rpc(
                "tasks/update",
                {
                    "taskId": task_id,
                    "inputResponses": {
                        "never-issued": {"action": "accept", "content": {"name": "x" * 65537}}
                    },
                },
                capabilities=FORM_CAPABILITIES,
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(payload["result"]["resultType"], "complete")
            _response, payload = await self.rpc(
                "tasks/get", {"taskId": task_id}, capabilities=FORM_CAPABILITIES
            )
            self.assertEqual(payload["result"], task)
            _response, payload = await self.rpc(
                "tasks/update",
                {
                    "taskId": task_id,
                    "inputResponses": {key: {"action": "accept", "content": {"name": "Ada"}}},
                },
                capabilities=FORM_CAPABILITIES,
            )
            self.assertEqual(payload["result"]["resultType"], "complete")
            _response, payload = await self.rpc(
                "tasks/get", {"taskId": task_id}, capabilities=FORM_CAPABILITIES
            )
            self.assertEqual(payload["result"]["status"], "completed")
            self.assertIn("Name: Ada", payload["result"]["result"]["content"][0]["text"])
            completed = payload["result"]
            # Replaying a previously answered key after completion is ignored.
            _response, payload = await self.rpc(
                "tasks/update",
                {"taskId": task_id, "inputResponses": {key: {"action": "decline"}}},
                capabilities=FORM_CAPABILITIES,
            )
            self.assertEqual(payload["result"]["resultType"], "complete")
            _response, payload = await self.rpc(
                "tasks/get", {"taskId": task_id}, capabilities=FORM_CAPABILITIES
            )
            self.assertEqual(payload["result"], completed)
            for method, extra in (
                ("tasks/get", {}),
                ("tasks/update", {"inputResponses": {}}),
                ("tasks/cancel", {}),
            ):
                with self.subTest(missing_extension=method):
                    _response, payload = await self.rpc(method, {"taskId": task_id, **extra})
                    self.assertEqual(payload["error"]["code"], -32021)
                    self.assertEqual(
                        payload["error"]["data"]["requiredCapabilities"], TASK_CAPABILITIES
                    )
        finally:
            await self.rpc("tasks/cancel", {"taskId": task_id}, capabilities=FORM_CAPABILITIES)

    async def test_protocol_metadata_validation_errors(self):
        version_key = "io.modelcontextprotocol/protocolVersion"
        capabilities_key = "io.modelcontextprotocol/clientCapabilities"
        for params, field in (
            ({}, "_meta"),
            ({"_meta": {version_key: MCP_PROTOCOL_VERSION}}, capabilities_key),
            (
                {"_meta": {version_key: MCP_PROTOCOL_VERSION, capabilities_key: []}},
                capabilities_key,
            ),
            ({"_meta": {version_key: 123, capabilities_key: {}}}, version_key),
        ):
            with self.subTest(params=params):
                async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
                    response = await client.post(
                        self.mcp_url,
                        headers={
                            **self.headers,
                            "Accept": "application/json, text/event-stream",
                            "Mcp-Method": "tools/list",
                            "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
                        },
                        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": params},
                    )
                self.assertEqual(response.status_code, 400)
                payload = response.json()
                self.assertEqual(payload["id"], 1)
                self.assertEqual(payload["error"]["code"], -32602)
                self.assertIn(field, payload["error"]["message"])

    async def test_header_parameter_validation_and_protocol_version_mismatch(self):
        for headers in (
            {},
            {"Mcp-Param-Region": "wrong-region"},
            {"Mcp-Param-Region": "=?base64?%%%?="},
        ):
            with self.subTest(headers=headers):
                response, payload = await self.rpc(
                    "tools/call",
                    {"name": "get-header-params", "arguments": {"region": "eu-west-1"}},
                    headers=headers,
                )
                self.assertEqual(response.status_code, 400)
                self.assertEqual(payload["error"]["code"], -32020)
        response, payload = await self.rpc(
            "tools/list", headers={"MCP-Protocol-Version": "2025-11-25"}
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(payload["error"]["code"], -32020)
