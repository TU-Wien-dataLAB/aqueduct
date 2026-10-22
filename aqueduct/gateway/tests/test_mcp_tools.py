"""Remaining server-everything v2 tools through the real ASGI gateway."""

import base64
import json

from mcp.types import CreateMessageResult, ElicitResult, ListRootsResult, Root, TextContent

from gateway.tests.utils.mcp import TASKS_EXTENSION, MCPLiveServerTestCase


class MCPModernToolsTest(MCPLiveServerTestCase):
    async def test_annotated_messages_and_optional_image(self):
        async with self.mcp_client() as client:
            for kind, text, priority, audience in (
                ("error", "Error: Operation failed", 1.0, ["user", "assistant"]),
                ("success", "Operation completed successfully", 0.7, ["user"]),
                ("debug", "Debug: Cache hit ratio 0.95, latency 150ms", 0.3, ["assistant"]),
            ):
                with self.subTest(kind=kind):
                    result = await client.call_tool(
                        "get-annotated-message",
                        {"messageType": kind, "includeImage": kind == "success"},
                    )
                    self.assertFalse(result.is_error)
                    self.assertEqual(len(result.content), 2 if kind == "success" else 1)
                    message = result.content[0]
                    self.assertEqual(message.type, "text")
                    self.assertEqual(message.text, text)
                    self.assertEqual(message.annotations.priority, priority)
                    self.assertEqual(message.annotations.audience, audience)
                    if kind == "success":
                        image = result.content[1]
                        self.assertEqual(image.type, "image")
                        self.assertEqual(image.mime_type, "image/png")
                        self.assertTrue(
                            base64.b64decode(image.data).startswith(b"\x89PNG\r\n\x1a\n")
                        )
                        self.assertEqual(image.annotations.priority, 0.5)
                        self.assertEqual(image.annotations.audience, ["user"])

    async def test_environment_is_upstream_process_environment(self):
        async with self.mcp_client() as client:
            result = await client.call_tool("get-env")
            # Never include the environment (or result) in assertion diagnostics:
            # the reference process inherits credentials from the test runner.
            self.assertFalse(bool(result.is_error))
            self.assertEqual(len(result.content), 1)
            self.assertEqual(result.content[0].type, "text")
            environment = json.loads(result.content[0].text)
            self.assertTrue(isinstance(environment, dict))
            self.assertTrue(all(isinstance(value, str) for value in environment.values()))
            self.assertTrue(environment.get("PORT") == str(self.upstream.port))
            self.assertTrue(bool(environment.get("PATH")))

    async def test_embedded_text_and_blob_resource_references(self):
        async with self.mcp_client() as client:
            for resource_type, resource_id in (("Text", 17), ("Blob", 18)):
                with self.subTest(resource_type=resource_type):
                    result = await client.call_tool(
                        "get-resource-reference",
                        {"resourceType": resource_type, "resourceId": resource_id},
                    )
                    self.assertFalse(result.is_error)
                    self.assertEqual(len(result.content), 3)
                    self.assertEqual(
                        result.content[0].text,
                        f"Returning resource reference for Resource {resource_id}:",
                    )
                    embedded = result.content[1]
                    self.assertEqual(embedded.type, "resource")
                    resource = embedded.resource
                    uri = f"demo://resource/dynamic/{resource_type.lower()}/{resource_id}"
                    self.assertEqual(str(resource.uri), uri)
                    self.assertEqual(
                        result.content[2].text, f"You can access this resource using the URI: {uri}"
                    )
                    if resource_type == "Text":
                        self.assertEqual(resource.mime_type, "text/plain")
                        self.assertTrue(
                            resource.text.startswith(
                                "Resource 17: This is a plaintext resource created at "
                            )
                        )
                    else:
                        self.assertEqual(resource.mime_type, "application/octet-stream")
                        self.assertTrue(
                            base64.b64decode(resource.blob).startswith(
                                b"Resource 18: This is a base64 blob created at "
                            )
                        )

    async def test_roots_callback_with_configured_and_empty_roots(self):
        calls = []
        roots = [Root(uri="file:///aqueduct-tests", name="Gateway fixtures")]

        async def list_roots(context):
            calls.append(context)
            return ListRootsResult(roots=roots)

        async with self.mcp_client(list_roots_callback=list_roots) as client:
            result = await client.call_tool("get-roots-list")
            self.assertFalse(result.is_error)
            self.assertEqual(len(calls), 1)
            self.assertIn(
                "Current MCP Roots (1 total):\n\n1. Gateway fixtures\n   URI: file:///aqueduct-tests",
                result.content[0].text,
            )
            roots.clear()
            empty = await client.call_tool("get-roots-list")
            self.assertFalse(empty.is_error)
            self.assertEqual(len(calls), 2)
            self.assertTrue(
                empty.content[0].text.startswith(
                    "The client supports roots but no roots are currently configured."
                )
            )

    async def test_form_elicitation_accept_decline_and_cancel(self):
        content = {
            "name": "Ada",
            "check": False,
            "email": "ada@example.com",
            "homepage": "https://example.com/ada",
            "birthdate": "1990-01-02",
            "integer": 42,
            "number": 3.14,
            "untitledSingleSelectEnum": "Phoebe",
            "untitledMultipleSelectEnum": ["Piano", "Violin"],
            "titledSingleSelectEnum": "hero-3",
            "titledMultipleSelectEnum": ["fish-2"],
            "legacyTitledEnum": "pet-1",
        }
        for action, expected in (
            ("accept", "✅ User provided the requested information!"),
            ("decline", "❌ User declined to provide the requested information."),
            ("cancel", "⚠️ User cancelled the elicitation dialog."),
        ):
            with self.subTest(action=action):
                questions = []

                async def elicit(context, params, *, action=action, questions=questions):
                    questions.append(params)
                    return ElicitResult(
                        action=action, content=content if action == "accept" else None
                    )

                async with self.mcp_client(elicitation_callback=elicit) as client:
                    result = await client.call_tool("trigger-elicitation-request")
                self.assertFalse(result.is_error)
                self.assertEqual(len(questions), 1)
                question = questions[0]
                self.assertEqual(question.mode, "form")
                self.assertEqual(
                    question.message, "Please provide inputs for the following fields:"
                )
                schema = question.requested_schema
                self.assertEqual(schema["required"], ["name"])
                self.assertEqual(schema["properties"]["check"]["type"], "boolean")
                self.assertEqual(schema["properties"]["integer"]["minimum"], 1)
                self.assertEqual(schema["properties"]["integer"]["maximum"], 100)
                self.assertEqual(schema["properties"]["email"]["format"], "email")
                self.assertEqual(result.content[0].text, expected)
                raw = json.loads(result.content[-1].text.removeprefix("\nRaw result: "))
                self.assertEqual(
                    raw,
                    {"action": action, "content": content}
                    if action == "accept"
                    else {"action": action},
                )
                if action == "accept":
                    self.assertEqual(
                        result.content[1].text,
                        "User inputs:\n- Name: Ada\n- Agreed to terms: false"
                        "\n- Email: ada@example.com\n- Homepage: https://example.com/ada"
                        "\n- Birthdate: 1990-01-02\n- Favorite Integer: 42"
                        "\n- Favorite Number: 3.14",
                    )
        # v2.0.0 reads inputResponse directly, without schema-aware acceptedContent:
        # this tool does not validate accepted content or implement a validation retry.

    async def test_sampling_callback_request_and_returned_completion(self):
        requests = []
        completion = CreateMessageResult(
            role="assistant",
            content=TextContent(type="text", text="A sampled answer"),
            model="aqueduct-test-model",
            stop_reason="endTurn",
        )

        async def sample(context, params):
            requests.append(params)
            return completion

        async with self.mcp_client(sampling_callback=sample) as client:
            result = await client.call_tool(
                "trigger-sampling-request", {"prompt": "Say hello", "maxTokens": 23}
            )
        self.assertFalse(result.is_error)
        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(request.max_tokens, 23)
        self.assertEqual(request.temperature, 0.7)
        self.assertEqual(request.system_prompt, "You are a helpful test server.")
        self.assertEqual(len(request.messages), 1)
        self.assertEqual(request.messages[0].role, "user")
        self.assertEqual(
            request.messages[0].content.text, "Resource trigger-sampling-request context: Say hello"
        )
        self.assertEqual(
            json.loads(result.content[0].text.removeprefix("LLM sampling result: \n")),
            completion.model_dump(by_alias=True, exclude_none=True),
        )

    async def test_async_sampling_task_get_update_and_completed_result(self):
        capabilities = {"extensions": {TASKS_EXTENSION: {}}, "sampling": {}}
        response, payload = await self.rpc(
            "tools/call",
            {
                "name": "trigger-sampling-request-async",
                "arguments": {"prompt": "Summarize the gateway", "maxTokens": 31},
            },
            capabilities=capabilities,
        )
        self.assertEqual(response.status_code, 200)
        task = payload["result"]
        self.assertEqual(task["resultType"], "task")
        self.assertEqual(task["status"], "input_required")
        task_id = task["taskId"]
        self.assertTrue(task_id)
        self.assertEqual(len(task["inputRequests"]), 1)
        key, request = next(iter(task["inputRequests"].items()))
        self.assertEqual(request["method"], "sampling/createMessage")
        self.assertEqual(
            request["params"],
            {
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": (
                                "Resource trigger-sampling-request-async context: "
                                "Summarize the gateway"
                            ),
                        },
                    }
                ],
                "systemPrompt": "You are a helpful test server.",
                "maxTokens": 31,
                "temperature": 0.7,
            },
        )
        response, payload = await self.rpc(
            "tasks/get", {"taskId": task_id}, capabilities=capabilities
        )
        self.assertEqual(response.status_code, 200)
        waiting = payload["result"]
        self.assertEqual(waiting["taskId"], task_id)
        self.assertEqual(waiting["status"], "input_required")
        self.assertEqual(waiting["inputRequests"], {key: request})
        completion = {
            "role": "assistant",
            "content": {"type": "text", "text": "The gateway relays MCP requests."},
            "model": "aqueduct-task-model",
            "stopReason": "endTurn",
        }
        response, payload = await self.rpc(
            "tasks/update",
            {"taskId": task_id, "inputResponses": {key: completion}},
            capabilities=capabilities,
        )
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("error", payload)
        response, payload = await self.rpc(
            "tasks/get", {"taskId": task_id}, capabilities=capabilities
        )
        self.assertEqual(response.status_code, 200)
        completed = payload["result"]
        self.assertEqual(completed["taskId"], task_id)
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["statusMessage"], "Sampling complete")
        self.assertNotIn("inputRequests", completed)
        result = completed["result"]
        self.assertFalse(result.get("isError", False))
        self.assertEqual(len(result["content"]), 1)
        self.assertEqual(result["content"][0]["type"], "text")
        self.assertEqual(
            json.loads(result["content"][0]["text"].removeprefix("LLM sampling result: \n")),
            completion,
        )

    async def test_url_elicitation_actions_use_in_band_flow_for_both_error_paths(self):
        url = "https://example.com/authorize"
        for error_path in (False, True):
            for action, expected in (
                ("accept", f"✅ User completed the URL elicitation flow.\nURL: {url}"),
                ("decline", "❌ User declined to open the URL."),
                ("cancel", "⚠️ User cancelled the URL elicitation."),
            ):
                with self.subTest(error_path=error_path, action=action):
                    questions = []

                    async def elicit(context, params, *, action=action, questions=questions):
                        questions.append(params)
                        return ElicitResult(action=action)

                    async with self.mcp_client(elicitation_callback=elicit) as client:
                        result = await client.call_tool(
                            "trigger-url-elicitation",
                            {
                                "url": url,
                                "message": "Authorize the test application",
                                "errorPath": error_path,
                                "elicitationId": "legacy-id-is-ignored",
                            },
                        )
                    self.assertFalse(result.is_error)
                    self.assertEqual(len(questions), 1)
                    question = questions[0]
                    self.assertEqual(question.mode, "url")
                    self.assertEqual(str(question.url), url)
                    self.assertEqual(question.message, "Authorize the test application")
                    self.assertNotIn(
                        "elicitationId", question.model_dump(by_alias=True, exclude_none=True)
                    )
                    self.assertEqual(len(result.content), 2)
                    self.assertEqual(result.content[0].text, expected)
                    self.assertEqual(
                        json.loads(result.content[1].text.removeprefix("\nRaw result: ")),
                        {"action": action},
                    )
