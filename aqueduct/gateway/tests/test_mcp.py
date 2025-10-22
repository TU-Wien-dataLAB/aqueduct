from datetime import timedelta

from pydantic.networks import AnyUrl

from gateway.tests.utils.mcp import MCPLiveServerTestCase


class MCPLiveClientTest(MCPLiveServerTestCase):
    async def test_list_tools(self):
        """Test MCP tools listing using the exact pattern from scratch file."""
        async with self.client_session() as session:
            await session.initialize()
            tools = await session.list_tools()

            self.assertIsNotNone(tools)
            self.assertIsInstance(tools.tools, list)
            self.assertGreater(len(tools.tools), 0)
            # Verify expected tools are present
            tool_names = [tool.name for tool in tools.tools]
            self.assertIn("echo", tool_names)

    async def test_call_tool(self):
        """Test calling a tool."""
        async with self.client_session() as session:
            await session.initialize()
            tool_name = "echo"
            result = await session.call_tool(tool_name, {"message": "test"})

            self.assertIsNotNone(result)
            self.assertIsInstance(result.content, list)
            self.assertGreater(len(result.content), 0)
            # Verify the echo tool returns the input message
            self.assertIn("test", result.content[0].text)

    async def test_call_tool_long_running(self):
        """Test calling a long-running tool with progress updates."""
        async with self.client_session() as session:
            await session.initialize()
            tool_name = "longRunningOperation"
            result = await session.call_tool(tool_name, {"duration": 3, "steps": 5})

            self.assertIsNotNone(result)
            self.assertIsInstance(result.content, list)
            self.assertGreater(len(result.content), 0)

    async def test_list_resources(self):
        """Test listing resources."""
        async with self.client_session() as session:
            await session.initialize()
            resources = await session.list_resources()
            self.assertIsNotNone(resources)
            self.assertIsInstance(resources.resources, list)
            # Resources may be empty, but the response should be valid
            self.assertGreater(len(resources.resources), 0)

    async def test_read_resource(self):
        """Test reading a resource."""
        async with self.client_session() as session:
            await session.initialize()
            resources = await session.list_resources()
            self.assertGreater(len(resources.resources), 0)

            resource_uri = resources.resources[0].uri
            result = await session.read_resource(resource_uri)

            self.assertIsNotNone(result)
            self.assertIsInstance(result.contents, list)
            self.assertGreater(len(result.contents), 0)

    async def test_list_prompts(self):
        """Test listing prompts."""
        async with self.client_session() as session:
            await session.initialize()
            prompts = await session.list_prompts()

            self.assertIsNotNone(prompts)
            self.assertIsInstance(prompts.prompts, list)
            self.assertGreater(len(prompts.prompts), 0)
            # Verify prompt names are strings
            for prompt in prompts.prompts:
                self.assertIsInstance(prompt.name, str)

    async def test_get_prompt(self):
        """Test getting a prompt."""
        async with self.client_session() as session:
            await session.initialize()
            prompts = await session.list_prompts()
            self.assertGreater(len(prompts.prompts), 0)

            prompt_name = prompts.prompts[0].name
            result = await session.get_prompt(prompt_name)

            self.assertIsNotNone(result)
            self.assertIsInstance(result.messages, list)
            self.assertGreater(len(result.messages), 0)

    async def test_send_ping(self):
        """Test sending ping."""
        async with self.client_session() as session:
            await session.initialize()
            result = await session.send_ping()

            self.assertIsNotNone(result)

    async def test_list_resource_templates(self):
        """Test listing resource templates."""
        async with self.client_session() as session:
            await session.initialize()
            templates = await session.list_resource_templates()

            self.assertIsNotNone(templates)
            self.assertIsInstance(templates.resourceTemplates, list)
            # Resource templates may be empty, but response should be valid
            if templates.resourceTemplates:
                for template in templates.resourceTemplates:
                    self.assertIsInstance(template.uriTemplate, str)

    async def test_complete_resource_template(self):
        """Test completion for resource template reference."""
        async with self.client_session() as session:
            await session.initialize()
            templates = await session.list_resource_templates()
            if templates.resourceTemplates:
                template = templates.resourceTemplates[0]
                from mcp.types import ResourceTemplateReference

                ref = ResourceTemplateReference(type="ref/resource", uri=template.uriTemplate)
                argument = {"name": "test", "value": "argument"}
                result = await session.complete(ref, argument)

                self.assertIsNotNone(result)
                self.assertIsNotNone(result.completion)
                self.assertIsInstance(result.completion.values, list)
            else:
                self.skipTest("No resource templates available for completion test")

    async def test_complete_prompt_reference(self):
        """Test completion for prompt reference."""
        async with self.client_session() as session:
            await session.initialize()
            prompts = await session.list_prompts()
            if prompts.prompts:
                prompt = prompts.prompts[0]
                from mcp.types import PromptReference

                ref = PromptReference(type="ref/prompt", name=prompt.name)
                argument = {"name": "test", "value": "argument"}
                result = await session.complete(ref, argument)

                self.assertIsNotNone(result)
                self.assertIsNotNone(result.completion)
                self.assertIsInstance(result.completion.values, list)
            else:
                self.skipTest("No prompts available for completion test")

    async def test_set_logging_level(self):
        """Test setting server logging level."""
        async with self.client_session() as session:
            await session.initialize()
            result = await session.set_logging_level("debug")

            self.assertIsNotNone(result)

    async def test_send_roots_list_changed(self):
        """Test sending roots list changed notification."""
        async with self.client_session() as session:
            await session.initialize()
            result = await session.send_roots_list_changed()

            self.assertIsNone(result)

    async def test_send_progress_notification(self):
        """Test sending progress notification."""
        async with self.client_session() as session:
            await session.initialize()
            result = await session.send_progress_notification(
                progress_token="test-token", progress=50.0, total=100.0, message="Processing..."
            )

            self.assertIsNone(result)

    async def test_call_tool_with_progress_callback(self):
        """Test calling tool with progress callback."""
        progress_updates = []

        def progress_callback(progress_token, progress, total=None):
            progress_updates.append((progress_token, progress, total))

        async with self.client_session() as session:
            await session.initialize()
            tool_name = "longRunningOperation"
            result = await session.call_tool(
                tool_name, {"duration": 2, "steps": 3}, progress_callback=progress_callback
            )

            self.assertIsNotNone(result)
            self.assertIsInstance(result.content, list)
            self.assertGreater(len(result.content), 0)
            # Verify progress updates were received
            self.assertGreater(len(progress_updates), 0)
            # Verify structure of progress updates
            for update in progress_updates:
                self.assertEqual(len(update), 3)
                progress_token, progress, total = update
                self.assertIsInstance(progress, (int, float))

    async def test_call_tool_with_timeout(self):
        """Test calling tool with read timeout."""
        async with self.client_session() as session:
            await session.initialize()
            tool_name = "echo"
            result = await session.call_tool(
                tool_name, {"message": "test"}, read_timeout_seconds=timedelta(seconds=10)
            )

            self.assertIsNotNone(result)
            self.assertIsInstance(result.content, list)
            self.assertGreater(len(result.content), 0)

    async def test_list_methods_with_cursor(self):
        """Test list methods with cursor parameter functionality."""
        async with self.client_session() as session:
            await session.initialize()

            # Test tools with cursor (if supported)
            tools = await session.list_tools()
            self.assertIsNotNone(tools)
            if hasattr(tools, "nextCursor") and tools.nextCursor:
                next_tools = await session.list_tools(cursor=tools.nextCursor)
                self.assertIsNotNone(next_tools)
                self.assertIsInstance(next_tools.tools, list)

            # Test resources with cursor (if supported)
            resources = await session.list_resources()
            self.assertIsNotNone(resources)
            if hasattr(resources, "nextCursor") and resources.nextCursor:
                next_resources = await session.list_resources(cursor=resources.nextCursor)
                self.assertIsNotNone(next_resources)
                self.assertIsInstance(next_resources.resources, list)

    async def test_error_invalid_tool_name(self):
        """Test error handling for invalid tool name."""
        async with self.client_session() as session:
            await session.initialize()
            with self.assertRaises(Exception) as context:
                await session.call_tool("nonexistent_tool", {})

            self.assertIsNotNone(context.exception)

    async def test_error_invalid_resource_uri(self):
        """Test error handling for invalid resource URI."""
        async with self.client_session() as session:
            await session.initialize()
            with self.assertRaises(Exception) as context:
                invalid_uri = AnyUrl("invalid://not-a-real-uri")
                await session.read_resource(invalid_uri)

            self.assertIsNotNone(context.exception)

    async def test_initialize(self):
        """Test initialize method directly."""
        async with self.client_session() as session:
            result = await session.initialize()

            self.assertIsNotNone(result)
            # Verify initialization result has expected attributes
            self.assertIsNotNone(result.serverInfo)
            self.assertIsNotNone(result.capabilities)

    async def test_session_creation(self):
        """Test that sessions have unique IDs."""
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        async with streamablehttp_client(self.mcp_url, headers=self.headers) as (
            r,
            w,
            get_session_id,
        ):
            async with ClientSession(r, w) as session:
                await session.initialize()
                s1 = get_session_id()

        async with streamablehttp_client(self.mcp_url, headers=self.headers) as (
            r,
            w,
            get_session_id,
        ):
            async with ClientSession(r, w) as session:
                await session.initialize()
                s2 = get_session_id()

        self.assertNotEqual(s1, s2)
