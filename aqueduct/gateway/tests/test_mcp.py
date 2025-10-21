from datetime import timedelta

from pydantic.networks import AnyUrl

from gateway.tests.utils.mcp import MCPLiveServerTestCase


class MCPLiveClientTest(MCPLiveServerTestCase):
    async def test_list_tools(self):
        """Test MCP tools listing using the exact pattern from scratch file."""
        async with self.client_session() as session:
            await session.initialize()
            tools = await session.list_tools()
            print(f"Available tools: {[tool.name for tool in tools.tools]}")

    async def test_call_tool(self):
        """Test calling a tool."""
        async with self.client_session() as session:
            await session.initialize()
            tool_name = "echo"
            result = await session.call_tool(tool_name, {"message": "test"})
            print(f"Tool result: {result.content}")

    async def test_call_tool_long_running(self):
        """Test calling a long-running tool with progress updates."""
        async with self.client_session() as session:
            await session.initialize()
            tool_name = "longRunningOperation"
            result = await session.call_tool(tool_name, {"duration": 3, "steps": 5})
            print(f"Tool result: {result.content}")

    async def test_list_resources(self):
        """Test listing resources."""
        async with self.client_session() as session:
            await session.initialize()
            resources = await session.list_resources()
            print(f"Available resources: {[r.uri for r in resources.resources]}")

    async def test_read_resource(self):
        """Test reading a resource."""
        async with self.client_session() as session:
            await session.initialize()
            resources = await session.list_resources()
            resource_uri = resources.resources[0].uri
            result = await session.read_resource(resource_uri)
            print(f"Resource content: {result.contents}")

    async def test_list_prompts(self):
        """Test listing prompts."""
        async with self.client_session() as session:
            await session.initialize()
            prompts = await session.list_prompts()
            print(f"Available prompts: {[p.name for p in prompts.prompts]}")

    async def test_get_prompt(self):
        """Test getting a prompt."""
        async with self.client_session() as session:
            await session.initialize()
            prompts = await session.list_prompts()
            prompt_name = prompts.prompts[0].name
            result = await session.get_prompt(prompt_name)
            print(f"Prompt messages: {result.messages}")

    async def test_send_ping(self):
        """Test sending ping."""
        async with self.client_session() as session:
            await session.initialize()
            result = await session.send_ping()
            print(f"Ping result: {result}")

    async def test_list_resource_templates(self):
        """Test listing resource templates."""
        async with self.client_session() as session:
            await session.initialize()
            templates = await session.list_resource_templates()
            print(
                f"Available resource templates: {[t.uriTemplate for t in templates.resourceTemplates]}"
            )

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
                print(f"Resource template completion result: {result.completion.values}")
            else:
                print("No resource templates available for completion test")

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
                print(f"Prompt completion result: {result.completion.values}")
            else:
                print("No prompts available for completion test")

    async def test_set_logging_level(self):
        """Test setting server logging level."""
        async with self.client_session() as session:
            await session.initialize()
            result = await session.set_logging_level("debug")
            print(f"Set logging level result: {result}")

    async def test_send_roots_list_changed(self):
        """Test sending roots list changed notification."""
        async with self.client_session() as session:
            await session.initialize()
            result = await session.send_roots_list_changed()
            print(f"Roots list changed result: {result}")

    async def test_send_progress_notification(self):
        """Test sending progress notification."""
        async with self.client_session() as session:
            await session.initialize()
            result = await session.send_progress_notification(
                progress_token="test-token", progress=50.0, total=100.0, message="Processing..."
            )
            print(f"Progress notification result: {result}")

    async def test_call_tool_with_progress_callback(self):
        """Test calling tool with progress callback."""
        progress_updates = []

        def progress_callback(progress_token, progress, total=None):
            progress_updates.append((progress_token, progress, total))
            print(
                f"Progress update - token: {progress_token}, progress: {progress}, total: {total}"
            )

        async with self.client_session() as session:
            await session.initialize()
            tool_name = "longRunningOperation"
            result = await session.call_tool(
                tool_name, {"duration": 2, "steps": 3}, progress_callback=progress_callback
            )
            print(f"Tool result: {result.content}")
            print(f"Received {len(progress_updates)} progress updates")

    async def test_call_tool_with_timeout(self):
        """Test calling tool with read timeout."""
        async with self.client_session() as session:
            await session.initialize()
            tool_name = "echo"
            result = await session.call_tool(
                tool_name, {"message": "test"}, read_timeout_seconds=timedelta(seconds=10)
            )
            print(f"Tool result: {result.content}")

    async def test_list_methods_with_cursor(self):
        """Test list methods with cursor parameter functionality."""
        async with self.client_session() as session:
            await session.initialize()

            # Test tools with cursor (if supported)
            tools = await session.list_tools()
            if hasattr(tools, "nextCursor") and tools.nextCursor:
                next_tools = await session.list_tools(cursor=tools.nextCursor)
                print(f"Next page tools: {[tool.name for tool in next_tools.tools]}")

            # Test resources with cursor (if supported)
            resources = await session.list_resources()
            if hasattr(resources, "nextCursor") and resources.nextCursor:
                next_resources = await session.list_resources(cursor=resources.nextCursor)
                print(f"Next page resources: {[r.uri for r in next_resources.resources]}")

    async def test_error_invalid_tool_name(self):
        """Test error handling for invalid tool name."""
        async with self.client_session() as session:
            await session.initialize()
            try:
                await session.call_tool("nonexistent_tool", {})
                print("ERROR: Should have raised exception for invalid tool")
            except Exception as e:
                print(f"Correctly caught error for invalid tool: {e}")

    async def test_error_invalid_resource_uri(self):
        """Test error handling for invalid resource URI."""
        async with self.client_session() as session:
            await session.initialize()
            try:
                invalid_uri = AnyUrl("invalid://not-a-real-uri")
                await session.read_resource(invalid_uri)
                print("ERROR: Should have raised exception for invalid URI")
            except Exception as e:
                print(f"Correctly caught error for invalid URI: {e}")

    async def test_initialize(self):
        """Test initialize method directly."""
        async with self.client_session() as session:
            result = await session.initialize()
            print(f"Initialize result: {result}")
