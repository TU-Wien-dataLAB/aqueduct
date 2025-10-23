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

        await self.assertRequestLogged()

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

        await self.assertRequestLogged()

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

        await self.assertRequestLogged()

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

        await self.assertRequestLogged()

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

        await self.assertRequestLogged()

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

        await self.assertRequestLogged()

    async def test_send_ping(self):
        """Test sending ping."""
        async with self.client_session() as session:
            await session.initialize()
            result = await session.send_ping()

            self.assertIsNotNone(result)

        await self.assertRequestLogged()

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

        await self.assertRequestLogged()

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

        await self.assertRequestLogged()

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

        await self.assertRequestLogged()

    async def test_set_logging_level(self):
        """Test setting server logging level."""
        async with self.client_session() as session:
            await session.initialize()
            result = await session.set_logging_level("debug")

            self.assertIsNotNone(result)

        await self.assertRequestLogged()

    async def test_send_roots_list_changed(self):
        """Test sending roots list changed notification."""
        async with self.client_session() as session:
            await session.initialize()
            result = await session.send_roots_list_changed()

            self.assertIsNone(result)

        await self.assertRequestLogged()

    async def test_send_progress_notification(self):
        """Test sending progress notification."""
        async with self.client_session() as session:
            await session.initialize()
            result = await session.send_progress_notification(
                progress_token="test-token", progress=50.0, total=100.0, message="Processing..."
            )

            self.assertIsNone(result)

        await self.assertRequestLogged()

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

        await self.assertRequestLogged()

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

        await self.assertRequestLogged()

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

        await self.assertRequestLogged()

    async def test_error_invalid_tool_name(self):
        """Test error handling for invalid tool name."""
        async with self.client_session() as session:
            await session.initialize()
            with self.assertRaises(Exception) as context:
                await session.call_tool("nonexistent_tool", {})

            self.assertIsNotNone(context.exception)

        await self.assertRequestLogged()

    async def test_error_invalid_resource_uri(self):
        """Test error handling for invalid resource URI."""
        async with self.client_session() as session:
            await session.initialize()
            with self.assertRaises(Exception) as context:
                invalid_uri = AnyUrl("invalid://not-a-real-uri")
                await session.read_resource(invalid_uri)

            self.assertIsNotNone(context.exception)

        await self.assertRequestLogged()

    async def test_initialize(self):
        """Test initialize method directly."""
        async with self.client_session() as session:
            result = await session.initialize()

            self.assertIsNotNone(result)
            # Verify initialization result has expected attributes
            self.assertIsNotNone(result.serverInfo)
            self.assertIsNotNone(result.capabilities)

        await self.assertRequestLogged()

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
        await self.assertRequestLogged(n=2)


class MCPTransportSecurityTest(MCPLiveServerTestCase):
    """Test MCP transport security (DNS rebinding protection)."""

    async def test_valid_host_allowed(self):
        """Test that valid hosts are allowed."""
        # The normal client_session should work with valid hosts (localhost:*)
        async with self.client_session() as session:
            result = await session.initialize()
            self.assertIsNotNone(result)
            self.assertIsNotNone(result.serverInfo)

        await self.assertRequestLogged()

    async def test_invalid_host_rejected(self):
        """Test that invalid Host header is rejected with 421."""
        import httpx

        # We need to test with a host that's valid for Django's ALLOWED_HOSTS
        # but invalid for our MCP security settings.
        # Since the test server runs on a random port, we use a different port on localhost
        headers = {
            "Authorization": self.headers["Authorization"],
            "Content-Type": "application/json",
        }

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        }

        # Create a custom request with a host not in our allowed list
        # Use a specific port that's not in localhost:* pattern would be caught,
        # but our test allows localhost:*, so we need to use a completely different host
        # that Django would allow (testserver) but our security wouldn't
        async with httpx.AsyncClient() as client:
            request = client.build_request("POST", self.mcp_url, json=payload, headers=headers)
            # Use a host that would fail our security check
            # Django test server allows 'testserver' by default
            request.headers["host"] = "evil.testserver:8000"
            response = await client.send(request)

            # Should return 421 for invalid Host header
            self.assertEqual(response.status_code, 421)
            self.assertIn("error", response.json())

        await self.assertRequestLogged(n=0)

    async def test_invalid_origin_rejected(self):
        """Test that invalid Origin header is rejected with 403."""
        from urllib.parse import urlparse

        import httpx

        # Extract the actual host from the live server URL
        parsed_url = urlparse(self.live_server_url)
        valid_host = parsed_url.netloc

        headers = {
            "Authorization": self.headers["Authorization"],
            "Content-Type": "application/json",
            "Host": valid_host,
            "Origin": "https://evil.com",
        }

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(self.mcp_url, json=payload, headers=headers)

            # Should return 403 for invalid Origin header
            self.assertEqual(response.status_code, 403)
            self.assertIn("error", response.json())

        await self.assertRequestLogged(0)

    async def test_invalid_content_type_rejected(self):
        """Test that invalid Content-Type is rejected with 400."""
        from urllib.parse import urlparse

        import httpx

        # Extract the actual host from the live server URL
        parsed_url = urlparse(self.live_server_url)
        valid_host = parsed_url.netloc

        headers = {
            "Authorization": self.headers["Authorization"],
            "Content-Type": "text/plain",
            "Host": valid_host,
        }

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(self.mcp_url, json=payload, headers=headers)

            # Should return 400 for invalid Content-Type
            self.assertEqual(response.status_code, 400)
            self.assertIn("error", response.json())

        await self.assertRequestLogged(n=0)

    async def test_missing_origin_allowed(self):
        """Test that missing Origin header is allowed (same-origin requests)."""
        # Normal requests without Origin should work
        async with self.client_session() as session:
            result = await session.initialize()
            self.assertIsNotNone(result)
            self.assertIsNotNone(result.serverInfo)

        await self.assertRequestLogged()

    async def test_wildcard_port_allowed(self):
        """Test that wildcard port patterns work (localhost:*)."""
        from urllib.parse import urlparse

        import httpx

        # Extract the actual host from the live server URL (should match localhost:*)
        parsed_url = urlparse(self.live_server_url)
        valid_host = parsed_url.netloc

        headers = {
            "Authorization": self.headers["Authorization"],
            "Content-Type": "application/json",
            "Host": valid_host,
        }

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(self.mcp_url, json=payload, headers=headers)

            # Should succeed because localhost:* is in allowed hosts
            self.assertEqual(response.status_code, 200)

        await self.assertRequestLogged(n=1)


class MCPServerExclusionTest(MCPLiveServerTestCase):
    """Test MCP server exclusion functionality."""

    async def test_mcp_server_access_allowed(self):
        """Test that MCP server is accessible when not excluded."""
        async with self.client_session() as session:
            result = await session.initialize()
            self.assertIsNotNone(result)
            self.assertIsNotNone(result.serverInfo)

        await self.assertRequestLogged()

    async def test_org_excluded_mcp_server(self):
        """Test that MCP server is blocked when excluded at org level."""
        from asgiref.sync import sync_to_async

        from management.models import Org

        # Get org and add exclusion
        org = await sync_to_async(Org.objects.get)(name="E060")
        await sync_to_async(org.add_excluded_mcp_server)("test-server")

        # Try to access the MCP server - should get 404
        import httpx

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(self.mcp_url, json=payload, headers=self.headers)
            self.assertEqual(response.status_code, 404)

        # Clean up
        await sync_to_async(org.remove_excluded_mcp_server)("test-server")

    async def test_team_excluded_mcp_server(self):
        """Test that MCP server is blocked when excluded at team level."""
        from asgiref.sync import sync_to_async

        from management.models import ServiceAccount, Team, Token

        # Get team and add exclusion
        team = await sync_to_async(Team.objects.get)(name="Whale")
        await sync_to_async(team.add_excluded_mcp_server)("test-server")

        # Create a service account for the team
        service_account = await sync_to_async(ServiceAccount.objects.create)(
            team=team, name="Test Service Account"
        )

        # Create a token for the service account
        from gateway.tests.utils.base import GatewayIntegrationTestCase

        token = await sync_to_async(Token.objects.get)(
            key_hash=Token._hash_key(GatewayIntegrationTestCase.AQUEDUCT_ACCESS_TOKEN)
        )
        token.service_account = service_account
        await sync_to_async(token.save)()

        # Try to access the MCP server - should get 404
        import httpx

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(self.mcp_url, json=payload, headers=self.headers)
            self.assertEqual(response.status_code, 404)

        # Clean up
        token.service_account = None
        await sync_to_async(token.save)()
        await sync_to_async(service_account.delete)()
        await sync_to_async(team.remove_excluded_mcp_server)("test-server")

    async def test_user_excluded_mcp_server(self):
        """Test that MCP server is blocked when excluded at user profile level."""
        from asgiref.sync import sync_to_async
        from django.contrib.auth import get_user_model

        User = get_user_model()

        # Get user and their profile
        user = await sync_to_async(User.objects.get)(username="Me")
        profile = await sync_to_async(lambda: user.profile)()

        # Add exclusion to user profile
        await sync_to_async(profile.add_excluded_mcp_server)("test-server")

        # Try to access the MCP server - should get 404
        import httpx

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(self.mcp_url, json=payload, headers=self.headers)
            self.assertEqual(response.status_code, 404)

        # Clean up
        await sync_to_async(profile.remove_excluded_mcp_server)("test-server")

    async def test_merged_exclusion_lists(self):
        """Test that exclusion lists merge correctly across hierarchy."""
        from asgiref.sync import sync_to_async
        from django.contrib.auth import get_user_model

        from gateway.tests.utils.base import GatewayIntegrationTestCase
        from management.models import Org, Token

        User = get_user_model()

        # Get the token to test exclusion list logic
        token = await sync_to_async(Token.objects.get)(
            key_hash=Token._hash_key(GatewayIntegrationTestCase.AQUEDUCT_ACCESS_TOKEN)
        )

        # Setup: Org excludes test-server, user has merge enabled (default)
        org = await sync_to_async(Org.objects.get)(name="E060")
        await sync_to_async(org.add_excluded_mcp_server)("test-server")

        user = await sync_to_async(User.objects.get)(username="Me")
        profile = await sync_to_async(lambda: user.profile)()

        # Verify merge_mcp_server_exclusion_lists is True by default
        merge_enabled = await sync_to_async(lambda: profile.merge_mcp_server_exclusion_lists)()
        self.assertTrue(merge_enabled)

        # Check that test-server is in exclusion list (should be merged from org)
        exclusion_list = await sync_to_async(token.mcp_server_exclusion_list)()
        self.assertIn("test-server", exclusion_list)

        # Verify the server is marked as excluded
        is_excluded = await sync_to_async(token.mcp_server_excluded)("test-server")
        self.assertTrue(is_excluded)

        # Now disable merging at user level
        profile.merge_mcp_server_exclusion_lists = False
        await sync_to_async(profile.save)()

        # Check that test-server is NOT in exclusion list (merge disabled, user has no exclusions)
        exclusion_list = await sync_to_async(token.mcp_server_exclusion_list)()
        self.assertNotIn("test-server", exclusion_list)

        # Verify the server is NOT marked as excluded
        is_excluded = await sync_to_async(token.mcp_server_excluded)("test-server")
        self.assertFalse(is_excluded)

        # Clean up
        profile.merge_mcp_server_exclusion_lists = True
        await sync_to_async(profile.save)()
        await sync_to_async(org.remove_excluded_mcp_server)("test-server")

    async def test_nonexistent_mcp_server(self):
        """Test that accessing a non-existent MCP server returns 404."""
        import httpx

        nonexistent_url = f"{self.live_server_url}/mcp-servers/nonexistent-server/mcp"

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(nonexistent_url, json=payload, headers=self.headers)
            self.assertEqual(response.status_code, 404)

        await self.assertRequestLogged(n=1)
