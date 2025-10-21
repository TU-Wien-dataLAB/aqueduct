from gateway.tests.utils.mcp import MCPLiveServerTestCase


class MCPLiveClientTest(MCPLiveServerTestCase):
    async def test_list_tools(self):
        """Test MCP tools listing using the exact pattern from scratch file."""
        async with self.client_session() as session:
            await session.initialize()
            tools = await session.list_tools()
            print(f"Available tools: {[tool.name for tool in tools.tools]}")
