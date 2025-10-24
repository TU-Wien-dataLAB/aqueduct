---
title: Model Context Protocol (MCP)
parent: API Reference
nav_order: 9
---

# Model Context Protocol (MCP)

The Model Context Protocol (MCP) endpoints allow you to connect to MCP servers through a simple HTTP interface. This
implements the 2025-06-18 version of the streamable HTTP transport specification, making it easy to integrate with
MCP-compatible tools.

## How It Works

The MCP gateway acts as a bridge between your client application and MCP servers. You can request tools, call functions,
and receive responses through standard HTTP requests while the gateway manages the underlying session.

## Available Endpoints (MCP 2025-08-16 spec)

```
GET    /mcp-servers/{name}/mcp     - Start streaming responses
POST   /mcp-servers/{name}/mcp     - Send MCP messages
DELETE /mcp-servers/{name}/mcp     - End a session
```

Replace `{name}` with your MCP server name from the MCP server list in the UI.

### Sequence Diagram (MCP 2025-08-16 spec)

``` mermaid
sequenceDiagram
    participant Client
    participant Server
    
    note over Client,Server: initialization
    Client->>Server: POST InitializeRequest
    Server->>Client: InitializeResponse<br/>Mcp-Session-Id: 1868a90c...
    Client->>Server: POST InitializedNotification<br/>Mcp-Session-Id: 1868a90c...
    Server->>Client: 202 Accepted
    
    note over Client,Server: client requests
    Client->>Server: POST ... request ...<br/>Mcp-Session-Id: 1868a90c...
    
    alt [single HTTP response]
        Server->>Client: ... response ...
    else [server opens SSE stream]
        loop [while connection remains open]
            Server->>Client: ... SSE messages from server ...
            Server->>Client: SSE event: ... response ...
        end
    end
    
    note over Client,Server: client notifications/responses
    Client->>Server: POST ... notification/response ...<br/>Mcp-Session-Id: 1868a90c...
    Server->>Client: 202 Accepted
    
    note over Client,Server: server requests
    loop [while connection remains open]
        Client->>Server: GET<br/>Mcp-Session-Id: 1868a90c...
        Server->>Client: ... SSE messages from server ...
    end
```

### Python Example

```python
import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def main():
    # Connect to your MCP server through Aqueduct
    url = "https://your-aqueduct-domain.com/mcp-servers/my-cool-server/mcp"
    headers = {"Authorization": "Bearer YOUR_AQUEDUCT_TOKEN"}

    async with streamablehttp_client(url, headers=headers) as (
            read_stream,
            write_stream,
            _,
    ):
        # Create a session
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the connection
            await session.initialize()

            # List available tools
            tools = await session.list_tools()
            print(f"Available tools: {[tool.name for tool in tools.tools]}")


if __name__ == "__main__":
    asyncio.run(main())
```

For more information about the Model Context Protocol,
visit [modelcontextprotocol.org](https://modelcontextprotocol.org/).