---
title: Model Context Protocol (MCP)
parent: API Reference
nav_order: 9
---

# Model Context Protocol (MCP)

The Model Context Protocol (MCP) endpoints allow you to connect to MCP servers through a simple HTTP interface. This
implements the **2026-07-28** version of the streamable HTTP transport specification, which has a **stateless protocol
core**. There is no handshake, no session id, and no server-initiated stream — every request is self-contained and can
land on any gateway instance behind a plain round-robin load balancer.

## How It Works

Aqueduct acts as a stateless bridge between your client application and MCP servers. Each `POST` carries the protocol
version, client identity, and capabilities either in HTTP headers or in the JSON-RPC `_meta` field. The gateway
validates the required transport headers and relays the request to the configured upstream server, returning its
response. Nothing is held in memory between requests.

## Available Endpoint (MCP 2026-07-28)

```
POST /mcp-servers/{name}/mcp     - Send one self-contained MCP request
```

Replace `{name}` with your MCP server name from the MCP server list in the UI. `GET` and `DELETE` are not supported —
the stateless protocol has no sessions to manage or tear down.

### Required headers

Per the 2026-07-28 transport specification, each request must carry:

- `MCP-Protocol-Version`: `2026-07-28`
- `Mcp-Method`: the JSON-RPC method (e.g. `tools/call`, `tools/list`)
- `Mcp-Name`: the tool/resource/prompt name, for name-bearing methods (e.g. `tools/call`, `resources/read`)

### Sequence

``` mermaid
sequenceDiagram
    participant Client
    participant Gateway
    participant Upstream

    note over Client,Gateway: no handshake, no session
    Client->>Gateway: POST MCP request<br/>Mcp-Method / Mcp-Name headers
    Gateway->>Upstream: POST MCP request (headers forwarded)
    alt [single JSON response]
        Upstream->>Gateway: JSON-RPC response
        Gateway->>Client: JSON-RPC response
    else [server streams SSE]
        Upstream->>Gateway: SSE response
        Gateway->>Client: SSE response
    end
```

### Python Example

```python
import httpx

url = "https://your-aqueduct-domain.com/mcp-servers/my-cool-server/mcp"
headers = {
    "Authorization": "Bearer YOUR_AQUEDUCT_TOKEN",
    "Content-Type": "application/json",
    "MCP-Protocol-Version": "2026-07-28",
    "Mcp-Method": "tools/call",
    "Mcp-Name": "echo",
}

body = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {"name": "echo", "arguments": {"message": "hello"}},
    "_meta": {"io.modelcontextprotocol/clientInfo": {"name": "my-app", "version": "1.0"}},
}

response = httpx.post(url, json=body, headers=headers)
print(response.json())
```

For more information about the Model Context Protocol,
visit [modelcontextprotocol.org](https://modelcontextprotocol.org/).
