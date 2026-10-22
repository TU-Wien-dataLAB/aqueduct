---
title: Model Context Protocol (MCP)
parent: API Reference
nav_order: 9
---

# Model Context Protocol (MCP)

Aqueduct relays Model Context Protocol (MCP) requests to configured upstream servers using the
**2026-07-28 Streamable HTTP** request model. This revision has no initialization handshake or
protocol-level sessions: each request carries its own protocol version and client capabilities.

## How It Works

Each request is an independent POST, with either a JSON response or a Server-Sent Events (SSE)
response stream. There is no standalone GET stream, but long-lived change notifications are still
possible through the SSE response to a POST `subscriptions/listen` request. Statelessness does not
prohibit subscriptions; their state is scoped to the request.

Aqueduct authenticates the caller, checks server access and transport security, and relays the request.
JSON responses and SSE response streams are returned with the upstream status code. This is not a
byte-for-byte HTTP proxy.

## Available Endpoint (MCP 2026-07-28)

```text
POST /mcp-servers/{name}/mcp
```

`{name}` is the configured MCP server name shown in the UI. Aqueduct rejects `GET` and `DELETE` with
`405 Method Not Allowed`. It does not translate older handshake/session-based protocols into this
request model. See the specification's [versioning and compatibility rules](https://modelcontextprotocol.org/specification/2026-07-28/basic/versioning).

### Required headers

Compliant 2026-07-28 requests require the following headers, in addition to Aqueduct authentication:

| Header | Value |
| ------ | ----- |
| `Authorization` | `Bearer <your-aqueduct-token>` |
| `Content-Type` | `application/json` |
| `Accept` | `application/json, text/event-stream` |
| `MCP-Protocol-Version` | `2026-07-28`, matching the protocol version in `params._meta` |
| `Mcp-Method` | The JSON-RPC `method`, on every request |
| `Mcp-Name` | `params.name` for `tools/call` and `prompts/get`; `params.uri` for `resources/read` |

The specification requires `Mcp-Name` for exactly those three methods, not all methods that happen to
contain a name or URI.

For `tools/call`, clients MUST also mirror tool arguments designated by `x-mcp-header` annotations in
the tool's `inputSchema` into `Mcp-Param-*` headers, omitting headers for absent or null values.
Aqueduct forwards incoming protocol-version and `Mcp-Param-*` headers to the upstream server.
Header names are case-insensitive; values are case-sensitive. See the specification's
[request metadata and encoding rules](https://modelcontextprotocol.org/specification/2026-07-28/basic/transports/streamable-http#request-metadata),
including Base64 sentinel encoding for header-unsafe `Mcp-Name` and parameter values.

### Required request metadata

Every request MUST include the required protocol metadata inside **`params._meta`**, not a top-level
`_meta`. The required and optional fields are:

| Key | Requirement |
| --- | ----------- |
| `io.modelcontextprotocol/protocolVersion` | Required; MUST match `MCP-Protocol-Version` |
| `io.modelcontextprotocol/clientCapabilities` | Required; `{}` is valid when no client capabilities are advertised |
| `io.modelcontextprotocol/clientInfo` | Optional; clients SHOULD include their name and version unless configured otherwise |

The upstream server uses this metadata independently for each request.

### Gateway behavior and limitations

- A missing `Mcp-Method` header returns `400 Bad Request`.
- For compatibility, the gateway can supply an SDK-default protocol version when the header is absent
  and derive a missing `Mcp-Name` from the body for `tools/call`, `resources/read`, and `prompts/get`.
  These fallbacks do not remove the specification's client requirements to send those headers.
- The relay does not enforce all required body metadata or header/body consistency rules; upstream
  validation remains necessary. It does not negotiate protocol versions or automatically retry requests.
  Under the specification, an unsupported version is reported by the upstream server with HTTP `400`
  and JSON-RPC error `-32022` (`UnsupportedProtocolVersion`), listing its supported versions.
- Upstream SSE reads currently have a 300-second read-idle timeout. This is an Aqueduct limitation,
  not a protocol restriction on long-lived `subscriptions/listen` requests.

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

This example requires `httpx` and `httpx-sse` and supports both permitted response types, including
SSE framing, multiline data, and keep-alive comments. It assumes the upstream `echo` tool has no
`x-mcp-header` annotations.

```python
import json

import httpx
from httpx_sse import EventSource

url = "https://your-aqueduct-domain.com/mcp-servers/my-cool-server/mcp"
headers = {
    "Authorization": "Bearer YOUR_AQUEDUCT_TOKEN",
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "MCP-Protocol-Version": "2026-07-28",
    "Mcp-Method": "tools/call",
    "Mcp-Name": "echo",
}
body = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
        "name": "echo",
        "arguments": {"message": "hello"},
        "_meta": {
            "io.modelcontextprotocol/protocolVersion": "2026-07-28",
            "io.modelcontextprotocol/clientCapabilities": {},
            "io.modelcontextprotocol/clientInfo": {"name": "my-app", "version": "1.0"},
        },
    },
}

with httpx.stream("POST", url, json=body, headers=headers, timeout=300) as response:
    if response.is_error:
        response.read()
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type == "application/json":
        response.read()
        print(response.json())
    elif content_type == "text/event-stream":
        for event in EventSource(response).iter_sse():
            if not event.data:
                continue
            message = json.loads(event.data)
            print(message)
            if message.get("id") == body["id"] and (
                "result" in message or "error" in message
            ):
                break
        else:
            raise RuntimeError("SSE stream ended before the JSON-RPC response")
    else:
        raise RuntimeError(f"Unexpected response content type: {content_type}")
```

This example displays the JSON-RPC result or error; it does not implement multi-round-trip input
handling or automatic protocol-version negotiation.

## Specification

- [Streamable HTTP (2026-07-28)](https://modelcontextprotocol.org/specification/2026-07-28/basic/transports/streamable-http)
- [Base protocol and per-request metadata](https://modelcontextprotocol.org/specification/2026-07-28/basic/index#meta)
- [Versioning and compatibility](https://modelcontextprotocol.org/specification/2026-07-28/basic/versioning)
