import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from django.core.handlers.asgi import ASGIRequest
from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from mcp.types import LATEST_PROTOCOL_VERSION, jsonrpc_message_adapter

from gateway.config import get_mcp_config
from gateway.views.decorators import (
    check_mcp_server_availability,
    log_request,
    mcp_transport_security,
    parse_body,
    parse_jsonrpc_message,
    token_authenticated,
    tos_accepted,
)
from gateway.views.errors import error_response

log = logging.getLogger("aqueduct")

# Streamable HTTP transport headers for the stateless 2026-07-28 protocol core.
# See: https://modelcontextprotocol.io/specification/2026-07-28/specification/basic/transports
MCP_PROTOCOL_VERSION_HEADER = "MCP-Protocol-Version"
MCP_METHOD_HEADER = "Mcp-Method"
MCP_NAME_HEADER = "Mcp-Name"
CONTENT_TYPE = "content-type"
ACCEPT = "accept"
JSON = "application/json"
SSE = "text/event-stream"

# Name-bearing methods carry their tool/resource/prompt name in params; these get an
# `Mcp-Name` header. Other methods (e.g. tools/list) only need `Mcp-Method`.
_NAME_BEARING_METHODS = {
    "tools/call",
    "tools/get",
    "prompts/get",
    "resources/read",
    "resources/templates/list",
    "completion/complete",
    "tasks/get",
    "tasks/list",
}


def _extract_name(message: Any) -> str | None:
    """Return the RPC name (tool/resource/prompt) for the Mcp-Name header, if any."""
    method = getattr(message, "method", None)
    if method not in _NAME_BEARING_METHODS:
        return None
    params = getattr(message, "params", None)
    if not isinstance(params, dict):
        return None
    name = params.get("name")
    return name if isinstance(name, str) else None


def _relay_headers(
    method_header: str | None, name_header: str | None, message: Any
) -> dict[str, str]:
    """Build the 2026-07-28 required transport headers for the upstream request.

    The gateway always advertises the latest protocol version and the JSON-RPC
    method, mirroring the client's ``Mcp-Method``/``Mcp-Name`` headers when sent and
    otherwise deriving them from the message body. This makes every relayed request
    self-describing so it can land on any instance behind a round-robin load balancer.
    """
    headers = {
        ACCEPT: f"{JSON}, {SSE}",
        CONTENT_TYPE: JSON,
        MCP_PROTOCOL_VERSION_HEADER: LATEST_PROTOCOL_VERSION,
        MCP_METHOD_HEADER: method_header or getattr(message, "method", "") or "",
    }
    name = name_header or _extract_name(message)
    if name:
        headers[MCP_NAME_HEADER] = name
    return headers


async def _relay(
    url: str, message: Any, method_header: str | None, name_header: str | None
) -> HttpResponse | StreamingHttpResponse:
    """Relay a single stateless JSON-RPC message to an upstream MCP server.

    Each request is fully independent: a fresh client, no session, no held-open
    stream. The upstream response is returned as-is (JSON or a streamed SSE body).
    """
    body = message.model_dump_json(exclude_none=True)
    headers = _relay_headers(method_header, name_header, message)
    client = httpx.AsyncClient(timeout=httpx.Timeout(30, read=300))
    request = httpx.Request("POST", url, content=body, headers=headers)
    response = await client.send(request, stream=True)

    content_type = response.headers.get(CONTENT_TYPE, "")
    if SSE in content_type.lower():
        # Stream the upstream SSE body through without buffering.
        async def sse_stream() -> AsyncGenerator[bytes, None]:
            try:
                async for chunk in response.aiter_bytes():
                    yield chunk
            finally:
                await client.aclose()

        return StreamingHttpResponse(sse_stream(), content_type=SSE, status=response.status_code)

    raw = await response.aread()
    await client.aclose()
    try:
        payload = json.loads(raw)
        return JsonResponse(payload, status=response.status_code)
    except (json.JSONDecodeError, UnicodeDecodeError):
        # Upstream returned non-JSON; pass the raw body through.
        return HttpResponse(raw, content_type=content_type or JSON, status=response.status_code)


async def handle_post_request(
    name: str, json_rpc_message: Any, method_header: str | None, name_header: str | None
) -> HttpResponse | StreamingHttpResponse:
    """Relay a stateless MCP request to the configured upstream server."""
    try:
        mcp_config = get_mcp_config()
        server_config = mcp_config[name]
        url = server_config["url"]
    except KeyError:
        log.exception("MCP server '%s' not found", name)
        return error_response(f"MCP server '{name}' not found", status=404)
    except RuntimeError as e:
        log.exception("Unable to load MCP config: %s", e)
        return error_response(f"MCP server config unavailable: {e!s}", status=503)

    log.info(
        "MCP relay POST - Server: '%s', Method: %s", name, getattr(json_rpc_message, "method", None)
    )
    return await _relay(url, json_rpc_message, method_header, name_header)


@csrf_exempt
@require_http_methods(["POST"])
@token_authenticated(token_auth_only=True)
@tos_accepted
@parse_body(model=jsonrpc_message_adapter)
@parse_jsonrpc_message
@check_mcp_server_availability
@mcp_transport_security
@log_request
async def mcp_server(
    request: ASGIRequest, name: str | None, json_rpc_message: Any = None, *args: Any, **kwargs: Any
) -> HttpResponse | StreamingHttpResponse:
    """
    Stateless MCP endpoint.

    Each POST is a self-contained JSON-RPC exchange: the gateway requires the
    2026-07-28 ``Mcp-Method`` header, then relays the request to the configured
    upstream server and returns its response. There is no session, no GET stream,
    and no DELETE endpoint.
    """
    if name is None:
        return error_response("Server name is required", status=400)

    method_header = request.headers.get(MCP_METHOD_HEADER)
    if not method_header:
        log.error("Missing %s header for MCP server '%s'", MCP_METHOD_HEADER, name)
        return error_response(f"Missing {MCP_METHOD_HEADER} header", status=400)

    name_header = request.headers.get(MCP_NAME_HEADER)
    return await handle_post_request(
        name,
        json_rpc_message=json_rpc_message,
        method_header=method_header,
        name_header=name_header,
    )
