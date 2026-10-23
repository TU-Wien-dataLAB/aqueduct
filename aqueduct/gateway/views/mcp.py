import json
import logging
from collections.abc import AsyncGenerator, Mapping
from typing import Any

import httpx
from django.core.handlers.asgi import ASGIRequest
from django.http import HttpResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from mcp.shared.inbound import encode_header_value
from mcp.types import LATEST_PROTOCOL_VERSION, JSONRPCMessage, jsonrpc_message_adapter

from gateway.config import get_mcp_config
from gateway.decorators import (
    check_mcp_server_availability,
    log_request,
    mcp_transport_security,
    parse_body,
    parse_jsonrpc_message,
    token_authenticated,
    tos_accepted,
)
from gateway.response_error import error_response
from gateway.response_type import RawJsonResponse

log = logging.getLogger("aqueduct")

# Streamable HTTP transport headers for the stateless 2026-07-28 protocol core.
# See: https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http
MCP_PROTOCOL_VERSION_HEADER = "MCP-Protocol-Version"
MCP_METHOD_HEADER = "Mcp-Method"
MCP_NAME_HEADER = "Mcp-Name"
CONTENT_TYPE = "content-type"
ACCEPT = "accept"
JSON = "application/json"
SSE = "text/event-stream"

_NAME_BEARING_METHODS = {"tools/call", "prompts/get", "resources/read"}


def _extract_name(message: JSONRPCMessage) -> str | None:
    method = getattr(message, "method", None)
    if method not in _NAME_BEARING_METHODS:
        return None
    params = getattr(message, "params", None)
    if not isinstance(params, dict):
        return None
    name = params.get("uri" if method == "resources/read" else "name")
    return name if isinstance(name, str) else None


def _relay_headers(message: JSONRPCMessage, request_headers: Mapping[str, str]) -> dict[str, str]:
    method_header = request_headers.get(MCP_METHOD_HEADER)
    name_header = request_headers.get(MCP_NAME_HEADER)
    headers = {
        ACCEPT: f"{JSON}, {SSE}",
        CONTENT_TYPE: JSON,
        MCP_PROTOCOL_VERSION_HEADER: request_headers.get(
            MCP_PROTOCOL_VERSION_HEADER, LATEST_PROTOCOL_VERSION
        ),
        MCP_METHOD_HEADER: method_header or getattr(message, "method", "") or "",
    }
    if name_header:
        headers[MCP_NAME_HEADER] = name_header
    elif (name := _extract_name(message)) is not None:
        headers[MCP_NAME_HEADER] = encode_header_value(name)
    # Custom tool-parameter headers are opaque to the gateway (including Base64 values).
    headers.update(
        (key, value)
        for key, value in request_headers.items()
        if key.lower().startswith("mcp-param-")
    )
    return headers


async def _relay(
    url: str, message: JSONRPCMessage, request_headers: Mapping[str, str]
) -> RawJsonResponse | StreamingHttpResponse | HttpResponse:
    body = message.model_dump_json(exclude_none=True)
    headers = _relay_headers(message, request_headers)
    client = httpx.AsyncClient(timeout=httpx.Timeout(30, read=300))
    request = httpx.Request("POST", url, content=body, headers=headers)
    try:
        response = await client.send(request, stream=True)
    except BaseException:
        await client.aclose()
        raise

    content_type = response.headers.get(CONTENT_TYPE, "")
    if SSE in content_type.lower():

        async def sse_stream() -> AsyncGenerator[bytes, None]:
            try:
                async for chunk in response.aiter_bytes():
                    yield chunk
            finally:
                try:
                    await response.aclose()
                finally:
                    await client.aclose()

        return StreamingHttpResponse(sse_stream(), content_type=SSE, status=response.status_code)

    try:
        raw = await response.aread()
    finally:
        try:
            await response.aclose()
        finally:
            await client.aclose()
    try:
        payload = json.loads(raw)
        return RawJsonResponse(payload, status=response.status_code)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return HttpResponse(raw, content_type=content_type or JSON, status=response.status_code)


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
    request: ASGIRequest,
    name: str | None,
    json_rpc_message: JSONRPCMessage | None = None,
    *args: Any,
    **kwargs: Any,
) -> RawJsonResponse | StreamingHttpResponse | HttpResponse:
    if name is None:
        return error_response("Server name is required", status=400)

    method_header = request.headers.get(MCP_METHOD_HEADER)
    if not method_header:
        log.error("Missing %s header for MCP server '%s'", MCP_METHOD_HEADER, name)
        return error_response(f"Missing {MCP_METHOD_HEADER} header", status=400)

    if json_rpc_message is None:
        return error_response("Missing JSON-RPC message", status=400)

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
    return await _relay(url, json_rpc_message, request.headers)
