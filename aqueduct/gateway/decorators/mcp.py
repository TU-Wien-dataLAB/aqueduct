import logging
from functools import wraps
from typing import TYPE_CHECKING, Any

from asgiref.sync import sync_to_async
from django.conf import settings
from django.core.handlers.asgi import ASGIRequest
from mcp.types import JSONRPCMessage

from gateway.decorators.types import AsyncView, ViewResult
from gateway.raw_response import error_response, in_wildcard

if TYPE_CHECKING:
    from management.models import Token

log = logging.getLogger("aqueduct")


def check_mcp_server_availability(view_func: AsyncView) -> AsyncView:
    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args: Any, **kwargs: Any) -> ViewResult:
        token: Token | None = kwargs.get("token")
        if not token:
            log.error(
                "check_mcp_server_availability decorator used without "
                "@token_authenticated decorator"
            )
            return error_response("Internal server error", status=500)
        server_name: str | None = kwargs.get("name")
        if not server_name:
            return await view_func(request, *args, **kwargs)
        if await sync_to_async(token.mcp_server_excluded)(server_name):
            log.error("MCP server not found - %s", server_name)
            return error_response("MCP server not found!", status=404)
        return await view_func(request, *args, **kwargs)

    return wrapper


def mcp_transport_security(view_func: AsyncView) -> AsyncView:
    """Validate MCP transport security (DNS rebinding protection).

    Validates:
    - Host header (DNS rebinding protection)
    - Origin header (CSRF protection)
    - Content-Type header for POST requests

    Returns appropriate status codes:
    - 421: Invalid Host header
    - 403: Invalid Origin header
    - 400: Invalid Content-Type header
    """

    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args: Any, **kwargs: Any) -> ViewResult:
        # Skip validation if DNS rebinding protection is disabled
        if not getattr(settings, "MCP_ENABLE_DNS_REBINDING_PROTECTION", True):
            return await view_func(request, *args, **kwargs)

        log.debug("MCP request headers: %s", dict(request.headers))

        # Validate Content-Type for POST requests
        if request.method == "POST":
            content_type = request.headers.get("content-type", "")
            log.debug("POST request Content-Type: %r", content_type)
            if not content_type.lower().startswith("application/json"):
                log.error("Invalid Content-Type header: %s", content_type)
                return error_response("Invalid Content-Type header", status=400)

        # Validate Host header against allowed values
        allowed_hosts = getattr(settings, "MCP_ALLOWED_HOSTS", [])
        host = request.headers.get("host")

        if not host:
            log.error("Missing Host header in request")
            return error_response("Invalid Host header", status=421)

        host_valid = in_wildcard(host, allowed_hosts)
        if not host_valid:
            log.error("Invalid Host header: %s", host)
            return error_response("Invalid Host header", status=421)

        # Validate Origin header against allowed values
        # Origin can be absent for same-origin requests, so it's only validated if present
        allowed_origins = getattr(settings, "MCP_ALLOWED_ORIGINS", [])
        origin = request.headers.get("origin")
        if origin:
            origin_valid = in_wildcard(origin, allowed_origins)
            if not origin_valid:
                log.error("Invalid Origin header: %s", origin)
                return error_response("Invalid Origin header", status=403)

        return await view_func(request, *args, **kwargs)

    return wrapper


def parse_jsonrpc_message(view_func: AsyncView) -> AsyncView:
    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args: Any, **kwargs: Any) -> ViewResult:
        session_id = request.headers.get("mcp-session-id")
        kwargs["session_id"] = session_id

        if request.method != "POST":
            if not session_id:
                log.error("Session ID required for MCP server %r", kwargs.get("name"))
                return error_response("Mcp-Session-Id header required", status=400)

            return await view_func(request, *args, request_log=None, **kwargs)

        data = kwargs["pydantic_model"]
        # For mcp requests, timeout should not be passed to the JSON RPC Message
        data.pop("timeout", None)
        json_rpc_message = JSONRPCMessage.model_validate(data)
        is_initialize = (
            hasattr(json_rpc_message.root, "method")
            and json_rpc_message.root.method == "initialize"
        )

        if not is_initialize and not session_id:
            log.error("Session ID required for MCP server %r", kwargs.get("name"))
            return error_response("Mcp-Session-Id header required", status=400)

        kwargs["json_rpc_message"] = json_rpc_message
        kwargs["is_initialize"] = is_initialize

        return await view_func(request, *args, **kwargs)

    return wrapper
