import logging
from functools import wraps
from typing import TYPE_CHECKING, Any

from asgiref.sync import sync_to_async
from django.conf import settings
from django.core.handlers.asgi import ASGIRequest
from django.urls import reverse
from openai.types.responses import ResponseCreateParams, ToolParam

from gateway.config import MCPServerConfig, get_mcp_config
from gateway.decorators.types import AsyncView, ViewResult
from gateway.raw_response import error_response, get_response_from_cache
from management.models import Token, VectorStore

if TYPE_CHECKING:
    from collections.abc import Iterable

log = logging.getLogger("aqueduct")


def validate_response_id(view_func: AsyncView) -> AsyncView:
    @wraps(view_func)
    async def wrapper(
        request: ASGIRequest, response_id: str, *args: Any, **kwargs: Any
    ) -> ViewResult:
        token = kwargs.get("token")
        if not token:
            log.error("validate_response_id decorator used without @token_authenticated decorator")
            return error_response("Internal server error", status=500)

        response = get_response_from_cache(response_id)
        if not response:
            return error_response("Response not found", status=404)

        if response["email"] != token.user.email:
            return error_response("Response not found", status=404)

        return await view_func(request, response_id, *args, **kwargs)

    return wrapper


async def _validate_mcp_tool(
    request: ASGIRequest, token: Token, tool: ToolParam
) -> ViewResult | None:
    # Note: mypy doesn't recognise the types of `ToolParam` attributes correctly
    server_name: str = tool.get("server_label")  # type: ignore[assignment]
    if await sync_to_async(token.mcp_server_excluded)(server_name):
        log.error("MCP server not found - %s", server_name)
        return error_response(f"MCP server not found - {server_name}", status=404)

    server_url = tool.get("server_url")
    if not server_url:
        if not settings.RESPONSES_API_ALLOW_EXTERNAL_MCP_SERVERS:
            log.error("MCP server not found - %s", server_name)
            return error_response(f"MCP server not found - {server_name}", status=404)
        return None

    mcp_config = get_mcp_config()
    server_config: MCPServerConfig | None = None
    for config_name, config in mcp_config.items():
        expected = request.build_absolute_uri(
            reverse("gateway:mcp_server", kwargs={"name": config_name})
        )
        if server_url == expected:
            server_config = config
            break

    if not server_config:
        if not settings.RESPONSES_API_ALLOW_EXTERNAL_MCP_SERVERS:
            log.error("MCP server not found - %s", server_name)
            return error_response(f"MCP server not found - {server_name}", status=404)
        return None

    tool["server_url"] = server_config["url"]  # type: ignore[typeddict-unknown-key]
    return None


async def _validate_file_search_tool(token: Token, tool: ToolParam) -> ViewResult | None:
    # Note: mypy doesn't recognise the types of `ToolParam` attributes correctly
    vector_store_ids: list[str] = tool.get("vector_store_ids", [])  # type: ignore[assignment]
    if not vector_store_ids:
        return None

    # Deduplicate to avoid false negatives in count check
    unique_vs_ids = list(set(vector_store_ids))
    # Verify ownership - users can only use their own vector stores
    if token.service_account:
        vs_count = await VectorStore.objects.filter(
            id__in=unique_vs_ids, token__service_account__team=token.service_account.team
        ).acount()
    else:
        vs_count = await VectorStore.objects.filter(
            id__in=unique_vs_ids, token__user=token.user
        ).acount()

    if vs_count != len(unique_vs_ids):
        return error_response("One or more vector stores not found", status=404)
    return None


def _validate_native_tool(tool_type: str | None) -> ViewResult | None:
    if tool_type not in settings.RESPONSES_API_ALLOWED_NATIVE_TOOLS:
        return error_response(f"Invalid tool type: {tool_type}", status=400)
    return None


async def _validate_tool(request: ASGIRequest, token: Token, tool: ToolParam) -> ViewResult | None:
    tool_type = tool.get("type")

    if tool_type in {"function", "custom"}:
        return None

    if tool_type == "mcp":
        return await _validate_mcp_tool(request, token, tool)

    if tool_type == "file_search":
        return await _validate_file_search_tool(token, tool)

    return _validate_native_tool(tool_type)


def check_tool_availability(view_func: AsyncView) -> AsyncView:
    """
    Validate tool availability and configuration for Responses API requests.

    Checks that MCP server tools are accessible to the user's token and properly
    configured. Validates server URLs for Aqueduct-managed MCP servers and ensures
    native tools are allowed in settings. It prevents unauthorized access
    to MCP servers and ensures tools are correctly configured.

    Used on Responses API endpoints that accept tools in the request body.
    Requires @token_authenticated and @parse_body decorators.
    """

    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args: Any, **kwargs: Any) -> ViewResult:
        token: Token | None = kwargs.get("token")
        pydantic_model: ResponseCreateParams | None = kwargs.get("pydantic_model")
        if not token or not pydantic_model:
            return error_response("Invalid request", status=400)

        tools: Iterable[ToolParam] = pydantic_model.get("tools") or []
        for tool in tools:
            error = await _validate_tool(request, token, tool)
            if error:
                return error

        return await view_func(request, *args, **kwargs)

    return wrapper
