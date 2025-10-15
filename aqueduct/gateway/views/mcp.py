import json

from django.core.handlers.asgi import ASGIRequest
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from mcp.server.streamable_http import EventStore


class StreamableHTTPSession:
    """Class representing a streamable HTTP session configuration."""

    def __init__(
        self,
        mcp_session_id: str | None = None,
        is_json_response_enabled: bool = False,
        event_store: EventStore | None = None,
    ):
        self.mcp_session_id = mcp_session_id
        self.is_json_response_enabled = is_json_response_enabled
        self.event_store = event_store
        self.terminated = False


async def handle_get_request(request: ASGIRequest, name: str) -> JsonResponse:
    """Handle GET requests to MCP server endpoint."""
    return JsonResponse(
        {
            "message": f"GET request to MCP server '{name}' processed successfully",
            "server_name": name,
            "method": "GET",
        }
    )


async def handle_post_request(request: ASGIRequest, name: str) -> JsonResponse:
    """Handle POST requests to MCP server endpoint."""
    try:
        data = json.loads(request.body) if request.body else {}
        return JsonResponse(
            {
                "message": f"POST request to MCP server '{name}' processed successfully",
                "server_name": name,
                "method": "POST",
                "received_data": data,
            }
        )
    except json.JSONDecodeError:
        return JsonResponse(
            {"error": "Invalid JSON in request body", "server_name": name, "method": "POST"},
            status=400,
        )


async def handle_delete_request(request: ASGIRequest, name: str) -> JsonResponse:
    """Handle DELETE requests to MCP server endpoint."""
    return JsonResponse(
        {
            "message": f"DELETE request to MCP server '{name}' processed successfully",
            "server_name": name,
            "method": "DELETE",
        }
    )


@require_http_methods(["GET", "POST", "DELETE"])
async def mcp_server(request: ASGIRequest, name):
    """
    Handles GET, POST and DELETE requests for /mcp-servers/{name}/mcp path.
    """
    if request.method == "GET":
        return await handle_get_request(request, name)
    elif request.method == "POST":
        return await handle_post_request(request, name)
    elif request.method == "DELETE":
        return await handle_delete_request(request, name)
