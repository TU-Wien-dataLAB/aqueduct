from django.core.handlers.asgi import ASGIRequest
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
import json

from mcp.server.streamable_http import StreamableHTTPServerTransport


@require_http_methods(["GET", "POST"])
async def mcp_server(request: ASGIRequest, name):
    """
    Dummy functional view for MCP server endpoint.
    Handles both GET and POST requests for /mcp-servers/{name}/mcp path.
    """
    server_transport = StreamableHTTPServerTransport(mcp_session_id=None)
    if request.method == "GET":
        # listen for messages from the server
        return JsonResponse({
            "message": f"GET request to MCP server '{name}' processed successfully",
            "server_name": name,
            "method": "GET"
        })
    elif request.method == "POST":
        # send messages to the server
        try:
            data = json.loads(request.body) if request.body else {}
            return JsonResponse({
                "message": f"POST request to MCP server '{name}' processed successfully",
                "server_name": name,
                "method": "POST",
                "received_data": data
            })
        except json.JSONDecodeError:
            return JsonResponse({
                "error": "Invalid JSON in request body",
                "server_name": name,
                "method": "POST"
            }, status=400)