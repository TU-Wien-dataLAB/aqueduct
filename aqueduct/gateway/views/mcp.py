import asyncio
import threading
import uuid
from typing import AsyncGenerator, Dict, Optional

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from django.core.handlers.asgi import ASGIRequest
from django.http import JsonResponse, StreamingHttpResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCMessage

from ..config import get_mcp_config
from .decorators import parse_jsonrpc_message


class ManagedMCPSession:
    """MCP session that stays alive via manually-managed context manager."""

    def __init__(self, session_id: str, url: str, headers: dict | None = None):
        self.session_id = session_id
        self.url = url
        self.headers = headers

        self._context = streamablehttp_client(url=url, headers=headers)

        self.read_stream: MemoryObjectReceiveStream[SessionMessage | Exception] | None = None
        self.write_stream: MemoryObjectSendStream[SessionMessage] | None = None
        self.get_session_id_callback = None
        self.terminated = False
        self.created_at = timezone.now()
        self.last_accessed_at = timezone.now()

    async def start(self):
        """Start the session by entering the context manager."""
        streams = await self._context.__aenter__()
        self.read_stream, self.write_stream, self.get_session_id_callback = streams

    async def stop(self):
        """Stop the session by exiting the context manager."""
        if self.terminated:
            return
        self.terminated = True
        await self._context.__aexit__(None, None, None)

    async def send_message(self, message: SessionMessage):
        """Send a message to the MCP server."""
        if self.terminated:
            raise ValueError("Session terminated")
        await self.write_stream.send(message)
        self.last_accessed_at = timezone.now()

    async def receive_message(self) -> SessionMessage | Exception:
        """Receive a message from the MCP server."""
        if self.terminated:
            raise ValueError("Session terminated")
        message = await self.read_stream.receive()
        self.last_accessed_at = timezone.now()
        return message

    def get_mcp_session_id(self) -> str | None:
        """Get the MCP protocol session ID (not our internal session_id)."""
        if self.get_session_id_callback:
            return self.get_session_id_callback()
        return None


class MCPSessionManager:
    """Manages MCP sessions in a dedicated background thread."""

    def __init__(self):
        self._sessions: Dict[str, ManagedMCPSession] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._started = False

    def start(self):
        """Start the background event loop thread."""
        if self._started:
            return

        self._started = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        # Wait for loop to be ready
        while self._loop is None:
            threading.Event().wait(0.01)

    def _run_loop(self):
        """Run the asyncio event loop in background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        # Start cleanup task
        self._loop.create_task(self._cleanup_idle_sessions())

        # Run forever
        self._loop.run_forever()

    async def _cleanup_idle_sessions(self, max_idle_seconds: int = 3600):
        """Background task to cleanup idle sessions."""
        while True:
            await asyncio.sleep(60)

            now = timezone.now()
            to_remove = []

            for session_id, session in self._sessions.items():
                idle_seconds = (now - session.last_accessed_at).total_seconds()
                if idle_seconds > max_idle_seconds:
                    to_remove.append(session_id)

            for session_id in to_remove:
                session = self._sessions.pop(session_id, None)
                if session:
                    await session.stop()

    def create_session(self, url: str, headers: dict | None = None) -> str:
        """Create a new MCP session (called from Django view)."""
        future = asyncio.run_coroutine_threadsafe(
            self._create_session_impl(url, headers), self._loop
        )
        return future.result(timeout=10)

    async def _create_session_impl(self, url: str, headers: dict | None) -> str:
        """Actually create the session (runs in background thread)."""
        session_id = str(uuid.uuid4())
        session = ManagedMCPSession(session_id, url, headers)

        await session.start()

        self._sessions[session_id] = session
        return session_id

    def get_session(self, session_id: str) -> Optional[ManagedMCPSession]:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def terminate_session(self, session_id: str):
        """Terminate a session (called from Django view)."""
        future = asyncio.run_coroutine_threadsafe(
            self._terminate_session_impl(session_id), self._loop
        )
        return future.result(timeout=10)

    async def _terminate_session_impl(self, session_id: str):
        """Actually terminate the session (runs in background thread)."""
        session = self._sessions.pop(session_id, None)
        if session:
            await session.stop()

    def send_message(self, session_id: str, message: SessionMessage):
        """Send message to session (called from Django view)."""
        future = asyncio.run_coroutine_threadsafe(
            self._send_message_impl(session_id, message), self._loop
        )
        return future.result(timeout=30)

    async def _send_message_impl(self, session_id: str, message: SessionMessage):
        """Actually send the message (runs in background thread)."""
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        await session.send_message(message)

    def receive_message(self, session_id: str):
        """Receive message from session (called from Django view)."""
        future = asyncio.run_coroutine_threadsafe(
            self._receive_message_impl(session_id), self._loop
        )
        return future.result(timeout=30)

    async def _receive_message_impl(self, session_id: str):
        """Actually receive the message (runs in background thread)."""
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        return await session.receive_message()

    def shutdown_all(self):
        """Shutdown all sessions (called on app shutdown)."""
        if not self._loop:
            return

        future = asyncio.run_coroutine_threadsafe(self._shutdown_all_impl(), self._loop)
        future.result(timeout=30)

        # Stop the event loop
        self._loop.call_soon_threadsafe(self._loop.stop)

    async def _shutdown_all_impl(self):
        """Actually shutdown all sessions."""
        for session in self._sessions.values():
            await session.stop()
        self._sessions.clear()


async def _mcp_sse_stream(session_id: str) -> AsyncGenerator[str, None]:
    """Async generator that streams MCP messages as Server-Sent Events."""
    try:
        while True:
            try:
                message = session_manager.receive_message(session_id)
                if isinstance(message, Exception):
                    # Send error event if we received an exception
                    yield f'data: {{"error": "{str(message)}"}}\n\n'
                    break
                else:
                    # Send the message as a JSON-RPC SSE event
                    message_json = message.message.model_dump_json(exclude_none=True)
                    yield f"data: {message_json}\n\n"
            except ValueError as e:
                if "terminated" in str(e).lower() or "not found" in str(e).lower():
                    # Session terminated or not found, end the stream
                    yield f'data: {{"event": "session_ended", "reason": "{str(e)}"}}\n\n'
                    break
                else:
                    # Other ValueError, send error and continue
                    yield f'data: {{"error": "{str(e)}"}}\n\n'
            except Exception as e:
                # Unexpected error, send error and end stream
                yield f'data: {{"error": "{str(e)}", "event": "stream_error"}}\n\n'
                break

    except Exception as e:
        # Top-level error, send final error
        yield f'data: {{"error": "Stream failed: {str(e)}", "event": "stream_ended"}}\n\n'


# Global session manager - starts automatically
session_manager = MCPSessionManager()
session_manager.start()


async def handle_get_request(
    request: ASGIRequest, name: str
) -> JsonResponse | StreamingHttpResponse:
    """Create new session or return SSE stream."""
    session_id = request.headers.get("Mcp-Session-Id")

    if not session_id:
        # Create new session (initialization)
        try:
            mcp_config = get_mcp_config()
            server_config = mcp_config[name]
            url = server_config["url"]
        except (KeyError, RuntimeError) as e:
            return JsonResponse({"error": f"MCP server '{name}' not found: {str(e)}"}, status=404)

        new_session_id = session_manager.create_session(url)

        session = session_manager.get_session(new_session_id)
        mcp_session_id = session.get_mcp_session_id() if session else None

        response = JsonResponse({"session_id": new_session_id, "server_name": name})
        if mcp_session_id:
            response["Mcp-Session-Id"] = mcp_session_id

        return response
    else:
        # Return SSE stream for existing session
        session = session_manager.get_session(session_id)
        if not session:
            return JsonResponse({"error": "Session not found"}, status=404)

        if session.terminated:
            return JsonResponse({"error": "Session terminated"}, status=410)

        return StreamingHttpResponse(
            streaming_content=_mcp_sse_stream(session_id),
            content_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control, Mcp-Session-Id",
                "Mcp-Session-Id": session.get_mcp_session_id() or session_id,
            },
        )


@parse_jsonrpc_message
async def handle_post_request(
    request: ASGIRequest,
    name: str,
    json_rpc_message: JSONRPCMessage = None,
    session_id: str | None = None,
    is_initialize: bool = False,
) -> JsonResponse:
    """Send message to MCP session."""
    try:
        if is_initialize and not session_id:
            mcp_config = get_mcp_config()
            server_config = mcp_config[name]
            url = server_config["url"]
            session_id = session_manager.create_session(url)

        session = session_manager.get_session(session_id)
        if not session:
            return JsonResponse({"error": "Session not found"}, status=404)

        session_message = SessionMessage(json_rpc_message)
        session_manager.send_message(session_id, session_message)

        received_message = session_manager.receive_message(session_id)

        if isinstance(received_message, Exception):
            return JsonResponse({"error": str(received_message)}, status=500)

        response_data = received_message.message.model_dump(exclude_none=True)
        response = JsonResponse(response_data)

        if is_initialize:
            mcp_session_id = session.get_mcp_session_id()
            if mcp_session_id:
                response["Mcp-Session-Id"] = mcp_session_id

        return response
    except (KeyError, RuntimeError) as e:
        return JsonResponse({"error": f"MCP server '{name}' not found: {str(e)}"}, status=404)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)


async def handle_delete_request(request: ASGIRequest, name: str) -> JsonResponse:
    """Terminate MCP session."""
    session_id = request.headers.get("Mcp-Session-Id")
    if not session_id:
        return JsonResponse({"error": "Mcp-Session-Id header required"}, status=400)

    try:
        session_manager.terminate_session(session_id)
        return JsonResponse({"status": "session_terminated"})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)


@csrf_exempt
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
