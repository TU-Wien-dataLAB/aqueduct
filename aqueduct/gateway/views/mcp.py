import asyncio
import json
import threading
import uuid
from typing import Dict, Optional

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from django.core.handlers.asgi import ASGIRequest
from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.http import require_http_methods
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCMessage

from ..config import get_mcp_config


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


# Global session manager - starts automatically
session_manager = MCPSessionManager()
session_manager.start()


async def handle_get_request(request: ASGIRequest, name: str) -> JsonResponse:
    """Create new session or return SSE stream."""
    session_id = request.headers.get("X-MCP-Session-ID")

    if not session_id:
        # Create new session
        try:
            mcp_config = get_mcp_config()
            server_config = mcp_config[name]
            url = server_config["url"]
        except (KeyError, RuntimeError) as e:
            return JsonResponse({"error": f"MCP server '{name}' not found: {str(e)}"}, status=404)

        new_session_id = session_manager.create_session(url)

        return JsonResponse({"session_id": new_session_id, "server_name": name})
    else:
        # TODO: Return SSE stream for existing session
        return JsonResponse({"error": "SSE streaming not yet implemented"}, status=501)


async def handle_post_request(request: ASGIRequest, name: str) -> JsonResponse:
    """Send message to MCP session."""
    session_id = request.headers.get("X-MCP-Session-ID")
    if not session_id:
        return JsonResponse({"error": "X-MCP-Session-ID header required"}, status=400)

    session = session_manager.get_session(session_id)
    if not session:
        return JsonResponse({"error": "Session not found"}, status=404)

    try:
        data = json.loads(request.body)
        json_rpc_message = JSONRPCMessage.model_validate(data)
        session_message = SessionMessage(json_rpc_message)

        session_manager.send_message(session_id, session_message)

        return JsonResponse({"status": "message_sent"})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)


async def handle_delete_request(request: ASGIRequest, name: str) -> JsonResponse:
    """Terminate MCP session."""
    session_id = request.headers.get("X-MCP-Session-ID")
    if not session_id:
        return JsonResponse({"error": "X-MCP-Session-ID header required"}, status=400)

    try:
        session_manager.terminate_session(session_id)
        return JsonResponse({"status": "session_terminated"})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)


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
