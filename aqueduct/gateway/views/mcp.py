import asyncio
import threading
import uuid
from typing import AsyncGenerator, Dict, Optional

import anyio
import httpx
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from django.core.handlers.asgi import ASGIRequest
from django.http import JsonResponse, StreamingHttpResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from mcp.client.streamable_http import StreamableHTTPTransport
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCMessage

from ..config import get_mcp_config
from .decorators import parse_jsonrpc_message


class SimpleTaskGroup:
    """Minimal task group implementation
    that provides task spawning without cancel scopes.

    This class exists to solve a specific problem with the MCP SDK's StreamableHTTPTransport:
    - The SDK's post_writer method requires a TaskGroup object with start_soon() and cancel_scope
    - The official anyio.create_task_group() creates cancel scopes tied to specific tasks
    - When the context is entered in one task and exited in another, anyio raises:
      RuntimeError: "Attempted to exit cancel scope in a different task than it was entered in"

    In our architecture, sessions are managed across different asyncio tasks via
    asyncio.run_coroutine_threadsafe(), so we cannot use anyio's task groups directly.

    This implementation provides the minimal interface that StreamableHTTPTransport needs:
    - start_soon(): Spawn concurrent tasks for handling JSONRPCRequests
    - cancel_scope: Mock cancel scope for compatibility (SDK calls .cancel() during cleanup)

    The actual cleanup happens in ManagedMCPSession.stop() by closing streams and awaiting tasks,
    so we don't need real cancel scope functionality here.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._tasks = []

    def start_soon(self, coro_func, *args):
        """Start a coroutine as a task."""
        task = self._loop.create_task(coro_func(*args))
        self._tasks.append(task)
        return task

    async def cancel_all(self):
        """Cancel all spawned tasks."""
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    @property
    def cancel_scope(self):
        """Mock cancel scope for compatibility."""
        return MockCancelScope()


class MockCancelScope:
    """Mock cancel scope for compatibility."""

    def cancel(self):
        """Mock cancel - does nothing since we don't have cancel scopes."""
        pass


class ManagedMCPSession:
    """MCP session that uses StreamableHTTPTransport directly without cancel scopes."""

    def __init__(self, session_id: str, url: str, headers: dict | None = None):
        self.session_id = session_id
        self.url = url
        self.headers = headers

        # Create transport directly (no context manager)
        self.transport = StreamableHTTPTransport(url=url, headers=headers)

        # Streams that transport will use
        self.read_stream: MemoryObjectReceiveStream[SessionMessage | Exception] | None = None
        self.write_stream: MemoryObjectSendStream[SessionMessage] | None = None

        # Internal streams and client
        self._read_stream_writer = None
        self._write_stream_reader = None
        self._httpx_client = None
        self._post_writer_task = None
        self._get_stream_task = None
        self._task_group = None

        self.terminated = False
        self.created_at = timezone.now()
        self.last_accessed_at = timezone.now()

    async def start(self):
        """Start the session by setting up streams and tasks."""
        # Create memory streams (exactly like streamablehttp_client does)
        self._read_stream_writer, self.read_stream = anyio.create_memory_object_stream[
            SessionMessage | Exception
        ](0)
        self.write_stream, self._write_stream_reader = anyio.create_memory_object_stream[
            SessionMessage
        ](0)

        # Create httpx client
        self._httpx_client = httpx.AsyncClient(
            headers=self.transport.request_headers,
            timeout=httpx.Timeout(self.transport.timeout, read=self.transport.sse_read_timeout),
        )

        # Create simple task group
        self._task_group = SimpleTaskGroup(asyncio.get_event_loop())

        # Start post_writer task (this is what handles all the logic)
        def start_get_stream_task():
            if self._httpx_client and self._read_stream_writer:
                self._get_stream_task = asyncio.create_task(
                    self.transport.handle_get_stream(self._httpx_client, self._read_stream_writer)  # type: ignore
                )

        self._post_writer_task = asyncio.create_task(
            self.transport.post_writer(
                self._httpx_client,  # type: ignore
                self._write_stream_reader,  # type: ignore
                self._read_stream_writer,  # type: ignore
                self.write_stream,
                start_get_stream_task,
                self._task_group,  # type: ignore - Pass our simple task group
            )
        )

    async def stop(self):
        """Stop the session cleanly."""
        if self.terminated:
            return
        self.terminated = True

        # Terminate session via DELETE request
        if self.transport.session_id and self._httpx_client:
            await self.transport.terminate_session(self._httpx_client)

        # Close streams (this will cause post_writer to exit)
        if self._read_stream_writer:
            await self._read_stream_writer.aclose()
        if self.write_stream:
            await self.write_stream.aclose()

        # Wait for tasks to complete
        if self._post_writer_task and not self._post_writer_task.done():
            await self._post_writer_task
        if self._get_stream_task and not self._get_stream_task.done():
            self._get_stream_task.cancel()
            try:
                await self._get_stream_task
            except asyncio.CancelledError:
                pass

        # Cancel any remaining tasks in task group
        if self._task_group:
            await self._task_group.cancel_all()

        # Close httpx client
        if self._httpx_client:
            await self._httpx_client.aclose()

    async def send_message(self, message: SessionMessage):
        """Send a message to the MCP server."""
        if self.terminated:
            raise ValueError("Session terminated")
        if not self.write_stream:
            raise ValueError("Session not started")
        await self.write_stream.send(message)
        self.last_accessed_at = timezone.now()

    async def receive_message(self) -> SessionMessage | Exception:
        """Receive a message from the MCP server."""
        if self.terminated:
            raise ValueError("Session terminated")
        if not self.read_stream:
            raise ValueError("Session not started")
        message = await self.read_stream.receive()
        self.last_accessed_at = timezone.now()
        return message

    def get_mcp_session_id(self) -> str | None:
        """Get the MCP protocol session ID (not our internal session_id)."""
        return self.transport.get_session_id()


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
        if not self._loop:
            raise RuntimeError("Session manager not started")
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
        if not self._loop:
            raise RuntimeError("Session manager not started")
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
        if not self._loop:
            raise RuntimeError("Session manager not started")
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
        if not self._loop:
            raise RuntimeError("Session manager not started")
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
    if not session_id:
        yield 'data: {"error": "Session ID required"}\n\n'
        return

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
    request: ASGIRequest, name: str, session_id: str | None = None
) -> JsonResponse | StreamingHttpResponse:
    """Create new session or return SSE stream."""

    # If no session_id, create a new session
    if not session_id:
        mcp_config = get_mcp_config()
        server_config = mcp_config[name]
        url = server_config["url"]
        session_id = session_manager.create_session(url)
        return JsonResponse(
            {"session_id": session_id, "server_name": name, "status": "session_created"}
        )

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
        },
    )


async def handle_post_request(
    request: ASGIRequest,
    name: str,
    json_rpc_message: JSONRPCMessage | None,
    session_id: str | None,
    is_initialize: bool,
) -> JsonResponse:
    """Send message to MCP session."""
    if json_rpc_message is None:
        return JsonResponse({"error": "Missing JSON-RPC message"}, status=400)

    print("Request:", session_id, json_rpc_message.model_dump_json(exclude_none=True))

    try:
        if is_initialize and not session_id:
            mcp_config = get_mcp_config()
            server_config = mcp_config[name]
            url = server_config["url"]
            session_id = session_manager.create_session(url)

        if not session_id:
            return JsonResponse({"error": "Mcp-Session-Id header required"}, status=400)

        session = session_manager.get_session(session_id)
        if not session:
            return JsonResponse({"error": "Session not found"}, status=404)

        session_message = SessionMessage(json_rpc_message)
        session_manager.send_message(session_id, session_message)

        received_message = session_manager.receive_message(session_id)

        if isinstance(received_message, Exception):
            return JsonResponse({"error": str(received_message)}, status=500)

        response_data = received_message.message.model_dump(exclude_none=True)
        print(response_data)
        response = JsonResponse(response_data)
        if session_id:
            response.headers["mcp-session-id"] = session_id
        return response
    except (KeyError, RuntimeError) as e:
        return JsonResponse({"error": f"MCP server '{name}' not found: {str(e)}"}, status=404)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)


async def handle_delete_request(
    request: ASGIRequest, name: str, session_id: str | None = None
) -> JsonResponse:
    """Terminate MCP session."""
    if not session_id:
        return JsonResponse({"error": "Mcp-Session-Id header required"}, status=400)

    try:
        session_manager.terminate_session(session_id)
        return JsonResponse({"status": "session_terminated"})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)


@csrf_exempt
@parse_jsonrpc_message
@require_http_methods(["GET", "POST", "DELETE"])
async def mcp_server(
    request: ASGIRequest,
    name,
    json_rpc_message: JSONRPCMessage | None = None,
    session_id: str | None = None,
    is_initialize: bool = False,
):
    """
    Handles GET, POST and DELETE requests for /mcp-servers/{name}/mcp path.
    """
    if request.method == "GET":
        return await handle_get_request(request, name, session_id=session_id)
    elif request.method == "POST":
        return await handle_post_request(
            request,
            name,
            json_rpc_message=json_rpc_message,
            session_id=session_id,
            is_initialize=is_initialize,
        )
    elif request.method == "DELETE":
        return await handle_delete_request(request, name, session_id)
