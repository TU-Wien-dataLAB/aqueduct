import asyncio
import logging
import uuid
from typing import AsyncGenerator, Dict

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
from mcp.types import JSONRPCMessage, JSONRPCNotification, JSONRPCRequest

from management.models import Request, Token

from ..config import get_mcp_config
from .decorators import log_request, parse_jsonrpc_message, token_authenticated

log = logging.getLogger("aqueduct")


class SimpleTaskGroup:
    """Minimal task group implementation that provides task spawning without cancel scopes.

    This class exists to solve a specific problem with the MCP SDK's StreamableHTTPTransport:
    - The SDK's post_writer method requires a TaskGroup object with start_soon() and cancel_scope
    - The official anyio.create_task_group() creates cancel scopes tied to specific async contexts
    - When the context is entered in one task and exited in another, anyio raises:
      RuntimeError: "Attempted to exit cancel scope in a different task than it was entered in"

    In our architecture, MCP sessions are long-lived and span multiple HTTP requests/responses.
    Each HTTP request runs in a separate async task, so we cannot use anyio's task groups which
    require the entire lifecycle (enter/exit) to occur within a single async context manager scope.

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
        self._read_stream_writer: MemoryObjectSendStream[SessionMessage | Exception] | None = None
        self._write_stream_reader: MemoryObjectReceiveStream[SessionMessage] | None = None
        self._httpx_client: httpx.AsyncClient | None = None
        self._post_writer_task: asyncio.Task | None = None
        self._get_stream_task: asyncio.Task | None = None
        self._task_group: SimpleTaskGroup | None = None

        self.terminated = False
        self.created_at = timezone.now()
        self.last_accessed_at = timezone.now()

    async def start(self):
        """Start the session by setting up streams and tasks."""
        log.info(f"Starting MCP session {self.session_id} for {self.url}")

        # Create memory streams (exactly like streamablehttp_client does)
        log.debug(f"{self.session_id}: Creating memory streams")
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
                log.debug(f"Starting GET stream task for session {self.session_id}")
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

        log.info(f"Stopping MCP session {self.session_id}")
        self.terminated = True

        # Terminate session via DELETE request
        if self.transport.session_id and self._httpx_client:
            log.debug(f"Terminating MCP protocol session {self.transport.session_id}")
            await self.transport.terminate_session(self._httpx_client)

        # Close streams (this will cause post_writer to exit)
        if self._read_stream_writer:
            log.debug(f"Closing read stream for session {self.session_id}")
            await self._read_stream_writer.aclose()
        if self.write_stream:
            log.debug(f"Closing write stream for session {self.session_id}")
            await self.write_stream.aclose()

        # Wait for tasks to complete
        if self._post_writer_task and not self._post_writer_task.done():
            log.debug(f"Waiting for post_writer task completion for session {self.session_id}")
            await self._post_writer_task
        if self._get_stream_task and not self._get_stream_task.done():
            log.debug(f"Cancelling get_stream task for session {self.session_id}")
            self._get_stream_task.cancel()
            try:
                await self._get_stream_task  # type: ignore
            except asyncio.CancelledError:
                pass

        # Cancel any remaining tasks in task group
        if self._task_group:
            log.debug(f"Cancelling all tasks in task group for session {self.session_id}")
            await self._task_group.cancel_all()

        # Close httpx client
        if self._httpx_client:
            log.debug(f"Closing httpx client for session {self.session_id}")
            await self._httpx_client.aclose()

        log.info(f"MCP session {self.session_id} stopped successfully")

    async def send_message(self, message: SessionMessage):
        """Send a message to the MCP server."""
        if self.terminated:
            raise ValueError("Session terminated")
        if not self.write_stream:
            raise ValueError("Session not started")

        message_type = message.message.root.__class__.__name__
        log.info(f"Sending: {message_type} to session {self.session_id}")
        await self.write_stream.send(message)
        log.debug(f"Message sent to stream in session {self.session_id}")
        self.last_accessed_at = timezone.now()

    async def receive_message(self) -> SessionMessage | Exception:
        """Receive a message from the MCP server."""
        if self.terminated:
            raise ValueError("Session terminated")
        if not self.read_stream:
            raise ValueError("Session not started")

        message = await self.read_stream.receive()
        self.last_accessed_at = timezone.now()

        if isinstance(message, Exception):
            log.warning(f"{self.session_id}: Received exception: {str(message)}")
        else:
            message_type = message.message.root.__class__.__name__
            log.info(f"{self.session_id}: Received {message_type}")

        return message


class MCPSessionManager:
    """Manages MCP sessions using Daphne's event loop."""

    def __init__(self):
        self._sessions: Dict[str, ManagedMCPSession] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task = None

    async def start(self):
        """Start the cleanup task."""
        if self._cleanup_task is None:
            log.info("Starting MCP session cleanup task")
            self._cleanup_task = asyncio.create_task(self._cleanup_idle_sessions())

    async def _cleanup_idle_sessions(self, max_idle_seconds: int = 3600):
        """Background task to cleanup idle sessions."""
        while True:
            await asyncio.sleep(60)

            now = timezone.now()
            to_remove = []

            async with self._lock:
                for session_id, session in self._sessions.items():
                    idle_seconds = (now - session.last_accessed_at).total_seconds()
                    if idle_seconds > max_idle_seconds:
                        to_remove.append(session_id)

            if to_remove:
                log.info(f"Cleaning up {len(to_remove)} idle sessions")
                for session_id in to_remove:
                    async with self._lock:
                        session = self._sessions.pop(session_id, None)
                    if session:
                        await session.stop()
                        log.debug(f"Cleaned up idle session {session_id}")

    async def create_session(self, url: str, headers: dict | None = None) -> str:
        """Create a new MCP session."""
        session_id = str(uuid.uuid4())
        log.info(f"Creating MCP session {session_id} for {url}")

        session = ManagedMCPSession(session_id, url, headers)
        await session.start()

        async with self._lock:
            self._sessions[session_id] = session

        log.info(f"MCP session {session_id} created and started")
        return session_id

    async def get_session(self, session_id: str) -> ManagedMCPSession | None:
        """Get a session by ID."""
        async with self._lock:
            return self._sessions.get(session_id)

    async def terminate_session(self, session_id: str):
        """Terminate a session."""
        log.info(f"Terminating session {session_id}")
        async with self._lock:
            session = self._sessions.pop(session_id, None)
        if session:
            await session.stop()
            log.info(f"Session {session_id} terminated")
        else:
            log.warning(f"Attempt to terminate non-existent session {session_id}")

    async def send_message(self, session_id: str, message: SessionMessage):
        """Send message to session."""
        session = await self.get_session(session_id)
        if not session:
            log.warning(f"Session {session_id} not found for send_message")
            raise ValueError(f"Session {session_id} not found")
        await session.send_message(message)

    async def receive_message(self, session_id: str):
        """Receive message from session."""
        session = await self.get_session(session_id)
        if not session:
            log.warning(f"Session {session_id} not found for receive_message")
            raise ValueError(f"Session {session_id} not found")
        return await session.receive_message()

    async def shutdown_all(self):
        """Shutdown all sessions."""
        log.info("Shutting down all MCP sessions")
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()

        for session in sessions:
            await session.stop()

        if self._cleanup_task:
            log.debug("Cancelling MCP session cleanup task")
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        log.info("All MCP sessions shutdown complete")


async def _mcp_sse_stream(session_id: str) -> AsyncGenerator[str, None]:
    """Async generator that streams MCP messages as Server-Sent Events."""
    if not session_id:
        log.warning("SSE stream missing session ID")
        # Create a proper JSONRPCNotification for the error
        error_msg = JSONRPCMessage(
            root=JSONRPCNotification(
                jsonrpc="2.0",
                method="stream_error",
                params={"code": -32000, "message": "Session ID required"},
            )
        )
        yield f"data: {error_msg.model_dump_json(exclude_none=True)}\n\n"
        return

    log.info(f"SSE stream started for session {session_id}")
    try:
        while True:
            try:
                message: SessionMessage | Exception = await session_manager.receive_message(
                    session_id
                )
                if isinstance(message, Exception):
                    log.error(
                        f"SSE stream for session {session_id} received exception: {str(message)}"
                    )
                    # Convert exception to JSONRPCNotification
                    error_msg = JSONRPCMessage(
                        root=JSONRPCNotification(
                            jsonrpc="2.0",
                            method="stream_error",
                            params={"code": -32001, "message": str(message)},
                        )
                    )
                    yield f"data: {error_msg.model_dump_json(exclude_none=True)}\n\n"
                    break
                else:
                    message_json = message.message.model_dump_json(exclude_none=True)
                    yield f"data: {message_json}\n\n"
            # TODO: handle timeout?
            except ValueError as e:
                if "terminated" in str(e).lower() or "not found" in str(e).lower():
                    log.info(f"SSE stream for session {session_id} ended: {str(e)}")
                    # Send proper session end notification
                    end_msg = JSONRPCMessage(
                        root=JSONRPCNotification(
                            jsonrpc="2.0", method="session_ended", params={"reason": str(e)}
                        )
                    )
                    yield f"data: {end_msg.model_dump_json(exclude_none=True)}\n\n"
                    break
                else:
                    log.error(f"SSE stream for session {session_id} ValueError: {str(e)}")
                    error_msg = JSONRPCMessage(
                        root=JSONRPCNotification(
                            jsonrpc="2.0",
                            method="stream_error",
                            params={"code": -32002, "message": str(e)},
                        )
                    )
                    yield f"data: {error_msg.model_dump_json(exclude_none=True)}\n\n"
            except Exception as e:
                log.error(f"SSE stream for session {session_id} unexpected error: {str(e)}")
                error_msg = JSONRPCMessage(
                    root=JSONRPCNotification(
                        jsonrpc="2.0",
                        method="stream_error",
                        params={"code": -32003, "message": str(e)},
                    )
                )
                yield f"data: {error_msg.model_dump_json(exclude_none=True)}\n\n"
                break

    except Exception as e:
        log.error(f"SSE stream for session {session_id} failed: {str(e)}")
        error_msg = JSONRPCMessage(
            root=JSONRPCNotification(
                jsonrpc="2.0",
                method="stream_error",
                params={"code": -32004, "message": f"Stream failed: {str(e)}"},
            )
        )
        yield f"data: {error_msg.model_dump_json(exclude_none=True)}\n\n"
        return

    log.info(f"SSE stream started for session {session_id}")
    try:
        while True:
            try:
                message: SessionMessage | Exception = await session_manager.receive_message(
                    session_id
                )
                if isinstance(message, Exception):
                    log.error(
                        f"SSE stream for session {session_id} received exception: {str(message)}"
                    )
                    # Convert exception to JSONRPCNotification
                    error_msg = JSONRPCMessage(
                        root=JSONRPCNotification(
                            jsonrpc="2.0",
                            method="stream_error",
                            params={"code": -32001, "message": str(message)},
                        )
                    )
                    yield f"data: {error_msg.model_dump_json(exclude_none=True)}\n\n"
                    break
                else:
                    message_json = message.message.model_dump_json(exclude_none=True)
                    yield f"data: {message_json}\n\n"
            # TODO: handle timeout?
            except ValueError as e:
                if "terminated" in str(e).lower() or "not found" in str(e).lower():
                    log.info(f"SSE stream for session {session_id} ended: {str(e)}")
                    # Send proper session end notification
                    end_msg = JSONRPCMessage(
                        root=JSONRPCNotification(
                            jsonrpc="2.0", method="session_ended", params={"reason": str(e)}
                        )
                    )
                    yield f"data: {end_msg.model_dump_json(exclude_none=True)}\n\n"
                    break
                else:
                    log.error(f"SSE stream for session {session_id} ValueError: {str(e)}")
                    error_msg = JSONRPCMessage(
                        root=JSONRPCNotification(
                            jsonrpc="2.0",
                            method="stream_error",
                            params={"code": -32002, "message": str(e)},
                        )
                    )
                    yield f"data: {error_msg.model_dump_json(exclude_none=True)}\n\n"
            except Exception as e:
                log.error(f"SSE stream for session {session_id} unexpected error: {str(e)}")
                error_msg = JSONRPCMessage(
                    root=JSONRPCNotification(
                        jsonrpc="2.0",
                        method="stream_error",
                        params={"code": -32003, "message": str(e)},
                    )
                )
                yield f"data: {error_msg.model_dump_json(exclude_none=True)}\n\n"
                break

    except Exception as e:
        log.error(f"SSE stream for session {session_id} failed: {str(e)}")
        error_msg = JSONRPCMessage(
            root=JSONRPCNotification(
                jsonrpc="2.0",
                method="stream_error",
                params={"code": -32004, "message": f"Stream failed: {str(e)}"},
            )
        )
        yield f"data: {error_msg.model_dump_json(exclude_none=True)}\n\n"


session_manager = MCPSessionManager()


async def _ensure_session_manager_started():
    """Ensure the session manager cleanup task is running."""
    log.debug("Ensuring MCP session manager is started")
    await session_manager.start()


async def handle_get_request(
    request: ASGIRequest, name: str, session_id: str | None = None
) -> JsonResponse | StreamingHttpResponse:
    """Return SSE stream for an existing session."""
    await _ensure_session_manager_started()

    if not session_id:
        log.warning(f"Session ID required for MCP server '{name}'")
        return JsonResponse({"error": "Session ID required"}, status=400)

    session = await session_manager.get_session(session_id)
    if not session:
        log.warning(f"Session {session_id} not found for MCP server '{name}'")
        return JsonResponse({"error": "Session not found"}, status=404)

    if session.terminated:
        log.warning(f"Session {session_id} is terminated for MCP server '{name}'")
        return JsonResponse({"error": "Session terminated"}, status=410)

    log.info(f"MCP GET {name} - SSE stream for existing session {session_id}")
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
    await _ensure_session_manager_started()

    if json_rpc_message is None:
        return JsonResponse({"error": "Missing JSON-RPC message"}, status=400)

    message_type = json_rpc_message.root.__class__.__name__
    log.info(f"MCP POST {name} - Session: {session_id}, Method: {message_type}")

    try:
        if is_initialize and not session_id:
            # TODO: before we initialize a session we should check if the server is available, otherwise we should return an error
            #  currently it runs quietly into a timeout
            log.info(f"Initializing session for MCP server '{name}'")
            mcp_config = get_mcp_config()
            server_config = mcp_config[name]
            url = server_config["url"]
            session_id = await session_manager.create_session(url)
            log.info(f"Session {session_id} initialized")

        if not session_id:
            log.warning("MCP request missing session ID")
            return JsonResponse({"error": "Mcp-Session-Id header required"}, status=400)

        session = await session_manager.get_session(session_id)
        if not session:
            log.warning(f"Session {session_id} not found in POST request")
            return JsonResponse({"error": "Session not found"}, status=404)

        log.debug(f"Sending message to session {session_id}")
        session_message = SessionMessage(json_rpc_message)
        await session_manager.send_message(session_id, session_message)

        # Only wait for response if this is a request (not a notification)
        # Per MCP spec: "The server MUST NOT send a response to notifications"
        if isinstance(json_rpc_message.root, JSONRPCRequest):
            log.debug(f"Waiting for response from session {session_id}")
            received_message = await session_manager.receive_message(session_id)
            log.debug(f"Received response from session {session_id}")

            if isinstance(received_message, Exception):
                log.error(f"Session {session_id} returned exception: {str(received_message)}")
                return JsonResponse({"error": str(received_message)}, status=500)

            response_data = received_message.message.model_dump(exclude_none=True)
            log.debug(f"MCP response session {session_id}: {len(str(response_data))} chars")
            response = JsonResponse(response_data)
            if session_id:
                response.headers["mcp-session-id"] = session_id
            return response
        else:
            log.debug(f"Notification processed for session {session_id}")
            response = JsonResponse({"status": "accepted"}, status=202)
            if session_id:
                response.headers["mcp-session-id"] = session_id
            return response
    except (KeyError, RuntimeError) as e:
        log.error(f"MCP server '{name}' not found: {str(e)}")
        return JsonResponse({"error": f"MCP server '{name}' not found: {str(e)}"}, status=404)
    except Exception as e:
        log.error(f"Error handling MCP request for server '{name}': {str(e)}")
        return JsonResponse({"error": str(e)}, status=400)


async def handle_delete_request(
    request: ASGIRequest, name: str, session_id: str | None = None
) -> JsonResponse:
    """Terminate MCP session."""
    await _ensure_session_manager_started()

    if not session_id:
        log.warning(f"Session {session_id} not found for MCP server '{name}'")
        return JsonResponse({"error": "Mcp-Session-Id header required"}, status=400)

    log.info(f"MCP DELETE {name} - Closing session {session_id}")
    try:
        await session_manager.terminate_session(session_id)
        return JsonResponse({"status": "session_terminated"})
    except Exception as e:
        log.error(f"Error terminating session {session_id}: {str(e)}")
        return JsonResponse({"error": str(e)}, status=400)


@csrf_exempt
@parse_jsonrpc_message
@require_http_methods(["GET", "POST", "DELETE"])
@token_authenticated(token_auth_only=True)
@log_request
async def mcp_server(
    request: ASGIRequest,
    token: Token,
    request_log: Request,
    name,
    json_rpc_message: JSONRPCMessage | None = None,
    session_id: str | None = None,
    is_initialize: bool = False,
):
    """
    Handles GET, POST and DELETE requests for /mcp-servers/{name}/mcp path.
    """
    log.debug(
        f"MCP server request - Method: {request.method}, Server: '{name}', Session: {session_id}"
    )

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
