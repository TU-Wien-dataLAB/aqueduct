import asyncio
import logging
import uuid
from enum import Enum
from typing import AsyncGenerator, Dict

import anyio
import httpx
from anyio import ClosedResourceError
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from django.core.handlers.asgi import ASGIRequest
from django.http import JsonResponse, StreamingHttpResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from mcp.client.streamable_http import StreamableHTTPTransport
from mcp.shared.message import SessionMessage
from mcp.types import CONNECTION_CLOSED, ErrorData, JSONRPCError, JSONRPCMessage, JSONRPCRequest
from pydantic import TypeAdapter

from gateway.config import get_mcp_config
from gateway.views.decorators import (
    check_mcp_server_availability,
    log_request,
    mcp_transport_security,
    parse_body,
    parse_jsonrpc_message,
    token_authenticated,
)

log = logging.getLogger("aqueduct")


class MCPSessionError(Exception):
    """Base exception for MCP session errors."""

    pass


class SessionNotFoundError(MCPSessionError):
    """Raised when a session is not found."""

    pass


class SessionNotReadyError(MCPSessionError):
    """Raised when a session is not ready for operations."""

    pass


class SessionNotStartedError(MCPSessionError):
    """Raised when a session has not been started."""

    pass


class SessionState(Enum):
    """Session lifecycle states for coordinating concurrent operations."""

    INITIALIZING = "initializing"
    READY = "ready"
    TERMINATING = "terminating"
    TERMINATED = "terminated"


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
        self.state = SessionState.INITIALIZING

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

        self._active_operations = 0
        self._operation_done_event = asyncio.Event()
        self._operation_done_event.set()

    def is_ready(self) -> bool:
        """Check if session is ready for operations."""
        return self.state == SessionState.READY

    def register_operation_start(self):
        """Register an operation has started. Used for coordinating cleanup."""
        self._active_operations += 1
        self._operation_done_event.clear()
        log.debug(f"Session {self.session_id}: active operations = {self._active_operations}")

    def register_operation_done(self):
        """Register an operation has completed. Must be paired with start()."""
        self._active_operations -= 1
        log.debug(f"Session {self.session_id}: active operations = {self._active_operations}")
        if self._active_operations == 0:
            self._operation_done_event.set()

    def has_active_operations(self) -> bool:
        """Check if there are active operations on this session."""
        return self._active_operations > 0

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

        # Mark session as ready
        self.state = SessionState.READY
        log.info(f"MCP session {self.session_id} is now ready")

    async def stop(self):
        """Stop the session cleanly, waiting for active operations to complete."""
        if self.terminated or self.state == SessionState.TERMINATED:
            return

        log.info(f"Stopping MCP session {self.session_id}")
        self.state = SessionState.TERMINATING
        self.terminated = True

        # Wait for active operations to complete
        if self.has_active_operations():
            log.info(
                f"Session {self.session_id}: waiting for {self._active_operations} active operations to complete"
            )
            try:
                await asyncio.wait_for(self._operation_done_event.wait(), timeout=5.0)
                log.info(f"Session {self.session_id}: all operations completed")
            except asyncio.TimeoutError:
                log.warning(
                    f"Session {self.session_id}: timeout waiting for operations to complete, forcing shutdown"
                )

        # Terminate session via DELETE request
        if self.transport.session_id and self._httpx_client:
            log.debug(f"Terminating MCP protocol session {self.transport.session_id}")
            await self.transport.terminate_session(self._httpx_client)

        # Close streams (this will cause post_writer to exit)
        # Add exception handling for already-closed streams
        if self._read_stream_writer:
            log.debug(f"Closing read stream for session {self.session_id}")
            try:
                await self._read_stream_writer.aclose()
            except Exception as e:
                log.debug(f"Error closing read stream: {e}")
        if self.write_stream:
            log.debug(f"Closing write stream for session {self.session_id}")
            try:
                await self.write_stream.aclose()
            except Exception as e:
                log.debug(f"Error closing write stream: {e}")

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

        self.state = SessionState.TERMINATED
        log.info(f"MCP session {self.session_id} stopped successfully")

    async def send_message(self, message: SessionMessage):
        """Send a message to the MCP server."""
        if not self.is_ready():
            raise SessionNotReadyError(f"Session not ready (current state: {self.state.value})")
        if not self.write_stream:
            raise SessionNotStartedError("Session not started")

        message_type = message.message.root.__class__.__name__
        log.info(f"Sending: {message_type} to session {self.session_id}")
        await self.write_stream.send(message)
        log.debug(f"Message sent to stream in session {self.session_id}")
        self.last_accessed_at = timezone.now()

    async def receive_message(self) -> SessionMessage | Exception:
        """Receive a message from the MCP server."""
        if not self.is_ready():
            raise SessionNotReadyError(f"Session not ready (current state: {self.state.value})")
        if not self.read_stream:
            raise SessionNotStartedError("Session not started")

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
                    # Get session first, mark for termination
                    async with self._lock:
                        session = self._sessions.get(session_id)
                        if session:
                            # Mark as terminating to prevent new operations
                            session.state = SessionState.TERMINATING
                            session.terminated = True

                    if session:
                        # Stop will wait for active operations to complete
                        await session.stop()
                        # Remove from dict after stopping
                        async with self._lock:
                            self._sessions.pop(session_id, None)
                        log.debug(f"Cleaned up idle session {session_id}")

    async def create_session(self, url: str, headers: dict | None = None) -> str:
        """Create a new MCP session."""
        session_id = str(uuid.uuid4())
        log.info(f"Creating MCP session {session_id} for {url}")

        session = ManagedMCPSession(session_id, url, headers)

        # Add session to manager first, then start it to avoid race conditions
        async with self._lock:
            self._sessions[session_id] = session

        # Start the session
        await session.start()

        # Verify session is ready before returning
        if not session.is_ready():
            log.error(f"Session {session_id} failed to become ready after start")
            # Clean up failed session
            async with self._lock:
                self._sessions.pop(session_id, None)
            await session.stop()
            raise ValueError(f"Session {session_id} failed to initialize")

        log.info(f"MCP session {session_id} created and ready")
        return session_id

    async def get_session(self, session_id: str | None) -> ManagedMCPSession | None:
        """Get a session by ID."""
        if not session_id:
            return None
        async with self._lock:
            return self._sessions.get(session_id)

    async def get_session_with_retry(
        self, session_id: str | None, max_retries: int = 3, retry_delay: float = 0.1
    ) -> ManagedMCPSession | None:
        """Get a session with brief retry logic for initialization race conditions."""
        if not session_id:
            return None
        for attempt in range(max_retries):
            session = await self.get_session(session_id)
            if session and session.is_ready():
                return session
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                log.debug(f"Retrying session retrieval for {session_id}, attempt {attempt + 2}")
        return await self.get_session(session_id)

    async def terminate_session(self, session_id: str | None):
        """Terminate a session gracefully, coordinating with active operations."""
        if not session_id:
            return
        log.info(f"Terminating session {session_id}")
        async with self._lock:
            session = self._sessions.get(session_id)

        if session:
            # Mark session as terminating but keep in manager for ongoing operations
            session.state = SessionState.TERMINATING
            session.terminated = True
            log.debug(f"Session {session_id} marked for termination")

            # Stop the session (this will wait for operations to complete)
            await session.stop()

            # Now remove from manager after all operations are done
            async with self._lock:
                self._sessions.pop(session_id, None)
            log.info(f"Session {session_id} terminated and removed")
        else:
            log.warning(f"Attempt to terminate non-existent session {session_id}")

    async def send_message(self, session_id: str | None, message: SessionMessage):
        """Send message to session."""
        if not session_id:
            raise ValueError("Session ID required")
        session = await self.get_session(session_id)
        if not session:
            log.warning(f"Session {session_id} not found for send_message")
            raise SessionNotFoundError(f"Session {session_id} not found")
        await session.send_message(message)

    async def receive_message(self, session_id: str | None):
        """Receive message from session."""
        if not session_id:
            raise ValueError("Session ID required")
        session = await self.get_session(session_id)
        if not session:
            log.warning(f"Session {session_id} not found for receive_message")
            raise SessionNotFoundError(f"Session {session_id} not found")
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


def parse_session_message(
    received_message: SessionMessage | Exception,
    reqeust_id: str | int,
    session_id: str,
    json: bool = False,
) -> dict | str:
    if isinstance(received_message, Exception):
        log.error(f"Session {session_id} returned exception: {str(received_message)}")
        jsonrpc_message = JSONRPCError(
            jsonrpc="2.0",
            id=reqeust_id,
            error=ErrorData(code=CONNECTION_CLOSED, message=str(received_message)),
        )
    else:
        jsonrpc_message = received_message.message

    if json:
        return jsonrpc_message.model_dump_json(exclude_none=True)
    else:
        return jsonrpc_message.model_dump(exclude_none=True)


def _validate_session(
    session: ManagedMCPSession | None, session_id: str | None, name: str
) -> JsonResponse | None:
    if not session:
        log.warning(f"Session {session_id} not found for MCP server '{name}'")
        return JsonResponse({"error": "Session not found"}, status=404)
    if session.terminated:
        log.warning(f"Session {session_id} is terminated for MCP server '{name}'")
        return JsonResponse({"error": "Session terminated"}, status=410)
    if not session.is_ready():
        log.warning(
            f"Session {session_id} not ready for MCP server '{name}' (state: {session.state.value})"
        )
        return JsonResponse(
            {"error": f"Session not ready (state: {session.state.value})"}, status=503
        )

    return None


async def _mcp_sse_stream(request_id: str | int, session_id: str) -> AsyncGenerator[str, None]:
    """Stream MCP messages via Server-Sent Events with lifecycle coordination."""
    log.info(f"SSE stream started for session {session_id}")
    session = await session_manager.get_session_with_retry(session_id)
    if not session or not session.is_ready():
        log.error(
            f"SSE stream cannot start - session {session_id} not ready (state: {session.state.value if session else 'None'})"
        )
        error_msg = JSONRPCError(
            jsonrpc="2.0",
            id=request_id,
            error=ErrorData(
                code=CONNECTION_CLOSED,
                message=f"Session not ready (state: {session.state.value if session else 'None'})",
            ),
        )

        yield f"data: {error_msg.model_dump_json(exclude_none=True)}\n\n"
        return

    # Register stream lifetime to coordinate with session termination
    session.register_operation_start()
    log.debug(f"SSE stream registered as active operation for session {session_id}")

    while True:
        try:
            message: SessionMessage | Exception = await session_manager.receive_message(session_id)
            message_json = parse_session_message(message, request_id, session_id, json=True)
            yield f"data: {message_json}\n\n"
        except (ClosedResourceError, httpx.HTTPStatusError, MCPSessionError) as e:
            log.info(f"SSE stream for session {session_id} ended: {str(e)}")
            # Send proper session end notification
            end_msg = parse_session_message(e, request_id, session_id, json=True)
            yield f"data: {end_msg}\n\n"
            break  # Session is gone, stop the stream
        except Exception as e:
            log.error(f"SSE stream for session {session_id} unexpected error: {str(e)}")
            error_msg = parse_session_message(e, request_id, session_id, json=True)
            yield f"data: {error_msg}\n\n"
            continue  # Keep streaming
        finally:
            # Signal that this operation is done
            session.register_operation_done()
            log.debug(f"SSE stream completed for session {session_id}, operation deregistered")


session_manager = MCPSessionManager()


async def _ensure_session_manager_started():
    """Ensure the session manager cleanup task is running."""
    log.debug("Ensuring MCP session manager is started")
    await session_manager.start()


async def handle_get_request(
    name: str, request_id: str | int, session_id: str
) -> JsonResponse | StreamingHttpResponse:
    """Return SSE stream for an existing session.

    See: https://modelcontextprotocol.io/specification/2025-06-18/basic/transports#listening-for-messages-from-the-server
    """
    await _ensure_session_manager_started()

    session = await session_manager.get_session(session_id)

    err_response = _validate_session(session, request_id, name)
    if err_response:
        return err_response

    log.info(f"MCP GET {name} - SSE stream for existing session {session_id}")
    headers = {"Cache-Control": "no-cache", "Connection": "keep-alive"}
    return StreamingHttpResponse(
        streaming_content=_mcp_sse_stream(request_id, session_id),
        content_type="text/event-stream",
        headers=headers,
    )


async def handle_post_request(
    name: str,
    json_rpc_message: JSONRPCMessage | None,
    request_id: str | int,
    session_id: str | None,
    is_initialize: bool,
) -> JsonResponse:
    """Send a message to MCP session or initialize new session when it is an initialization request.

    See: https://modelcontextprotocol.io/specification/2025-06-18/basic/transports#sending-messages-to-the-server
    """

    await _ensure_session_manager_started()

    if json_rpc_message is None:
        return JsonResponse({"error": "Missing JSON-RPC message"}, status=400)

    message_type = json_rpc_message.root.__class__.__name__
    log.info(f"MCP POST {name} - Session: {session_id}, Method: {message_type}")

    if is_initialize:
        try:
            log.info(f"Initializing session for MCP server '{name}'")
            mcp_config = get_mcp_config()
            server_config = mcp_config[name]
            url = server_config["url"]
            session_id = await session_manager.create_session(url)
            log.info(f"Session {session_id} initialized")
        except (KeyError, RuntimeError) as e:
            log.error(f"MCP server '{name}' not found: {str(e)}")
            return JsonResponse({"error": f"MCP server '{name}' not found: {str(e)}"}, status=404)

    session = await session_manager.get_session_with_retry(session_id)

    err_response = _validate_session(session, request_id, name)
    if err_response:
        return err_response

    # Register operation for requests that expect responses
    operation_registered = False
    if isinstance(json_rpc_message.root, JSONRPCRequest):
        session.register_operation_start()
        operation_registered = True

    try:
        log.debug(f"Sending message to session {session_id}")
        session_message = SessionMessage(json_rpc_message)
        await session_manager.send_message(session_id, session_message)

        # Only wait for response if this is a request (not a notification)
        # Per MCP spec: "The server MUST NOT send a response to notifications"
        if isinstance(json_rpc_message.root, JSONRPCRequest):
            received_message = await session_manager.receive_message(session_id)
            response_data = parse_session_message(
                received_message, request_id, session_id, json=False
            )
            response = JsonResponse(response_data)
        else:
            # If the request was a notification, return 202
            response = JsonResponse({"status": "accepted"}, status=202)

    except (ClosedResourceError, httpx.HTTPStatusError, MCPSessionError) as e:
        # Transport errors should be converted to JSON-RPC errors
        log.error(f"Transport error for MCP server '{name}': {str(e)}")
        error_response = parse_session_message(e, request_id, session_id, json=False)
        response = JsonResponse(error_response, status=200)  # 200 for JSON-RPC errors
    finally:
        if operation_registered:
            session.register_operation_done()

    response.headers["mcp-session-id"] = session_id
    return response


async def handle_delete_request(name: str, session_id: str | None = None) -> JsonResponse:
    """Terminate MCP session.

    See: https://modelcontextprotocol.io/specification/2025-06-18/basic/transports#session-management
    """
    await _ensure_session_manager_started()

    log.info(f"MCP DELETE {name} - Closing session {session_id}")
    await session_manager.terminate_session(session_id)
    return JsonResponse({"status": "session_terminated"})


@csrf_exempt
@parse_body(model=TypeAdapter(JSONRPCMessage))
@parse_jsonrpc_message
@require_http_methods(["GET", "POST", "DELETE"])
@token_authenticated(token_auth_only=True)
# @tos_accepted
@check_mcp_server_availability
@mcp_transport_security
@log_request
async def mcp_server(
    request: ASGIRequest,
    name: str | None,
    json_rpc_message: JSONRPCMessage | None = None,
    session_id: str | None = None,
    is_initialize: bool = False,
    *args,
    **kwargs,
):
    """
    Handles GET, POST and DELETE requests for /mcp-servers/{name}/mcp path.
    """
    log.info(
        f"MCP server request - Method: {request.method}, Server: '{name}', Session: {session_id}, "
        f"Headers: {dict(request.headers)}"
    )

    if json_rpc_message and isinstance(json_rpc_message.root, JSONRPCRequest):
        request_id = json_rpc_message.root.id
    else:
        request_id = 0

    if request.method == "GET":
        return await handle_get_request(name, request_id=request_id, session_id=session_id)
    elif request.method == "POST":
        return await handle_post_request(
            name,
            json_rpc_message=json_rpc_message,
            request_id=request_id,
            session_id=session_id,
            is_initialize=is_initialize,
        )
    elif request.method == "DELETE":
        return await handle_delete_request(name, session_id=session_id)
