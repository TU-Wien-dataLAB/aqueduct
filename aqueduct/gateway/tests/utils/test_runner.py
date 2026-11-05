import atexit
import logging
from typing import TYPE_CHECKING

from django.conf import settings
from django.test.runner import DiscoverRunner

from mock_api.mock_server import MockAPIServer

if TYPE_CHECKING:
    from gateway.tests.utils.mcp import MCPEverythingServer

logger = logging.getLogger("aqueduct")


class MockServerState:
    _mock_server: MockAPIServer | None = None
    _mcp_server: "MCPEverythingServer | None" = None

    @classmethod
    def cleanup_mcp_server(cls):
        server, cls._mcp_server = cls._mcp_server, None
        if server is not None:
            server.stop()

    @classmethod
    def get_mock_server(cls) -> MockAPIServer | None:
        if cls._mock_server is None:
            raise RuntimeError("Mock server not initialized. Use MockServerTestRunner.")
        return cls._mock_server

    @classmethod
    def cleanup_mock_server(cls):
        if cls._mock_server is not None and cls._mock_server.process:
            cls._mock_server.stop()
            cls._mock_server = None


def get_shared_mock_server() -> MockAPIServer | None:
    return MockServerState.get_mock_server()


def get_shared_mcp_server() -> "MCPEverythingServer":
    """Start the real npm reference server only when a live MCP test needs it."""
    if MockServerState._mcp_server is None:
        from gateway.tests.utils.mcp import MCPEverythingServer

        server = MCPEverythingServer()
        MockServerState._mcp_server = server
        try:
            server.start()
        except BaseException:
            # Include KeyboardInterrupt: npm may already have spawned node.
            MockServerState.cleanup_mcp_server()
            raise
    return MockServerState._mcp_server


class MockServerTestRunner(DiscoverRunner):
    """
    Start the shared mock API and clean up it and the lazy live MCP server.
    """

    def setup_test_environment(self, **kwargs):
        """Set up the test environment, starting the mock API server."""
        super().setup_test_environment(**kwargs)
        atexit.register(MockServerState.cleanup_mcp_server)

        if settings.TESTS_USE_MOCK_API:
            MockServerState._mock_server = MockAPIServer(host="localhost", delays=False)
            try:
                MockServerState._mock_server.start()
                logger.info("✓ Mock server started on %s.", MockServerState._mock_server.base_url)
            except RuntimeError as err:
                logger.exception("✗ Failed to start mock server: %s", err)
                raise

            atexit.register(MockServerState.cleanup_mock_server)
        else:
            logger.warning("Skipping the initialisation of the mock server.")

    def teardown_test_environment(self, **kwargs):
        """Stop both subprocesses, including after a failed test class setup."""
        try:
            MockServerState.cleanup_mcp_server()
        finally:
            MockServerState.cleanup_mock_server()
            super().teardown_test_environment(**kwargs)
