import atexit
import logging
import os
import signal
import subprocess

from django.conf import settings
from django.test.runner import DiscoverRunner

from mock_api.mock_server import MockAPIServer

logger = logging.getLogger("aqueduct")


# Tests need access to the global mock server instance
_shared_mock_server: MockAPIServer | None = None
# ...and some tests also need a running MCP server
_shared_mcp_server_process: subprocess.Popen | None = None
_mcp_server_port: int | None = None


def get_shared_mock_server() -> MockAPIServer | None:
    """Get the mock server instance shared by all tests in the suite."""
    if _shared_mock_server is None:
        raise RuntimeError("Mock server not initialized. Use MockServerTestRunner.")
    return _shared_mock_server


def get_shared_mcp_server_process() -> subprocess.Popen | None:
    """Get the global MCP server process shared by all MCP-related tests."""
    return _shared_mcp_server_process


def get_mcp_server_port() -> int | None:
    """Get the port used by the global MCP server."""
    return _mcp_server_port


def set_shared_mcp_server(process: subprocess.Popen, port: int) -> None:
    """Set the global MCP server process and port."""
    global _shared_mcp_server_process, _mcp_server_port
    _shared_mcp_server_process = process
    _mcp_server_port = port


class MockServerTestRunner(DiscoverRunner):
    """
    Custom Django test runner that starts a single mock server instance
    for the entire test suite.

    It also stops the test MCP server if it has been started by any tests that needed it.
    """

    def setup_test_environment(self, **kwargs):
        """Set up the test environment, starting the mock API server."""
        super().setup_test_environment(**kwargs)

        if settings.TESTS_USE_MOCK_API:
            # Start the mock server once for the entire test suite
            global _shared_mock_server
            _shared_mock_server = MockAPIServer(host="localhost", delays=False)
            try:
                _shared_mock_server.start()
                logger.info(f"✓ Mock server started on {_shared_mock_server.base_url}.")
            except RuntimeError as err:
                logger.exception(f"✗ Failed to start mock server: {err}")
                raise

            # Ensure the mock server is stopped even if tests crash
            atexit.register(self._cleanup_mock_server)
        else:
            logger.warning("Skipping the initialisation of the mock server.")

        # If the MCP server has been started, ensure it is stopped even if tests crash
        atexit.register(self._cleanup_mcp_server)

    def teardown_test_environment(self, **kwargs):
        """Tear down the test environment, stopping the mock and MCP server."""
        self._cleanup_mock_server()
        self._cleanup_mcp_server()
        super().teardown_test_environment(**kwargs)

    def _cleanup_mock_server(self):
        """Stop the mock server if it's running."""
        global _shared_mock_server
        if _shared_mock_server is not None and _shared_mock_server.process:
            logger.info("\nStopping mock server...")
            _shared_mock_server.stop()
            _shared_mock_server = None

    def _cleanup_mcp_server(self):
        """Stop the MCP server if it's running."""
        global _shared_mcp_server_process, _mcp_server_port

        if _shared_mcp_server_process is not None and _shared_mcp_server_process.poll() is None:
            logger.info("\nStopping MCP everything server...")

            # Kill the entire process group (npx spawns child processes)
            try:
                os.killpg(os.getpgid(_shared_mcp_server_process.pid), signal.SIGTERM)
                _shared_mcp_server_process.wait(timeout=5)
                logger.info("MCP server stopped successfully")
            except subprocess.TimeoutExpired:
                logger.warning("MCP server did not stop gracefully, forcing kill")
                try:
                    os.killpg(os.getpgid(_shared_mcp_server_process.pid), signal.SIGKILL)
                    _shared_mcp_server_process.wait()
                    logger.info("MCP server stopped successfully (forced)")
                except ProcessLookupError:
                    logger.info("Process group already terminated")
            except ProcessLookupError:
                logger.info("Process already terminated")

            _shared_mcp_server_process = None
            _mcp_server_port = None

        from gateway.tests.utils.mcp import MCP_CONFIG_PATH

        if MCP_CONFIG_PATH.exists():
            MCP_CONFIG_PATH.unlink()
