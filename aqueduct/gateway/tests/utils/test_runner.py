import atexit
import logging
import os
import subprocess
from typing import Optional

from django.conf import settings
from django.test.runner import DiscoverRunner

from mock_api.mock_server import MockAPIServer

logger = logging.getLogger("aqueduct")


# Tests need access to the global mock server instance
_shared_mock_server: Optional[MockAPIServer] = None
# TODO: this is a process; do we want a separate server class?
shared_mcp_server_process: Optional[subprocess.Popen] = None
mcp_server_port: Optional[int] = None


def get_shared_mock_server():
    """Get the mock server instance shared by all tests in the suite."""
    global _shared_mock_server
    if _shared_mock_server is None:
        raise RuntimeError("Mock server not initialized. Use MockServerTestRunner.")
    return _shared_mock_server


# def get_shared_mcp_server_process():
#     """Get the mock server process shared by all tests in the suite."""
#     global shared_mcp_server_process
#     if shared_mcp_server_process is None:
#         raise RuntimeError(
#             "MCP server not initialized. Use MockServerTestRunner. "
#             # "Did you set `--no-mcp` flag in the test command?"
#         )
#     return shared_mcp_server_process


class MockServerTestRunner(DiscoverRunner):
    """
    Custom Django test runner that starts a single mock server instance
    for the entire test suite.
    """

    # def __init__(self, **kwargs):
    #     self.no_mcp = kwargs.pop("no_mcp", False)
    #     super().__init__(**kwargs)
    #
    # @classmethod
    # def add_arguments(cls, parser: ArgumentParser):
    #     super().add_arguments(parser)
    #     parser.add_argument(
    #         "--no-mcp",
    #         action="store_true",
    #         dest="no_mcp",
    #         help=(
    #             "Disables starting of the mock MCP server. This can speed up the test suite "
    #             "when only a subset of tests is specified and they do not need the MCP server."
    #         ),
    #     )

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
                logger.error(f"✗ Failed to start mock server: {err}")
                raise

            # Ensure the mock server is stopped even if tests crash
            atexit.register(self._cleanup_mock_server)
        else:
            logger.warning("Skipping the initialisation of the mock server.")

        # if self.no_mcp:
        #     # Do not start the MCP server
        #     logger.warning("Skipping the initialisation of the MCP server.")
        #     return

        # Ensure the MCP server is stopped even if tests crash
        atexit.register(self._cleanup_mcp_server)

    def teardown_test_environment(self, **kwargs):
        """Tear down the test environment, stopping the mock server."""
        super().teardown_test_environment(**kwargs)
        self._cleanup_mock_server()
        self._cleanup_mcp_server()

    def _cleanup_mock_server(self):
        """Stop the mock server if it's running."""
        global _shared_mock_server
        if _shared_mock_server is not None and _shared_mock_server.process:
            logger.info("\nStopping mock server...")
            _shared_mock_server.stop()
            _shared_mock_server = None

    def _cleanup_mcp_server(self):
        """Stop the MCP server if it's running."""
        global shared_mcp_server_process
        if shared_mcp_server_process is not None and shared_mcp_server_process.poll() is None:
            logger.info("\nStopping MCP everything server...")
            shared_mcp_server_process.terminate()
            try:
                shared_mcp_server_process.wait(timeout=10)
                logger.info("MCP server stopped successfully")
            except subprocess.TimeoutExpired:
                logger.warning("MCP server did not stop gracefully, forcing kill")
                shared_mcp_server_process.kill()
                shared_mcp_server_process.wait()
            shared_mcp_server_process = None

        from gateway.tests.utils.mcp import MCP_CONFIG_PATH

        if os.path.exists(MCP_CONFIG_PATH):
            os.remove(MCP_CONFIG_PATH)
