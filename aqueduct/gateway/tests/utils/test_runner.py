import atexit
import logging
from typing import Optional

from django.test.runner import DiscoverRunner

from mock_api.mock_server import MockAPIServer

logger = logging.getLogger("aqueduct")


# Tests need access to the global mock server instance
_shared_mock_server: Optional[MockAPIServer] = None


def get_shared_mock_server():
    """Get the mock server instance shared by all tests in the suite."""
    global _shared_mock_server
    if _shared_mock_server is None:
        raise RuntimeError("Mock server not initialized. Use MockServerTestRunner.")
    return _shared_mock_server


class MockServerTestRunner(DiscoverRunner):
    """
    Custom Django test runner that starts a single mock server instance
    for the entire test suite.
    """

    def setup_test_environment(self, **kwargs):
        """Set up the test environment, starting the mock server."""
        super().setup_test_environment(**kwargs)

        # Start the mock server once for the entire test suite
        global _shared_mock_server
        _shared_mock_server = MockAPIServer(host="localhost")
        try:
            _shared_mock_server.start()
            logger.info(f"✓ Mock server started on {_shared_mock_server.base_url}.")
        except RuntimeError as err:
            logger.error(f"✗ Failed to start mock server: {err}")
            raise

        # Ensure server is stopped even if tests crash
        atexit.register(self._cleanup_server)

    def teardown_test_environment(self, **kwargs):
        """Tear down the test environment, stopping the mock server."""
        super().teardown_test_environment(**kwargs)
        self._cleanup_server()

    def _cleanup_server(self):
        """Stop the mock server if it's running."""
        global _shared_mock_server
        if _shared_mock_server is not None and _shared_mock_server.process:
            logger.info("\nStopping mock server...")
            _shared_mock_server.stop()
            _shared_mock_server = None
