import atexit
import logging

from django.conf import settings
from django.test.runner import DiscoverRunner

from mock_api.mock_server import MockAPIServer

logger = logging.getLogger("aqueduct")


class MockServerState:
    _mock_server: MockAPIServer | None = None

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


class MockServerTestRunner(DiscoverRunner):
    """
    Custom Django test runner that starts a single mock server instance
    for the entire test suite.
    """

    def setup_test_environment(self, **kwargs):
        """Set up the test environment, starting the mock API server."""
        super().setup_test_environment(**kwargs)

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
        """Tear down the test environment, stopping the mock server."""
        MockServerState.cleanup_mock_server()
        super().teardown_test_environment(**kwargs)
