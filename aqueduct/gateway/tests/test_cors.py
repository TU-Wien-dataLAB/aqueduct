"""
Tests for CORS configuration — middleware matching logic and header presence
on API vs non-API endpoints.
"""

from unittest.mock import Mock

from corsheaders.middleware import CorsMiddleware
from django.http import HttpRequest
from django.test import SimpleTestCase

from gateway.tests.utils import _build_chat_headers
from gateway.tests.utils.base import GatewayIntegrationTestCase


def _make_request(path: str) -> HttpRequest:
    """Build a minimal HttpRequest with just path_info set."""
    request = Mock(spec=HttpRequest)
    request.path_info = path
    return request


class CorsMiddlewareIsEnabledTest(SimpleTestCase):
    """
    Unit-test CorsMiddleware.is_enabled() directly by passing mocked HttpRequests.
    This tests the actual middleware matching logic, not just the raw regex.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Instantiate the middleware with a no-op get_response.
        cls.middleware = CorsMiddleware(lambda r: None)

    def _assert_enabled(self, path: str):
        request = _make_request(path)
        self.assertTrue(
            self.middleware.is_enabled(request), f"Expected CORS to be enabled for '{path}'"
        )

    def _assert_not_enabled(self, path: str):
        request = _make_request(path)
        self.assertFalse(
            self.middleware.is_enabled(request), f"Expected CORS to be disabled for '{path}'"
        )

    # ── API paths (CORS enabled) ─────────────────────────────────────────

    def test_api_paths_enabled(self):
        api_paths = [
            "/completions",
            "/v1/completions",
            "/chat/completions",
            "/v1/chat/completions",
            "/embeddings",
            "/v1/embeddings",
            "/models",
            "/v1/models",
            "/audio/speech",
            "/v1/audio/speech",
            "/audio/transcriptions",
            "/v1/audio/transcriptions",
            "/images/generations",
            "/v1/images/generations",
            "/files",
            "/v1/files",
            "/files/file-abc",
            "/v1/files/file-abc",
            "/files/file-abc/content",
            "/v1/files/file-abc/content",
            "/batches",
            "/v1/batches",
            "/batches/batch-abc",
            "/v1/batches/batch-abc",
            "/batches/batch-abc/cancel",
            "/v1/batches/batch-abc/cancel",
            "/responses",
            "/v1/responses",
            "/responses/resp-abc",
            "/v1/responses/resp-abc",
            "/responses/resp-abc/input_items",
            "/v1/responses/resp-abc/input_items",
            "/mcp-servers/my-server/mcp",
            "/vector_stores",
            "/v1/vector_stores",
            "/vector_stores/vs-abc",
            "/v1/vector_stores/vs-abc",
            "/vector_stores/vs-abc/files",
            "/v1/vector_stores/vs-abc/files",
            "/vector_stores/vs-abc/files/file-abc",
            "/v1/vector_stores/vs-abc/files/file-abc",
            "/vector_stores/vs-abc/file_batches",
            "/v1/vector_stores/vs-abc/file_batches",
            "/vector_stores/vs-abc/file_batches/batch-abc",
            "/v1/vector_stores/vs-abc/file_batches/batch-abc",
            "/vector_stores/vs-abc/file_batches/batch-abc/cancel",
            "/v1/vector_stores/vs-abc/file_batches/batch-abc/cancel",
            "/vector_stores/vs-abc/search",
            "/v1/vector_stores/vs-abc/search",
            "/vector_stores/vs-abc/files/file-abc/content",
            "/v1/vector_stores/vs-abc/files/file-abc/content",
            "/vector_stores/vs-abc/file_batches/batch-abc/files",
            "/v1/vector_stores/vs-abc/file_batches/batch-abc/files",
        ]
        for path in api_paths:
            with self.subTest(path=path):
                self._assert_enabled(path)

    # ── non-API paths (CORS disabled) ─────────────────────────────────────

    def test_non_api_paths_disabled(self):
        non_api_paths = [
            "/",
            "/login/",
            "/login",
            "/oidc/",
            "/oidc/callback/",
            "/admin/",
            "/aqueduct/admin/",
            "/aqueduct/admin/foo/",
            "/aqueduct/management/tokens/",
            "/static/",
            "/static/css/main.css",
            "/silk/",
            "/silk/requests/",
            "/health/",
            "/api/",
            "/anything-else",
        ]
        for path in non_api_paths:
            with self.subTest(path=path):
                self._assert_not_enabled(path)


class CorsHeadersIntegrationTest(GatewayIntegrationTestCase):
    """
    Verify that CORS response headers are actually present on API endpoints
    and absent on non-API endpoints.
    """

    def test_api_endpoint_returns_cors_header(self):
        """An API endpoint should include access-control-allow-origin when Origin is sent."""
        headers = _build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN)
        headers["Origin"] = "http://example.com"

        response = self.client.get("/models", headers=headers)
        self.assertEqual(response.status_code, 200)
        # CORS_ALLOW_ALL_ORIGINS=True + no credentials → "*"
        self.assertEqual(response["Access-Control-Allow-Origin"], "*")

    def test_api_endpoint_returns_cors_header_with_v1_prefix(self):
        """Same as above but with /v1/ prefix."""
        headers = _build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN)
        headers["Origin"] = "http://example.com"

        response = self.client.get("/v1/models", headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Access-Control-Allow-Origin"], "*")

    def test_non_api_endpoint_omits_cors_header(self):
        """A non-API endpoint must NOT include access-control-allow-origin."""
        headers = {"Origin": "http://example.com"}

        response = self.client.get("/oidc/authenticate/", headers=headers)
        self.assertNotIn("Access-Control-Allow-Origin", response)

    def test_admin_endpoint_omits_cors_header(self):
        """Admin paths must NOT include CORS headers."""
        headers = {"Origin": "http://example.com"}

        response = self.client.get("/admin/", headers=headers)
        self.assertNotIn("Access-Control-Allow-Origin", response)

    def test_preflight_options_on_api_endpoint(self):
        """OPTIONS preflight on an API URL should return CORS headers."""
        response = self.client.options(
            "/models",
            headers={"Origin": "http://example.com", "Access-Control-Request-Method": "GET"},
        )
        self.assertEqual(response["Access-Control-Allow-Origin"], "*")
        self.assertIn("Access-Control-Allow-Methods", response)

    def test_preflight_options_on_non_api_endpoint(self):
        """OPTIONS preflight on a non-API URL should NOT return CORS headers."""
        response = self.client.options(
            "/oidc/authenticate/",
            headers={"Origin": "http://example.com", "Access-Control-Request-Method": "GET"},
        )
        self.assertNotIn("Access-Control-Allow-Origin", response)
