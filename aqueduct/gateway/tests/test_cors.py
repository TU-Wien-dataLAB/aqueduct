"""
Tests for CORS configuration — regex matching, middleware logic, and header presence
on API vs non-API endpoints.
"""

import re
from unittest.mock import Mock

from corsheaders.middleware import CorsMiddleware
from django.conf import settings
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
            "/images/generations",
            "/files",
            "/files/file-abc/content",
            "/batches/batch-abc/cancel",
            "/responses/resp-abc/input_items",
            "/mcp-servers/my-server/mcp",
            "/vector_stores/vs-abc/file_batches/batch-abc/files",
        ]
        for path in api_paths:
            with self.subTest(path=path):
                self._assert_enabled(path)

    # ── non-API paths (CORS disabled) ─────────────────────────────────────

    def test_non_api_paths_disabled(self):
        non_api_paths = [
            "/",
            "/login/",
            "/oidc/callback/",
            "/admin/",
            "/aqueduct/admin/",
            "/aqueduct/management/tokens/",
            "/static/css/main.css",
            "/silk/requests/",
            "/health/",
            "/anything-else",
        ]
        for path in non_api_paths:
            with self.subTest(path=path):
                self._assert_not_enabled(path)


class CorsRegexTest(SimpleTestCase):
    """
    Assert that the default CORS_URLS_REGEX matches all API paths in gateway/urls.py
    and excludes internal/admin/static paths.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.regex = settings.CORS_URLS_REGEX

    def _assert_matches(self, path: str):
        self.assertIsNotNone(
            re.match(self.regex, path), f"Expected CORS_URLS_REGEX to match '{path}'"
        )

    def _assert_no_match(self, path: str):
        self.assertIsNone(
            re.match(self.regex, path), f"Expected CORS_URLS_REGEX to NOT match '{path}'"
        )

    # ── positive cases (API paths that must be matched) ──────────────────────

    def test_completions(self):
        self._assert_matches("/completions")
        self._assert_matches("/v1/completions")

    def test_chat_completions(self):
        self._assert_matches("/chat/completions")
        self._assert_matches("/v1/chat/completions")

    def test_embeddings(self):
        self._assert_matches("/embeddings")
        self._assert_matches("/v1/embeddings")

    def test_models(self):
        self._assert_matches("/models")
        self._assert_matches("/v1/models")

    def test_audio(self):
        self._assert_matches("/audio/speech")
        self._assert_matches("/v1/audio/speech")
        self._assert_matches("/audio/transcriptions")
        self._assert_matches("/v1/audio/transcriptions")

    def test_images(self):
        self._assert_matches("/images/generations")
        self._assert_matches("/v1/images/generations")

    def test_files(self):
        self._assert_matches("/files")
        self._assert_matches("/v1/files")
        self._assert_matches("/files/file-abc")
        self._assert_matches("/v1/files/file-abc")
        self._assert_matches("/files/file-abc/content")
        self._assert_matches("/v1/files/file-abc/content")

    def test_batches(self):
        self._assert_matches("/batches")
        self._assert_matches("/v1/batches")
        self._assert_matches("/batches/batch-abc")
        self._assert_matches("/v1/batches/batch-abc")
        self._assert_matches("/batches/batch-abc/cancel")
        self._assert_matches("/v1/batches/batch-abc/cancel")

    def test_responses(self):
        self._assert_matches("/responses")
        self._assert_matches("/v1/responses")
        self._assert_matches("/responses/resp-abc")
        self._assert_matches("/v1/responses/resp-abc")
        self._assert_matches("/responses/resp-abc/input_items")
        self._assert_matches("/v1/responses/resp-abc/input_items")

    def test_mcp_servers(self):
        self._assert_matches("/mcp-servers/my-server/mcp")

    def test_vector_stores(self):
        self._assert_matches("/vector_stores")
        self._assert_matches("/v1/vector_stores")
        self._assert_matches("/vector_stores/vs-abc")
        self._assert_matches("/v1/vector_stores/vs-abc")
        self._assert_matches("/vector_stores/vs-abc/files")
        self._assert_matches("/v1/vector_stores/vs-abc/files")
        self._assert_matches("/vector_stores/vs-abc/files/file-abc")
        self._assert_matches("/v1/vector_stores/vs-abc/files/file-abc")
        self._assert_matches("/vector_stores/vs-abc/file_batches")
        self._assert_matches("/v1/vector_stores/vs-abc/file_batches")
        self._assert_matches("/vector_stores/vs-abc/file_batches/batch-abc")
        self._assert_matches("/v1/vector_stores/vs-abc/file_batches/batch-abc")
        self._assert_matches("/vector_stores/vs-abc/file_batches/batch-abc/cancel")
        self._assert_matches("/v1/vector_stores/vs-abc/file_batches/batch-abc/cancel")
        self._assert_matches("/vector_stores/vs-abc/search")
        self._assert_matches("/v1/vector_stores/vs-abc/search")
        self._assert_matches("/vector_stores/vs-abc/files/file-abc/content")
        self._assert_matches("/v1/vector_stores/vs-abc/files/file-abc/content")
        self._assert_matches("/vector_stores/vs-abc/file_batches/batch-abc/files")
        self._assert_matches("/v1/vector_stores/vs-abc/file_batches/batch-abc/files")

    # ── negative cases (non-API paths that must be excluded) ──────────────────

    def test_root(self):
        self._assert_no_match("/")

    def test_login(self):
        self._assert_no_match("/login/")
        self._assert_no_match("/login")

    def test_oidc(self):
        self._assert_no_match("/oidc/")
        self._assert_no_match("/oidc/callback/")

    def test_admin(self):
        self._assert_no_match("/admin/")
        self._assert_no_match("/aqueduct/admin/")
        self._assert_no_match("/aqueduct/admin/foo/")

    def test_management(self):
        self._assert_no_match("/aqueduct/management/tokens/")

    def test_static(self):
        self._assert_no_match("/static/")
        self._assert_no_match("/static/css/main.css")

    def test_silk(self):
        self._assert_no_match("/silk/")
        self._assert_no_match("/silk/requests/")

    def test_health(self):
        self._assert_no_match("/health/")

    def test_unrelated_paths(self):
        self._assert_no_match("/api/")
        self._assert_no_match("/anything-else")


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

        # Hit a non-API URL that doesn't need staticfiles (e.g. /oidc/ returns 302)
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
