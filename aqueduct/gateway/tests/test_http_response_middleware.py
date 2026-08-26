import json
from http import HTTPStatus
from pathlib import Path
from unittest.mock import Mock, patch

from asgiref.sync import sync_to_async
from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.auth.middleware import AuthenticationMiddleware
from django.contrib.sessions.middleware import SessionMiddleware
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import AsyncRequestFactory, override_settings
from django.urls import resolve, reverse

from gateway.middleware import HttpResponseMiddleware
from gateway.tests.utils.base import GatewayBatchesTestCase
from gateway.views import batch as batch_view
from gateway.views import (
    batch_cancel,
    batches,
    chat_completions,
    completions,
    create_response,
    embeddings,
    file,
    files,
    get_response_input_items,
    image_generation,
    transcriptions,
    vector_store,
    vector_store_file,
    vector_store_file_batch,
    vector_store_file_batches,
    vector_store_files,
    vector_store_search,
    vector_stores,
)
from gateway.views import response as response_view
from gateway.views.mcp import ManagedMCPSession, mcp_server, session_manager
from gateway.views.utils import RawJsonResponse, RawStreamingResponse, register_response_in_cache
from management.models import Batch as BatchModel
from management.models import (
    BatchStatus,
    FileObject,
    Token,
    VectorStore,
    VectorStoreFile,
    VectorStoreStatus,
)

User = get_user_model()

ROOT_DIR = Path(__file__).parent.parent.parent.parent


@override_settings(
    OIDC_OP_JWKS_ENDPOINT="https://example.com/application/o/example/jwks/",
    LITELLM_ROUTER_CONFIG_FILE_PATH=Path(ROOT_DIR / "example_router_config.yaml"),
    AQUEDUCT_FILES_API_URL="https://files-api.example.com",
    AQUEDUCT_FILES_API_KEY="test_key",
)
class TestHttpResponseMiddleware(GatewayBatchesTestCase):
    """Test that HttpResponseMiddleware properly transforms raw responses from views."""

    @classmethod
    def setUpTestData(cls):
        super().setUpTestData()
        cls.factory = AsyncRequestFactory()
        cls.token_header = {"Authorization": "Bearer sk-123abc"}
        cls.token = Token.objects.first()
        cls.user = User.objects.first()
        cls.vs_obj = VectorStore.objects.create(
            id="vs-mock-123",
            token=cls.token,
            name="Test Store",
            status=VectorStoreStatus.COMPLETED,
            usage_bytes=0,
            created_at=42,
            upstream_url=settings.AQUEDUCT_FILES_API_URL,
        )
        cls.vs_id = cls.vs_obj.id

    async def _wrap_with_middleware(self, view_func):
        """Wrap a view function with the necessary middleware and HttpResponseMiddleware."""

        async def get_response(request):
            # Manually populate resolver_match so URL kwargs are available to the view
            resolver_match = resolve(request.path)
            request.resolver_match = resolver_match
            return await view_func(request, *resolver_match.args, **resolver_match.kwargs)

        wrapped = get_response
        wrapped = HttpResponseMiddleware(wrapped)
        wrapped = AuthenticationMiddleware(wrapped)
        return SessionMiddleware(wrapped)

    async def test_completions_returns_raw_response(self):
        url = reverse("gateway:completions")
        payload = {"model": "gpt-4.1-nano", "prompt": "Hello"}
        request = self.factory.post(
            url, data=json.dumps(payload), content_type="application/json", **self.token_header
        )

        middleware = await self._wrap_with_middleware(completions)
        response = await middleware(request)

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertIsInstance(response, RawJsonResponse)

    async def test_completions_streaming_returns_raw_response(self):
        url = reverse("gateway:completions")
        payload = {"model": "gpt-4.1-nano", "prompt": "Hello", "stream": True}
        request = self.factory.post(
            url, data=json.dumps(payload), content_type="application/json", **self.token_header
        )

        middleware = await self._wrap_with_middleware(completions)
        response = await middleware(request)

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertIsInstance(response, RawStreamingResponse)

    async def test_chat_completions_returns_raw_response(self):
        url = reverse("gateway:chat_completions")
        payload = {
            "model": "gpt-4.1-nano",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ],
        }
        request = self.factory.post(
            url, data=json.dumps(payload), content_type="application/json", **self.token_header
        )

        middleware = await self._wrap_with_middleware(chat_completions)
        response = await middleware(request)

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertIsInstance(response, RawJsonResponse)

    async def test_chat_completions_streaming_returns_raw_response(self):
        url = reverse("gateway:chat_completions")
        payload = {
            "model": "gpt-4.1-nano",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ],
            "stream": True,
        }
        request = self.factory.post(
            url, data=json.dumps(payload), content_type="application/json", **self.token_header
        )

        middleware = await self._wrap_with_middleware(chat_completions)
        response = await middleware(request)

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertIsInstance(response, RawStreamingResponse)

    async def test_embeddings_returns_raw_response(self):
        url = reverse("gateway:embeddings")
        payload = {
            "model": "gpt-4.1-nano",
            "input": ["The quick brown fox jumps over the lazy dog."],
        }
        request = self.factory.post(
            url, data=json.dumps(payload), content_type="application/json", **self.token_header
        )

        middleware = await self._wrap_with_middleware(embeddings)
        response = await middleware(request)

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertIsInstance(response, RawJsonResponse)

    async def test_transcriptions_returns_raw_response(self):
        url = reverse("gateway:transcriptions")
        file = SimpleUploadedFile("test.oga", b"", content_type="audio/ogg")
        request = self.factory.post(
            url, data={"file": file, "model": "whisper-1"}, **self.token_header
        )

        middleware = await self._wrap_with_middleware(transcriptions)
        response = await middleware(request)

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertIsInstance(response, RawJsonResponse)

    async def test_transcriptions_streaming_returns_raw_response(self):
        url = reverse("gateway:transcriptions")
        file = SimpleUploadedFile("test.oga", b"", content_type="audio/ogg")
        request = self.factory.post(
            url, data={"file": file, "model": "whisper-1", "stream": "true"}, **self.token_header
        )

        middleware = await self._wrap_with_middleware(transcriptions)
        response = await middleware(request)

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertIsInstance(response, RawStreamingResponse)

    async def test_image_generation_returns_raw_response(self):
        url = reverse("gateway:image_generation")
        payload = {"model": "dall-e-2", "prompt": "A beautiful landscape", "size": "256x256"}
        request = self.factory.post(
            url, data=json.dumps(payload), content_type="application/json", **self.token_header
        )

        middleware = await self._wrap_with_middleware(image_generation)
        response = await middleware(request)

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertIsInstance(response, RawJsonResponse)

    async def test_files_returns_raw_response(self):
        url = reverse("gateway:files")
        file = SimpleUploadedFile(
            "test.jsonl", b'{"custom_id": "bar"}\n', content_type="application/jsonl"
        )
        request = self.factory.post(
            url, data={"file": file, "purpose": "batch"}, **self.token_header
        )

        middleware = await self._wrap_with_middleware(files)
        response = await middleware(request)

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertIsInstance(response, RawJsonResponse)

    async def test_file_get_delete_returns_raw_response(self):
        url = reverse("gateway:file", kwargs={"file_id": 1})
        get_request = self.factory.get(url, **self.token_header)

        middleware = await self._wrap_with_middleware(file)
        get_response = await middleware(get_request)

        self.assertIsInstance(get_response, RawJsonResponse)

        delete_request = self.factory.delete(url, **self.token_header)

        middleware = await self._wrap_with_middleware(file)
        delete_response = await middleware(delete_request)

        self.assertIsInstance(delete_response, RawJsonResponse)

    async def test_batches_returns_raw_response(self):
        file_obj = await FileObject.objects.acreate(
            id="file-remote-123",
            bytes=1,
            created_at=42,
            token=self.token,
            purpose="batch",
            upstream_url=settings.AQUEDUCT_FILES_API_URL,
        )
        url = reverse("gateway:batches")
        payload = {
            "input_file_id": file_obj.id,
            "completion_window": "24h",
            "endpoint": reverse("gateway:v1_completions"),
        }
        request = self.factory.post(
            url, data=json.dumps(payload), content_type="application/json", **self.token_header
        )

        middleware = await self._wrap_with_middleware(batches)
        response = await middleware(request)

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertIsInstance(response, RawJsonResponse)

    async def test_batch_get_returns_raw_response(self):
        # Create a batch to retrieve
        file_id = await sync_to_async(self._create_jsonl_file)(name="session-auth")
        token = await Token.objects.aget(pk=1)
        batch = await BatchModel.objects.acreate(
            completion_window="24h",
            created_at=1773058900,
            endpoint=self.url_chat,
            id="batch-session-auth",
            input_file_id=file_id,
            status=BatchStatus.IN_PROGRESS,
            token=token,
        )

        url = reverse("gateway:batch", kwargs={"batch_id": batch.id})
        request = self.factory.get(url, **self.token_header)

        middleware = await self._wrap_with_middleware(batch_view)
        response = await middleware(request)

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertIsInstance(response, RawJsonResponse)

    async def test_batch_cancel_returns_raw_response(self):
        # Create a batch to cancel
        file_id = await sync_to_async(self._create_jsonl_file)(name="session-auth")
        token = await Token.objects.aget(pk=1)
        batch = await BatchModel.objects.acreate(
            completion_window="24h",
            created_at=1773058900,
            endpoint=self.url_chat,
            id="batch-session-auth",
            input_file_id=file_id,
            status=BatchStatus.IN_PROGRESS,
            token=token,
        )

        url = reverse("gateway:batch_cancel", kwargs={"batch_id": batch.id})
        request = self.factory.post(url, **self.token_header)

        middleware = await self._wrap_with_middleware(batch_cancel)
        response = await middleware(request)

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertIsInstance(response, RawJsonResponse)

    async def test_create_response_returns_raw_response(self):
        url = reverse("gateway:responses")
        payload = {
            "model": "gpt-4.1-nano",
            "input": [{"role": "user", "content": "Hello"}],
            "max_output_tokens": 50,
        }
        request = self.factory.post(
            url, data=json.dumps(payload), content_type="application/json", **self.token_header
        )

        middleware = await self._wrap_with_middleware(create_response)
        response = await middleware(request)

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertIsInstance(response, RawJsonResponse)

    async def test_create_response_streaming_returns_raw_response(self):
        url = reverse("gateway:responses")
        payload = {
            "model": "gpt-4.1-nano",
            "input": [{"role": "user", "content": "Hello"}],
            "max_output_tokens": 50,
            "stream": True,
        }
        request = self.factory.post(
            url, data=json.dumps(payload), content_type="application/json", **self.token_header
        )

        middleware = await self._wrap_with_middleware(create_response)
        response = await middleware(request)

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertIsInstance(response, RawStreamingResponse)

    async def test_response_get_delete_returns_raw_response(self):
        response_id = "resp_test123"
        register_response_in_cache(response_id, model=self.model, email="me@example.com")

        url = reverse("gateway:response", kwargs={"response_id": response_id})
        request = self.factory.get(url, **self.token_header)

        middleware = await self._wrap_with_middleware(response_view)
        response = await middleware(request)

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertIsInstance(response, RawJsonResponse)

        request = self.factory.delete(url, **self.token_header)

        middleware = await self._wrap_with_middleware(response_view)
        response = await middleware(request)

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertIsInstance(response, RawJsonResponse)

    async def test_get_response_input_items_returns_raw_response(self):
        response_id = "resp_test123"
        register_response_in_cache(response_id, model=self.model, email="me@example.com")

        url = reverse("gateway:response_input_items", kwargs={"response_id": response_id})
        request = self.factory.get(url, **self.token_header)

        middleware = await self._wrap_with_middleware(get_response_input_items)
        response = await middleware(request)

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertIsInstance(response, RawJsonResponse)

    @override_settings(MCP_ENABLE_DNS_REBINDING_PROTECTION=False)
    @patch("gateway.views.mcp.get_mcp_config")
    async def test_mcp_server_returns_raw_response(self, mock_get_mcp_config):
        mock_get_mcp_config.return_value = {"test_mcp_server": {"url": self.mock_server.base_url}}

        url = reverse("gateway:mcp_server", kwargs={"name": "test_mcp_server"})
        payload = {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        }
        request = self.factory.post(
            url, data=json.dumps(payload), content_type="application/json", **self.token_header
        )

        middleware = await self._wrap_with_middleware(mcp_server)
        response = await middleware(request)

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertIsInstance(response, RawJsonResponse)

    @override_settings(MCP_ENABLE_DNS_REBINDING_PROTECTION=False)
    @patch("gateway.views.mcp.get_mcp_config")
    @patch.object(session_manager, "get_session")
    async def test_mcp_server_get_returns_raw_response(self, mock_get_session, mock_get_mcp_config):
        mock_get_mcp_config.return_value = {"test_mcp_server": {"url": self.mock_server.base_url}}
        session_id = "1868a90c"
        mock_session = Mock(spec=ManagedMCPSession, session_id=session_id, terminated=False)
        mock_get_session.return_value = mock_session

        url = reverse("gateway:mcp_server", kwargs={"name": "test_mcp_server"})

        # with patch.object(session_manager, "_sessions", return_value={session_id: mock_session}):
        request = self.factory.get(
            url,
            content_type="application/json",
            headers={**self.token_header, "Mcp-Session-Id": session_id},
        )

        middleware = await self._wrap_with_middleware(mcp_server)
        response = await middleware(request)

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertIsInstance(response, RawStreamingResponse)

    async def test_vector_stores_returns_raw_response(self):
        url = reverse("gateway:vector_stores")
        payload = {"name": "Test Store"}
        request = self.factory.post(
            url, data=json.dumps(payload), content_type="application/json", **self.token_header
        )

        middleware = await self._wrap_with_middleware(vector_stores)
        response = await middleware(request)

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertIsInstance(response, RawJsonResponse)

    async def test_vector_store_get_returns_raw_response(self):
        url = reverse("gateway:vector_store", kwargs={"vector_store_id": self.vs_id})
        request = self.factory.get(url, **self.token_header)

        middleware = await self._wrap_with_middleware(vector_store)
        response = await middleware(request)

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertIsInstance(response, RawJsonResponse)

    async def test_vector_store_post_returns_raw_response(self):
        url = reverse("gateway:vector_store", kwargs={"vector_store_id": self.vs_id})
        payload = {"name": "Updated Test Store"}
        request = self.factory.post(
            url, data=json.dumps(payload), content_type="application/json", **self.token_header
        )

        middleware = await self._wrap_with_middleware(vector_store)
        response = await middleware(request)

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertIsInstance(response, RawJsonResponse)

    async def test_vector_store_delete_returns_raw_response(self):
        # Create a new vector store which can be deleted
        vs_obj = await VectorStore.objects.acreate(
            id="vs-to-delete",
            token=self.token,
            name="Test Store",
            status=VectorStoreStatus.COMPLETED,
            usage_bytes=0,
            created_at=42,
            upstream_url=settings.AQUEDUCT_FILES_API_URL,
        )

        url = reverse("gateway:vector_store", kwargs={"vector_store_id": vs_obj.id})
        request = self.factory.delete(url, **self.token_header)

        middleware = await self._wrap_with_middleware(vector_store)
        response = await middleware(request)

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertIsInstance(response, RawJsonResponse)

    async def test_vector_store_search_returns_raw_response(self):
        url = reverse("gateway:vector_store_search", kwargs={"vector_store_id": self.vs_id})
        payload = {"query": "test query"}
        request = self.factory.post(
            url, data=json.dumps(payload), content_type="application/json", **self.token_header
        )

        middleware = await self._wrap_with_middleware(vector_store_search)
        response = await middleware(request)

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertIsInstance(response, RawJsonResponse)

    async def test_vector_store_files_returns_raw_response(self):
        file_obj = await FileObject.objects.acreate(
            id="file-remote-123",
            bytes=1,
            created_at=42,
            token=self.token,
            purpose="user_data",
            upstream_url="https://files-api.example.com",
        )

        url = reverse("gateway:vector_store_files", kwargs={"vector_store_id": self.vs_id})
        payload = {"file_id": file_obj.id}
        request = self.factory.post(
            url, data=json.dumps(payload), content_type="application/json", **self.token_header
        )

        middleware = await self._wrap_with_middleware(vector_store_files)
        response = await middleware(request)

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertIsInstance(response, RawJsonResponse)

    async def test_vector_store_file_get_returns_raw_response(self):
        file_obj = await FileObject.objects.acreate(
            id="file-remote-999",
            bytes=1,
            created_at=42,
            token=self.token,
            purpose="user_data",
            upstream_url=settings.AQUEDUCT_FILES_API_URL,
        )
        vsf = await VectorStoreFile.objects.acreate(
            id="vsf-mock-42",
            vector_store=self.vs_obj,
            file_obj=file_obj,
            status="in_progress",
            created_at=9999,
        )
        url = reverse(
            "gateway:vector_store_file", kwargs={"vector_store_id": self.vs_id, "file_id": vsf.id}
        )
        request = self.factory.get(url, **self.token_header)

        middleware = await self._wrap_with_middleware(vector_store_file)
        response = await middleware(request)

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertIsInstance(response, RawJsonResponse)

    async def test_vector_store_file_post_returns_raw_response(self):
        file_obj = await FileObject.objects.acreate(
            id="file-remote-999",
            bytes=1,
            created_at=42,
            token=self.token,
            purpose="user_data",
            upstream_url=settings.AQUEDUCT_FILES_API_URL,
        )
        vsf = await VectorStoreFile.objects.acreate(
            id="vsf-mock-42",
            vector_store=self.vs_obj,
            file_obj=file_obj,
            status="in_progress",
            created_at=9999,
        )
        url = reverse(
            "gateway:vector_store_file", kwargs={"vector_store_id": self.vs_id, "file_id": vsf.id}
        )
        payload = {"attributes": {"key": "value"}}
        request = self.factory.post(
            url, data=json.dumps(payload), content_type="application/json", **self.token_header
        )

        middleware = await self._wrap_with_middleware(vector_store_file)
        response = await middleware(request)

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertIsInstance(response, RawJsonResponse)

    async def test_vector_store_file_delete_returns_raw_response(self):
        file_obj = await FileObject.objects.acreate(
            id="file-remote-999",
            bytes=1,
            created_at=42,
            token=self.token,
            purpose="user_data",
            upstream_url=settings.AQUEDUCT_FILES_API_URL,
        )
        vsf = await VectorStoreFile.objects.acreate(
            id="vsf-mock-42",
            vector_store=self.vs_obj,
            file_obj=file_obj,
            status="in_progress",
            created_at=9999,
        )
        url = reverse(
            "gateway:vector_store_file", kwargs={"vector_store_id": self.vs_id, "file_id": vsf.id}
        )
        request = self.factory.delete(url, **self.token_header)

        middleware = await self._wrap_with_middleware(vector_store_file)
        response = await middleware(request)

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertIsInstance(response, RawJsonResponse)

    async def test_vector_store_file_content_get_returns_raw_response(self):
        file_obj = await FileObject.objects.acreate(
            id="file-remote-999",
            bytes=1,
            created_at=42,
            token=self.token,
            purpose="user_data",
            upstream_url=settings.AQUEDUCT_FILES_API_URL,
        )
        vsf = await VectorStoreFile.objects.acreate(
            id="vsf-mock-42",
            vector_store=self.vs_obj,
            file_obj=file_obj,
            status="in_progress",
            created_at=9999,
        )
        url = reverse(
            "gateway:vector_store_file_content",
            kwargs={"vector_store_id": self.vs_id, "file_id": vsf.id},
        )
        request = self.factory.get(url, **self.token_header)

        middleware = await self._wrap_with_middleware(vector_store_file)
        response = await middleware(request)

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertIsInstance(response, RawJsonResponse)

    async def test_vector_store_file_batches_returns_raw_response(self):
        file_obj = await FileObject.objects.acreate(
            id="file-remote-123",
            bytes=1,
            created_at=42,
            token=self.token,
            purpose="user_data",
            upstream_url="https://files-api.example.com",
        )
        url = reverse("gateway:vector_store_file_batches", kwargs={"vector_store_id": self.vs_id})
        payload = {"file_ids": [file_obj.id]}
        request = self.factory.post(
            url, data=json.dumps(payload), content_type="application/json", **self.token_header
        )

        middleware = await self._wrap_with_middleware(vector_store_file_batches)
        response = await middleware(request)

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertIsInstance(response, RawJsonResponse)

    async def test_vector_store_file_batch_get_returns_raw_response(self):
        # Create a batch first
        file = await FileObject.objects.acreate(
            id="file-mock-123",
            bytes=100,
            created_at=1234,
            filename="test.txt",
            purpose="user_data",
            token=self.token,
            upstream_url=settings.AQUEDUCT_FILES_API_URL,
        )
        batches_url = reverse(
            "gateway:vector_store_file_batches", kwargs={"vector_store_id": self.vs_id}
        )
        resp = await self.async_client.post(
            batches_url,
            data=json.dumps({"file_ids": [file.id]}),
            headers=self.headers,
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200, f"Create batch failed: {resp.json()}")
        batch_id = resp.json()["id"]

        url = reverse(
            "gateway:vector_store_file_batch",
            kwargs={"vector_store_id": self.vs_id, "batch_id": batch_id},
        )
        request = self.factory.get(url, **self.token_header)

        middleware = await self._wrap_with_middleware(vector_store_file_batch)
        response = await middleware(request)

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertIsInstance(response, RawJsonResponse)
