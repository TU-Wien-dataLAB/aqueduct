import json
from http import HTTPStatus
from pathlib import Path
from unittest.mock import patch

from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import override_settings
from django.urls import reverse
from mcp import JSONRPCResponse
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCMessage

from gateway.tests.utils.base import GatewayIntegrationTestCase
from management.models import FileObject, Request, Token, VectorStore, VectorStoreStatus

ROOT_DIR = Path(__file__).parent.parent.parent.parent


@override_settings(
    OIDC_OP_JWKS_ENDPOINT="https://example.com/application/o/example/jwks/",
    LITELLM_ROUTER_CONFIG_FILE_PATH=Path(ROOT_DIR / "example_router_config.yaml"),
    AQUEDUCT_FILES_API_URL="https://files-api.example.com",
    AQUEDUCT_FILES_API_KEY="test_key",
)
class TestUserId(GatewayIntegrationTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.multipart_headers = {"Authorization": f"Bearer {cls.AQUEDUCT_ACCESS_TOKEN}"}
        cls.token = Token.objects.first()
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

    def test_batches_with_user_id(self):
        file_obj = FileObject.objects.create(
            id="file-remote-123",
            bytes=1,
            created_at=42,
            token=self.token,
            purpose="batch",
            upstream_url=settings.AQUEDUCT_FILES_API_URL,
        )

        url = reverse("gateway:batches")
        user_id = "testuser"
        payload = {
            "input_file_id": file_obj.id,
            "completion_window": "24h",
            "endpoint": reverse("gateway:v1_completions"),
            "user_id": user_id,
        }

        resp = self.client.post(
            url, data=json.dumps(payload), headers=self.headers, content_type="application/json"
        )

        self.assertEqual(resp.status_code, HTTPStatus.OK, resp.json())
        req = Request.objects.last()
        self.assertEqual(req.user_id, user_id)

    def test_completions_with_user_id(self):
        url = reverse("gateway:completions")
        user_id = "testuser"
        payload = {"model": self.model, "prompt": "Hello", "user_id": user_id}
        resp = self.client.post(
            url, data=json.dumps(payload), headers=self.headers, content_type="application/json"
        )

        self.assertEqual(resp.status_code, HTTPStatus.OK, resp.json())
        req = Request.objects.last()
        self.assertEqual(req.user_id, user_id)

    def test_chat_completions_with_user_id(self):
        url = reverse("gateway:chat_completions")
        user_id = "testuser"
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write me a short poem!"},
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": 2,
            "user_id": user_id,
        }

        resp = self.client.post(
            url, data=json.dumps(payload), headers=self.headers, content_type="application/json"
        )

        self.assertEqual(resp.status_code, HTTPStatus.OK, resp.json())
        req = Request.objects.get()
        self.assertEqual(req.user_id, user_id)

    def test_embeddings_with_user_id(self):
        url = reverse("gateway:embeddings")
        user_id = "testuser"
        payload = {
            "model": self.model,
            "input": ["The quick brown fox jumps over the lazy dog."],
            "user_id": user_id,
        }

        resp = self.client.post(
            url, data=json.dumps(payload), headers=self.headers, content_type="application/json"
        )

        self.assertEqual(resp.status_code, HTTPStatus.OK, resp.json())
        req = Request.objects.get()
        self.assertEqual(req.user_id, user_id)

    def test_file_upload_with_user_id(self):
        url = reverse("gateway:files")
        user_id = "testuser"
        file = SimpleUploadedFile(
            "test.jsonl", b'{"custom_id": "bar"}\n', content_type="application/jsonl"
        )
        payload = {"file": file, "purpose": "batch", "user_id": user_id}
        resp = self.client.post(url, data=payload, headers=self.multipart_headers)

        self.assertEqual(resp.status_code, HTTPStatus.OK, resp.json())
        req = Request.objects.get()
        self.assertEqual(req.user_id, user_id)

    def test_image_generation_with_user_id(self):
        url = reverse("gateway:image_generation")
        user_id = "testuser"
        img_model = "dall-e-2"

        payload = {
            "model": img_model,
            "prompt": "A beautiful landscape with mountains and a lake",
            "size": "256x256",
            "user_id": user_id,
        }

        resp = self.client.post(
            url, data=json.dumps(payload), headers=self.headers, content_type="application/json"
        )

        self.assertEqual(resp.status_code, HTTPStatus.OK, resp.json())
        req = Request.objects.get()
        self.assertEqual(req.user_id, user_id)

    @override_settings(MCP_ENABLE_DNS_REBINDING_PROTECTION=False)
    @patch("gateway.views.mcp.get_mcp_config")
    def test_mcp_with_user_id_in_body(self, mock_get_mcp_config):
        mock_get_mcp_config.return_value = {"test_mcp_server": {"url": self.mock_server.base_url}}

        user_id = "testuser"
        payload = {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
            "user_id": user_id,
        }
        mcp_url = reverse("gateway:mcp_server", kwargs={"name": "test_mcp_server"})

        mock_msg = SessionMessage(
            message=JSONRPCMessage(JSONRPCResponse(jsonrpc="2.0", id=0, result={"test": "yes"}))
        )
        with patch("gateway.views.mcp.ManagedMCPSession.receive_message", return_value=mock_msg):
            resp = self.client.post(
                mcp_url,
                data=json.dumps(payload),
                headers=self.headers,
                content_type="application/json",
            )

        self.assertEqual(resp.status_code, HTTPStatus.OK)
        req = Request.objects.get()
        self.assertEqual(req.user_id, user_id)

    def test_responses_with_user_id_in_body(self):
        url = reverse("gateway:responses")
        user_id = "testuser"
        payload = {
            "model": self.model,
            "input": [{"role": "user", "content": "Hello, how are you?"}],
            "max_output_tokens": 50,
            "user_id": user_id,
        }

        resp = self.client.post(
            url, data=json.dumps(payload), headers=self.headers, content_type="application/json"
        )

        self.assertEqual(resp.status_code, HTTPStatus.OK, resp.json())
        req = Request.objects.get()
        self.assertEqual(req.user_id, user_id)

    def test_speech_with_user_id_in_body(self):
        url = reverse("gateway:speech")
        user_id = "testuser"
        payload = {
            "model": self.model,
            "input": "Hello, this is a test of the text-to-speech system.",
            "voice": "alloy",
            "response_format": "mp3",
            "user_id": user_id,
        }

        resp = self.client.post(
            url, data=json.dumps(payload), headers=self.headers, content_type="application/json"
        )

        self.assertEqual(resp.status_code, HTTPStatus.OK)
        req = Request.objects.get()
        self.assertEqual(req.user_id, user_id)

    def test_transcriptions_with_user_id_in_body(self):
        url = reverse("gateway:transcriptions")
        user_id = "testuser"
        file = SimpleUploadedFile("test.oga", b"", content_type="audio/ogg")

        resp = self.client.post(
            url,
            {"file": file, "model": "whisper-1", "user_id": user_id},
            headers=self.multipart_headers,
        )

        self.assertEqual(resp.status_code, HTTPStatus.OK, resp.json())
        req = Request.objects.get()
        self.assertEqual(req.user_id, user_id)

    def test_vector_stores_with_user_id_in_body(self):
        url = reverse("gateway:vector_stores")
        user_id = "testuser"
        payload = {"name": "Test Store", "user_id": user_id}

        resp = self.client.post(
            url, data=json.dumps(payload), headers=self.headers, content_type="application/json"
        )

        self.assertEqual(resp.status_code, HTTPStatus.OK, resp.json())
        req = Request.objects.get()
        self.assertEqual(req.user_id, user_id)

    def test_vector_store_search_with_user_id_in_body(self):
        url = reverse("gateway:vector_store_search", kwargs={"vector_store_id": self.vs_id})
        user_id = "testuser"
        payload = {"query": "test query", "user_id": user_id}

        resp = self.client.post(
            url, data=json.dumps(payload), headers=self.headers, content_type="application/json"
        )

        self.assertEqual(resp.status_code, HTTPStatus.OK, resp.json())
        req = Request.objects.filter(path__contains="search").last()
        self.assertEqual(req.user_id, user_id)

    def test_vector_store_files_with_user_id_in_body(self):
        file_obj = FileObject.objects.create(
            id="file-remote-123",
            bytes=1,
            created_at=42,
            token=self.token,
            purpose="user_data",
            upstream_url="https://files-api.example.com",
        )

        url = reverse("gateway:vector_store_files", kwargs={"vector_store_id": self.vs_id})
        user_id = "testuser"
        payload = {"file_id": file_obj.id, "user_id": user_id}

        resp = self.client.post(
            url, data=json.dumps(payload), headers=self.headers, content_type="application/json"
        )

        self.assertEqual(resp.status_code, HTTPStatus.OK, resp.json())
        req = Request.objects.filter(path__contains="vector_stores").last()
        self.assertEqual(req.user_id, user_id)

    def test_vector_store_file_batches_with_user_id_in_body(self):
        file_obj = FileObject.objects.create(
            id="file-remote-123",
            bytes=1,
            created_at=42,
            token=self.token,
            purpose="user_data",
            upstream_url="https://files-api.example.com",
        )

        url = reverse("gateway:vector_store_file_batches", kwargs={"vector_store_id": self.vs_id})
        user_id = "testuser"
        payload = {"file_ids": [file_obj.id], "user_id": user_id}

        resp = self.client.post(
            url, data=json.dumps(payload), headers=self.headers, content_type="application/json"
        )

        self.assertEqual(resp.status_code, HTTPStatus.OK, resp.json())
        req = Request.objects.filter(path__contains="vector_stores").last()
        self.assertEqual(req.user_id, user_id)

    def test_valid_user_id_values(self):
        url = reverse("gateway:completions")
        user_ids = [
            42,  # int will be converted to str
            "42",
            "Emmy Noether",
            "emmy.noether@example.com",
            "",
        ]
        for i, user_id in enumerate(user_ids):
            with self.subTest(user_id=user_id, i=i):
                data = json.dumps({"model": self.model, "prompt": "Hello", "user_id": user_id})
                resp = self.client.post(
                    url, data=data, headers=self.headers, content_type="application/json"
                )

                self.assertEqual(resp.status_code, HTTPStatus.OK, resp.json())
                self.assertIsNotNone(Request.objects.filter(user_id=user_id))

        self.assertEqual(Request.objects.filter(user_id="42").count(), 2)
