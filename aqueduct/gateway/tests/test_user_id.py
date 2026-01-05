import json
from http import HTTPStatus
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import urlparse

import httpx
from asgiref.sync import sync_to_async
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import override_settings
from django.urls import reverse
from openai.types.audio import Transcription

from gateway.tests.utils.mcp import MCPLiveServerTestCase

ROOT_DIR = Path(__file__).parent.parent.parent.parent


@override_settings(
    OIDC_OP_JWKS_ENDPOINT="https://example.com/application/o/example/jwks/",
    LITELLM_ROUTER_CONFIG_FILE_PATH=Path(ROOT_DIR / "example_router_config.yaml"),
)
class TestUserId(MCPLiveServerTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.AQUEDUCT_ACCESS_TOKEN = "sk-123abc"
        cls.model = "test-model"
        cls.multipart_headers = {"Authorization": f"Bearer {cls.AQUEDUCT_ACCESS_TOKEN}"}

    def test_batches_with_user_id(self):
        from management.models import FileObject, Request, Token

        token = Token.objects.first()
        file_obj = FileObject.objects.create(bytes=1, created_at=42, token=token, purpose="batch")

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
        from gateway.tests.utils.base import get_mock_router
        from management.models import Request

        url = reverse("gateway:completions")
        user_id = "testuser"
        payload = {"model": self.model, "prompt": "Hello", "user_id": user_id}
        with patch("gateway.views.completions.get_router", return_value=get_mock_router()):
            resp = self.client.post(
                url, data=json.dumps(payload), headers=self.headers, content_type="application/json"
            )

        self.assertEqual(resp.status_code, HTTPStatus.OK, resp.json())
        req = Request.objects.last()
        self.assertEqual(req.user_id, user_id)

    def test_chat_completions_with_user_id(self):
        from gateway.tests.utils.base import get_mock_router
        from management.models import Request

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

        with patch("gateway.views.chat_completions.get_router", return_value=get_mock_router()):
            resp = self.client.post(
                url, data=json.dumps(payload), headers=self.headers, content_type="application/json"
            )

        self.assertEqual(resp.status_code, HTTPStatus.OK, resp.json())
        req = Request.objects.get()
        self.assertEqual(req.user_id, user_id)

    def test_embeddings_with_user_id(self):
        from gateway.tests.utils.base import get_mock_router
        from management.models import Request

        url = reverse("gateway:embeddings")
        user_id = "testuser"
        payload = {
            "model": self.model,
            "input": ["The quick brown fox jumps over the lazy dog."],
            "user_id": user_id,
        }

        with patch("gateway.views.embeddings.get_router", return_value=get_mock_router()):
            resp = self.client.post(
                url, data=json.dumps(payload), headers=self.headers, content_type="application/json"
            )

        self.assertEqual(resp.status_code, HTTPStatus.OK, resp.json())
        req = Request.objects.get()
        self.assertEqual(req.user_id, user_id)

    def test_file_upload_with_user_id(self):
        from management.models import Request

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
        from management.models import Request

        url = reverse("gateway:image_generation")
        user_id = "testuser"
        img_model = "dall-e-2"

        payload = {
            "model": img_model,
            "prompt": "A beautiful landscape with mountains and a lake",
            "size": "256x256",
            "user_id": user_id,
        }

        with patch("gateway.views.utils.get_openai_client") as mock_client:
            mock_openai_client = AsyncMock()

            mock_retrieve_response = MagicMock()
            mock_retrieve_response.model_dump.return_value = {}
            mock_openai_client.images.generate.return_value = mock_retrieve_response

            mock_client.return_value = mock_openai_client

            resp = self.client.post(
                url, data=json.dumps(payload), headers=self.headers, content_type="application/json"
            )

        self.assertEqual(resp.status_code, HTTPStatus.OK, resp.json())
        req = Request.objects.get()
        self.assertEqual(req.user_id, user_id)

    async def test_mcp_with_user_id_in_body(self):
        from management.models import Request

        user_id = "testuser"
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
            "user_id": user_id,
        }
        parsed_url = urlparse(self.live_server_url)
        valid_host = parsed_url.netloc
        headers = {"Host": valid_host, **self.headers}

        async with httpx.AsyncClient() as client:
            resp = await client.post(self.mcp_url, json=payload, headers=headers)

        self.assertEqual(resp.status_code, HTTPStatus.OK, resp.json())
        req = await sync_to_async(Request.objects.get)()
        self.assertEqual(req.user_id, user_id)

    def test_speech_with_user_id_in_body(self):
        from gateway.tests.utils.base import get_mock_router
        from management.models import Request

        url = reverse("gateway:speech")
        user_id = "testuser"

        payload = {
            "model": self.model,
            "input": "Hello, this is a test of the text-to-speech system.",
            "voice": "alloy",
            "response_format": "mp3",
            "user_id": user_id,
        }

        with patch("gateway.views.speech.get_router", return_value=get_mock_router()):
            resp = self.client.post(
                url, data=json.dumps(payload), headers=self.headers, content_type="application/json"
            )

        self.assertEqual(resp.status_code, HTTPStatus.OK)
        req = Request.objects.get()
        self.assertEqual(req.user_id, user_id)

    def test_transcriptions_with_user_id_in_body(self):
        from management.models import Request

        url = reverse("gateway:transcriptions")
        user_id = "testuser"
        file = SimpleUploadedFile("test.oga", b"", content_type="audio/ogg")

        with patch("gateway.views.utils.get_openai_client") as mock_client:
            mock_openai_client = AsyncMock()
            mock_openai_client.audio.transcriptions.create.return_value = Transcription(
                text="How much is the fish?"
            )
            mock_client.return_value = mock_openai_client

            resp = self.client.post(
                url,
                {"file": file, "model": "whisper-1", "user_id": user_id},
                headers=self.multipart_headers,
            )

        self.assertEqual(resp.status_code, HTTPStatus.OK, resp.json())
        req = Request.objects.get()
        self.assertEqual(req.user_id, user_id)

    def test_valid_user_id_values(self):
        from gateway.tests.utils.base import get_mock_router
        from management.models import Request

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
                with patch("gateway.views.completions.get_router", return_value=get_mock_router()):
                    resp = self.client.post(
                        url, data=data, headers=self.headers, content_type="application/json"
                    )

                self.assertEqual(resp.status_code, HTTPStatus.OK, resp.json())
                self.assertIsNotNone(Request.objects.filter(user_id=user_id))

        self.assertEqual(Request.objects.filter(user_id="42").count(), 2)
