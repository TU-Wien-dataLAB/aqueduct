import json
from http import HTTPStatus
from pathlib import Path
from unittest.mock import AsyncMock, patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse
from openai.types.audio import Transcription

from gateway.tests.utils import _build_chat_headers
from gateway.tests.utils.base import get_mock_router
from management.models import Request

ROOT_DIR = Path(__file__).parent.parent.parent.parent


@override_settings(
    OIDC_OP_JWKS_ENDPOINT="https://example.com/application/o/example/jwks/",
    LITELLM_ROUTER_CONFIG_FILE_PATH=Path(ROOT_DIR / "example_router_config.yaml"),
)
class TestUserId(TestCase):
    fixtures = ["gateway_data.json"]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.AQUEDUCT_ACCESS_TOKEN = "sk-123abc"
        cls.model = "test-model"
        cls.headers = _build_chat_headers(cls.AQUEDUCT_ACCESS_TOKEN)
        cls.multipart_headers = {"Authorization": f"Bearer {cls.AQUEDUCT_ACCESS_TOKEN}"}

    # def test_batches_with_user_id(self):
    #     token = Token.objects.first()
    #     file_obj = FileObject.objects.create(bytes=1, created_at=42, token=token, purpose="batch")
    #
    #     url = reverse("gateway:batches")
    #     user_id = "testuser"
    #     payload = {
    #         "input_file_id": file_obj.id,
    #         "completion_window": "24h",
    #         "endpoint": reverse("gateway:v1_completions"), "user_id": user_id}
    #     # with patch("gateway.views.batches.TypeAdapter"), patch("gateway.views.batches.BatchService.create_batch"):
    #     resp = self.client.post(
    #         url, data=json.dumps(payload), headers=self.headers, content_type="application/json"
    #     )
    #
    #     self.assertEqual(resp.status_code, HTTPStatus.OK, resp.json())
    #     req = Request.objects.last()
    #     TODO: the batches endpoints don't use the `parse_body` decorator!
    #     self.assertEqual(req.user_id, user_id)

    def test_completions_with_user_id(self):
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

        payload = {
            "model": self.model,
            "prompt": "A beautiful landscape with mountains and a lake",
            "size": "256x256",
            "user_id": user_id,
        }

        with patch("gateway.views.image_generation.get_router", return_value=get_mock_router()):
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

        with patch("gateway.views.speech.get_router", return_value=get_mock_router()):
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
        mock_client = AsyncMock()
        mock_client.audio.transcriptions.create.return_value = Transcription(
            text="How much is the fish?"
        )

        with (
            patch("gateway.views.transcriptions.get_router", return_value=get_mock_router()),
            patch("gateway.views.transcriptions.get_openai_client", return_value=mock_client),
            patch(
                "litellm.get_llm_provider", return_value=("test-model", "test-provider", None, None)
            ),
        ):
            resp = self.client.post(
                url,
                {"file": file, "model": self.model, "user_id": user_id},
                headers=self.multipart_headers,
            )

        self.assertEqual(resp.status_code, HTTPStatus.OK, resp.json())
        req = Request.objects.get()
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
                with patch("gateway.views.completions.get_router", return_value=get_mock_router()):
                    resp = self.client.post(
                        url, data=data, headers=self.headers, content_type="application/json"
                    )

                self.assertEqual(resp.status_code, HTTPStatus.OK, resp.json())
                self.assertIsNotNone(Request.objects.filter(user_id=user_id))

        self.assertEqual(Request.objects.filter(user_id="42").count(), 2)
